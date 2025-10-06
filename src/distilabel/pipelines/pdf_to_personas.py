from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, DatasetDict
from logging import getLogger
from itertools import accumulate
import random
from typing import Any
from functools import partial

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    LoadPydanticAsColumns,
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)
from distilabel.steps import (
    NoOp,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage
from distilabel.models.llms import OpenAILM

from distilabel.configs.lc_sft.single_page_q import (
    config,
    EXCLUDE_PDFS,
    DS_PATH,
    PDF_ROOT,
    CACHE_DIR,
    IMAGES_DS_PATH,
)

logger = getLogger(__name__)

def _resolve_path(path: str) -> str:
    return path.replace(config.path_substitution[0], config.path_substitution[1])

def get_ds(n: int, front=True) -> Dataset:
    dataset = load_from_disk(DS_PATH)
    dataset = dataset.shuffle(seed=0)
    n = min(n, len(dataset))
    dataset = dataset.select(range(n) if front else range(len(dataset) - n, len(dataset)))
    dataset = dataset.map(lambda x: {'source': [_resolve_path(x['image_filename'])]}, num_proc=1)
    # dataset = dataset.select_columns(['source', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])
    return dataset

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 256

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    
    stages = config.stages
    # dataset = get_ds(5_000_000, front=True)
    # dataset = utils.remove_pdfs_from_dataset(dataset, EXCLUDE_PDFS, row_to_ifn=lambda row: row['source'][0])
    # dataset = utils.remove_pdfs_with_pages_(dataset, PDF_ROOT, CACHE_DIR, less_than=2, more_than=336, row_to_ifn=lambda row: row['source'][0])
    dataset = get_ds(100)
    logger.info(f"Dataset size: {len(dataset)}")

    with Pipeline(
        name="pdf_to_personas",
        description="Generate personas from pages in a document.",
        cache_dir=CACHE_DIR / 'pdf_to_personas',
    ) as pipeline:
        ################## STAGE 0: GENERATE PERSONAS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)  # cols: ['source', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=False)
        generate_personas = [
            LMGenerationTask(
                name=f"persona_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'persona_system', 'model_name': 'persona_model_name'},
                use_cache=False,
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['personas', 'persona_system', 'persona_model_name', ...]
        drop_none_personas = FilterRows(  # drop rows where the persona is None (structured output failed)
            name="drop_none_personas",
            cols=['persona'],
            condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )  # cols: ['persona', ...] -> ['persona', ...]

        ## Pipeline
        (
            load_data >> data_router >> generate_personas >> drop_none_personas
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_personas, drop_none_personas],
                len(stage.available_gpus),
            )
        ),
        use_cache=False,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    global FN_TO_PAGE_COUNT, IDX_TO_IFN_IMAGES_DS
    FN_TO_PAGE_COUNT = utils.count_all_pages(PDF_ROOT, CACHE_DIR)
    IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)

    distiset, cost_tracker = run_pipeline(config)
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    pass

