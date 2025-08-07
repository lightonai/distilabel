import random
from pathlib import Path

from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
    NoOp,
    ConcatenateBranches,
    Map,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)

from distilabel.utils.pipelines.fill_in_middle_utils import format_distiset
from distilabel.pydantics import Config
from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.configs.fill_in_middle import config, DS_PATH, PDF_ROOT, CACHE_DIR, EXCLUDE_PDFS

def get_ds(n: int) -> Dataset:
    dataset = load_from_disk(DS_PATH)
    dataset = dataset.shuffle(seed=0)
    # this way we are taking different ones from kp_retrieval, 
    # with flexibility for value of n until that is high enough they overlap
    # hacking in additional values while maintaining the order of the first 2_000_000 for getting the cache
    dataset = dataset.select(list(range(len(dataset) - n, len(dataset))) + list(range(len(dataset) - 10_000_000, len(dataset) - n)))
    dataset = dataset.map(lambda x: {'source': [x['image_filename']]}, num_proc=64)
    dataset = dataset.select_columns(['source'])
    return dataset

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 256

def run_pipeline(config: Config):
    global STAGE
    random.seed(0)
    
    stages = config.stages
    dataset = get_ds(2_000_000)
    # dataset = get_ds(16)
    dataset = utils.remove_pdfs_from_dataset(dataset, EXCLUDE_PDFS, row_to_ifn=lambda row: row['source'][0])
    dataset = utils.remove_pdfs_with_pages_(dataset, PDF_ROOT, CACHE_DIR, less_than=3, row_to_ifn=lambda row: row['source'][0])

    with Pipeline(
        name='fill_in_middle',
        description=(
            'Transcribe pages. These are used to make the task: fill in the missing page given the rest of the document'
        ),
        cache_dir=Path(CACHE_DIR) / 'fill_in_middle',
    ) as pipeline:
        ################## STAGE 0: TRANSCRIBE THE PAGE ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)

        lms = pipe_utils.make_lms(config, stage, use_cache=False)
        generate_transcribe = [
            LMGenerationTask(
                name=f"transcribe_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'transcribe_system', 'model_name': 'transcribe_model_name', 'generation': 'md'},
                # invalidate_cache=True,
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]
        filter_transcribe = FilterRows(
            name='filter_transcribe',
            cols=['md'],
            condition=utils.logical_and_filters(
                utils.not_empty_string,
                utils.generation_is_structured,
            ),
            input_batch_size=BATCH_SIZE,
        )

        ## Pipeline
        load_data >> generate_transcribe >> filter_transcribe

    distiset = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [
                    load_data,
                    *generate_transcribe,
                    filter_transcribe,
                ],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=True,
    )
    return distiset

if __name__ == '__main__':
    distiset: Dataset = run_pipeline(config)['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])  # don't need this for this pipeline
    distiset = format_distiset(distiset)
    distiset.save_to_disk(CACHE_DIR / 'fill_in_middle_ds')
