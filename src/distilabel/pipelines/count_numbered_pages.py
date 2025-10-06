import random
import os

from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDataset,
    FilterRows,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config
from distilabel.models.llms import OpenAILM

from distilabel.configs.count_numbered_pages import config

STAGE = 0
'''tracks the current stage of the pipeline'''

def run_pipeline(config: Config, dataset: Dataset):
    global STAGE
    random.seed(0)
    
    stages = config.stages

    dataset = dataset.select(range(1024))
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'image_filename'])
    dataset = dataset.rename_column('image_filename', 'source')
    dataset = dataset.map(lambda x: {'source': [x['source']]}, num_proc=16)

    with Pipeline(
        name="count_numbered_pages",
        description="Use a LM to find pages with visible page numbers.",
        cache_dir='out/count_numbered_pages',

    ) as pipeline:
        ################## STAGE 0 ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=256)
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=False)
        label_pages = [
            LMGenerationTask(
                name=f"label_pages_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=256,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'page_number_system', 'model_name': 'page_number_model_name'},
                use_cache=True,
                invalidate_cache=True,
                **lm.lm_config.task_kwargs,
            ) 
            for i, lm in enumerate(lms)
        ]
        filter_pages = FilterRows(
            name="filter_pages",
            cols=['is_page_number_visible'],
            condition=utils.generation_is_structured,
            input_batch_size=256,
        )

        ## Pipeline
        (
            load_data >> data_router >> label_pages >> filter_pages
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *label_pages, filter_pages],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    dataset = load_from_disk('uns/out/unshuffle/images')
    distiset, cost_tracker = run_pipeline(config, dataset)
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])
    
    visible = sum(distiset['is_page_number_visible']) / len(distiset)
    print(f"Ratio of pages with visible page numbers: {visible}")

    pages_with_numbers = distiset.filter(lambda row: row['is_page_number_visible'])
    pass

