from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset
import random

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

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage
from distilabel.models.llms import OpenAILM

from distilabel.configs.multi_page_answers import config

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 256

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    random.seed(0)
    
    stages = config.stages
    dataset = load_from_disk('out/mp_synthetic_data')
    assert dataset['hard_negs_short'][0]['question'] != '' and dataset['adjacent_pages_short'][0]['question'] != '', (
        'The splits must have questions, make sure to run single_page_qa.py first'
    )

    # add split labels and randomize order for hard negatives and sort for adjacent pages
    hns = utils.add_split_label(list(dataset['hard_negs_short']), 'hard_negs_short')  # track which is which for splitting at the end
    aps = utils.add_split_label(list(dataset['adjacent_pages_short']), 'adjacent_pages_short')
    hns = utils.randomize_source_order(hns)
    aps = utils.sort_adjacent_pages(aps)
    dataset = Dataset.from_list(hns + aps)

    with Pipeline(
        name="multi_page_answers",
        description="Generate answers for the single page questions using multi-page context",
        cache_dir='out/multi_page_answers',
    ) as pipeline:
        ################## STAGE 0 ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)  # cols: ['source', 'question', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage)
        generate_answers = [
            LMGenerationTask(
                name=f"answer_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question'],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
                **lm.lm_config.task_kwargs,
            ) 
            for i, lm in enumerate(lms)
        ]  # cols: ['source', 'question', ...] -> ['answer', 'answer_system', 'answer_model_name', ...]
        drop_none_answers = FilterRows(  # drop rows where the answer is None (structured output failed)
            name="drop_none_answers",
            cols=['answer'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )  # cols: ['answer', ...] -> ['answer', ...]

        ## Pipeline
        (
            load_data >> data_router >> generate_answers >> drop_none_answers
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(  # handles breaking up steps so each load_group has enough gpus
                # data_router is not included because it's not quite a step, but it does actually still run
                [load_data, *generate_answers, drop_none_answers],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    distiset, cost_tracker = run_pipeline(config)
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    # replace the source col with the original dataset to retain the original order
    dataset = load_from_disk('out/mp_synthetic_data')
    dataset = list(dataset['hard_negs_short']) + list(dataset['adjacent_pages_short'])
    # distiset = utils.replace_source_col(distiset, dataset)

    hard_negs_short = distiset.filter(lambda x: x['split'] == 'hard_negs_short').remove_columns(['split'])
    adjacent_pages_short = distiset.filter(lambda x: x['split'] == 'adjacent_pages_short').remove_columns(['split'])

    del dataset
    utils.overwrite_dataset_dict_split('out/mp_synthetic_data', 'hard_negs_short', hard_negs_short)
    utils.overwrite_dataset_dict_split('out/mp_synthetic_data', 'adjacent_pages_short', adjacent_pages_short)
