from distilabel.pipeline import Pipeline
from datasets import load_from_disk, concatenate_datasets, Dataset
import random

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

from distilabel.configs.lc_sft.plain_distill import (
    config,
    # SP_DS_PATH,
    # MP_DS_PATH,
    DS_PATH,
    IMAGES_DS_PATH,
    CACHE_DIR,
    PIPELINE_NAME,
)

STAGE = 0
BATCH_SIZE = 256


def run_pipeline(config: Config, dataset: Dataset):
    global STAGE, BATCH_SIZE
    random.seed(0)

    with Pipeline(
        name=PIPELINE_NAME,
        description='Full visual context answer from strong visual lc models',
        cache_dir=CACHE_DIR / 'plain_distill',
        disable_output_queue_timeout=True,
    ) as pipeline:
        stage = config.stages[STAGE]
        load_data = LoadDataFromDataset(name='load_data', dataset=dataset, batch_size=BATCH_SIZE)

        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in lms]
        )
        generate_answers = [
            LMGenerationTask(
                use_cache=True,
                name=f"answer_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                system_col='default_system',  # use model default system prompt
                lm_input_cols=['question'],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                input_mappings={'source': 'question_source'},
                output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]
        drop_none_answers = FilterRows(
            name='drop_none_answers',
            cols=['answer'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        load_data >> answer_router >> generate_answers >> drop_none_answers

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_answers, drop_none_answers],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True, 
    )
    return distiset, cost_tracker


if __name__ == '__main__':
    cols_to_keep = ['source', 'question', 'split', 'question_model_name']
    # sp_ds_dict = load_from_disk(SP_DS_PATH)
    # mp_ds_dict = load_from_disk(MP_DS_PATH)

    # sp_splits = [
    #     'distractors_short',
    #     'adj_short',
    #     'hn_short',
    #     'recursive_hn',
    #     'recursive_doc',
    #     'full_context_one_shot_hn',
    #     'full_context_one_shot_doc',
    #     'reasoning_hn',
    #     'reasoning_doc',
    # ]
    # mp_splits = [
    #     'true_multi_page_short_hn',
    #     'true_multi_page_short_doc',
    #     'recursive_hn',
    #     'recursive_doc',
    #     'full_context_one_shot_hn',
    #     'full_context_one_shot_doc',
    #     'reasoning_hn',
    #     'reasoning_doc',
    # ]

    # datasets: list[Dataset] = []
    # for split in sp_splits:
    #     datasets.append(utils.add_split_label_ds(sp_ds_dict[split], f'sp_{split}'))
    # for split in mp_splits:
    #     datasets.append(utils.add_split_label_ds(mp_ds_dict[split], f'mp_{split}'))
    # dataset = concatenate_datasets(datasets).select_columns(cols_to_keep)

    dataset = load_from_disk(DS_PATH)
    dataset = dataset.select_columns([col for col in cols_to_keep + ['question_source'] if col in dataset.column_names])

    distiset, cost_tracker = run_pipeline(config, dataset)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'question_source'],#, 'split'], 
        n_workers=16,
    )

    distiset.save_to_disk(CACHE_DIR / 'plain_distill_question_source_vds')
