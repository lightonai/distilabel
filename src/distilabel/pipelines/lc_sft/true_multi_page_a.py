from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, concatenate_datasets
import random
from logging import getLogger

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

from distilabel.configs.lc_sft.true_multi_page_a import (
    config,
    DS_PATH,
    CACHE_DIR,
    PDF_ROOT,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

STAGE = 0
BATCH_SIZE = 256


def run_pipeline(config: Config):
    global STAGE, BATCH_SIZE
    random.seed(0)

    ds_dict = load_from_disk(DS_PATH)
    hn = ds_dict['true_multi_page_short_hn']
    adj = ds_dict['true_multi_page_short_doc']
    dataset = concatenate_datasets(
        [
            utils.add_split_label_ds(hn, 'hn_short'), 
            utils.add_split_label_ds(adj, 'adj_short'),
        ]
    )

    with Pipeline(
        name=PIPELINE_NAME,
        description='Generate multi-page answers with full visual context for the true multi-page questions with 2-5 pages.',
        cache_dir=CACHE_DIR / 'true_multi_page_a',
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
                # invalidate_cache=True,
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
        invalidate_distiset=True,
    )
    return distiset, cost_tracker


if __name__ == '__main__':
    distiset, cost_tracker = run_pipeline(config)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=1,
    )

    distiset.save_to_disk(CACHE_DIR / 'true_multi_page_short_vds')
