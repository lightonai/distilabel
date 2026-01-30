from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset
import random
import logging

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    FilterRows,
    Map,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config

from distilabel.configs.lc_sft.single_page_a import (
    config,
    DS_PATH,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

STAGE = 0
BATCH_SIZE = 256

# Filled inside run_pipeline
_IDX_TO_FILENAME: dict[int, str] | None = None


def add_distant_negatives(
    source: list[str] | str,
    hard_negs_idx_img_img: list[int],
    hard_negs_idx_txt_img: list[int],
    max_distractors: int = 15,
    **kwargs,
) -> dict:
    """Append up to max_distractors random distant hard negatives to 'source' using idx→filename mapping.

    Distant = beyond the first 32 in each list
    """
    global _IDX_TO_FILENAME
    hn_img = hard_negs_idx_img_img or []
    hn_txt = hard_negs_idx_txt_img or []
    distant_negs = hn_img[32:] + hn_txt[32:]
    if len(distant_negs) == 0:
        return {'source': source}
    k = min(random.randint(1, max_distractors), len(distant_negs))
    chosen = random.sample(distant_negs, k=k)
    neg_fns = [
        _IDX_TO_FILENAME[i]
        for i in chosen
    ]
    source = source + list(set(neg_fns))
    return {'source': source}


def run_pipeline(config: Config):
    global STAGE, BATCH_SIZE, _IDX_TO_FILENAME
    random.seed(0)

    # Build index→filename map from canonical images dataset
    _IDX_TO_FILENAME = utils.get_idx_to_filename(IMAGES_DS_PATH)

    dataset = load_from_disk(DS_PATH)

    with Pipeline(
        name=PIPELINE_NAME,
        description='Generate single-page answers and add distant negatives to context',
        cache_dir=CACHE_DIR / 'single_page_a',
    ) as pipeline:
        stage = config.stages[STAGE]
        load_data = LoadDataFromDataset(name='load_data', dataset=dataset, batch_size=BATCH_SIZE)

        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in lms]
        )
        generate_answers = [
            LMGenerationTask(
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
                output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
                use_cache=True,
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

        add_distant = Map(
            name='add_distant',
            fn=add_distant_negatives,
            cols=['source', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'],
            output_cols=['source'],
            input_batch_size=BATCH_SIZE,
            # use_cache=True,
        )

        load_data >> answer_router >> generate_answers >> drop_none_answers >> add_distant

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_answers, drop_none_answers, add_distant],
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
        cols_to_keep=['answer_model_name'], 
        n_workers=4,
    )
    distiset.save_to_disk(CACHE_DIR / 'distractors_short_vds')
