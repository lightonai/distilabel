from distilabel.pipeline import Pipeline
from datasets import load_from_disk, concatenate_datasets, Dataset
from logging import getLogger

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    Split,
    Rejoin,
    NoOp,
    Map,
    FilterRows,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config

from distilabel.configs.lc_sft.full_context_one_shot_a import (
    config,
    SP_DS_PATH,
    MP_DS_PATH,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

STAGE = 0
BATCH_SIZE = 256


def _combine_transcriptions(md: list[str], **kwargs) -> dict:
    md = [f'Page {i}:\n{t.strip()}' for i, t in enumerate(md)]
    return {'combined_md': '\n\n'.join(md)}

def run_pipeline(config: Config, dataset: Dataset, pipeline_name: str):
    global STAGE, BATCH_SIZE
 
    with Pipeline(
        name=pipeline_name,
        description='Transcribe each page, then use entire text to answer the question with strong LC models.',
        cache_dir=CACHE_DIR / pipeline_name,
        disable_output_queue_timeout=True,  # qwen 3 thinking takes forever, disable timeout
    ) as pipeline:
        # ---------------------- Stage 0: transcribe each page ----------------------
        STAGE = 0
        stage0 = config.stages[STAGE]

        load_data = LoadDataFromDataset(name='load_data', dataset=dataset, batch_size=BATCH_SIZE)

        # Chunk source into 1-page chunks per row
        split_chunks = Split(
            name='split_chunks',
            input_col='source',
            chunk_size=1,
            input_batch_size=BATCH_SIZE,
        )

        evidence_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage0.lm_configs]
        )

        lms0 = pipe_utils.make_lms(config, stage0, use_cache=True)
        transcribe_pages = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'transcribe_pages_{i}',
                stage=stage0,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'generation': 'md', 'system': 'transcribe_system', 'model_name': 'transcribe_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms0)
        ]  # cols: ['source', 'question', ...] -> ['md', 'transcribe_system', 'transcribe_model_name', ...]

        filter_transcriptions = FilterRows(
            name='filter_transcriptions',
            cols=['md'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
            use_cache=True,
            # invalidate_cache=True,
        )

        # Rejoin all chunks for each row (global step), restoring original source
        rejoin_chunks = Rejoin(
            name='rejoin_chunks',
            input_col='source',
            duplicates_cols=['question', 'question_model_name', 'split', 'transcribe_system'],
            input_batch_size=BATCH_SIZE,
        )

        # Combine md text from chunks
        combine_transcriptions = Map(
            name='combine_transcriptions',
            fn=_combine_transcriptions,
            cols=['md', 'source'],
            output_cols=['combined_md'],
            input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'page_source'},
        )  # cols: ['source', 'md'] -> ['page_source', 'combined_md']

        # Prepare inputs for stage 1: preserve original pages as page_source and set combined md as source
        set_md_as_source = NoOp(
            name='set_md_as_source',
            cols=['source', 'combined_md'],
            output_mappings={'combined_md': 'source'},
            input_batch_size=BATCH_SIZE,
        )  # cols: ['combined_md'] -> ['source']

        # ---------------------- Stage 1: answer using combined md ----------------------
        STAGE = 1
        stage1 = config.stages[STAGE]

        answer_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage1.lm_configs],
            # invalidate_cache=True,
        )

        lms1 = pipe_utils.make_lms(config, stage1, use_cache=True)
        generate_answers = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache='Qwen3' not in lm.lm_config.path,
                name=f'answer_generation_{i}',
                stage=stage1,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question'],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={
                    'system': 'answer_system', 
                    'model_name': 'answer_model_name', 
                    'generation': 'answer',
                    'source': 'md',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms1)
        ]  # cols: ['source', 'question', ...] -> ['md', 'answer', 'answer_system', 'answer_model_name', ...]

        # Restore original pages as source for output
        restore_pages = NoOp(
            name='restore_pages',
            cols=['page_source'],
            output_mappings={'page_source': 'source'},
            input_batch_size=BATCH_SIZE,
        )  # cols: ['page_source'] -> ['source']

        filter_answers = FilterRows(
            name='filter_answers',
            cols=['answer'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        # ---------------------- Pipeline ----------------------
        (
            load_data >> split_chunks >> evidence_router >> transcribe_pages >> filter_transcriptions >> rejoin_chunks >> combine_transcriptions 
            >> set_md_as_source >> answer_router >> generate_answers >> restore_pages >> filter_answers
        )

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, split_chunks, *transcribe_pages, filter_transcriptions],
                len(config.stages[0].available_gpus),
            )
            + [[rejoin_chunks.name]]  # global step on its own
            + pipe_utils.steps_to_load_groups(
                [combine_transcriptions, set_md_as_source, *generate_answers, restore_pages, filter_answers],
                len(config.stages[1].available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=True,
    )
    return distiset, cost_tracker


if __name__ == '__main__':
    cols_to_keep = ['source', 'question', 'split', 'question_model_name']
    sp_ds_dict = load_from_disk(SP_DS_PATH)
    mp_ds_dict = load_from_disk(MP_DS_PATH)

    sp_splits = [
        'full_context_one_shot_hn',
        'full_context_one_shot_doc',
    ]
    mp_splits = [
        'full_context_one_shot_hn',
        'full_context_one_shot_doc',
    ]

    datasets: list[Dataset] = []
    for split in sp_splits:
        datasets.append(utils.add_split_label_ds(sp_ds_dict[split], f'sp_{split}'))
    for split in mp_splits:
        datasets.append(utils.add_split_label_ds(mp_ds_dict[split], f'mp_{split}'))
    dataset = concatenate_datasets(datasets)

    distiset, cost_tracker = run_pipeline(config, dataset, PIPELINE_NAME)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=16,
    )

    distiset.save_to_disk(CACHE_DIR / 'full_context_one_shot_vds')
