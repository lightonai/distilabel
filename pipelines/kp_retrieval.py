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

from distilabel.utils.pipelines.kp_retrieval_utils import format_distiset
from distilabel.pydantics import Config
from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils

from distilabel.configs.kp_retrieval import config, DS_PATH, CACHE_DIR, EXCLUDE_PDFS

def slice_key(key_extraction: str, **kwargs) -> str:
    '''Take key (key_selection) as a random slice of the target to extract (key_extraction)'''
    words = key_extraction.split(' ')
    # take at least 8 words, or up to the size of the string, and up to 40% of the words in addition to 8
    start = random.randint(0, max(0, min(int(len(words) * 0.8), len(words) - 8)))
    end = random.randint(min(start + 8, len(words)), min(len(words), start + 9 + int(0.4 * len(words))))
    return {
        'key_selection': ' '.join(words[start:end])
    }

def key_and_selection_reasonable(row: dict, cols: list[str], **kwargs) -> bool:
    '''The key to search for shouldn't be the entire retrieval and the key should be in the retrieved sentence'''
    return row['key_selection'] != row['key_extraction'] and row['key_selection'] in row['key_extraction']

def get_ds(n: int) -> Dataset:
    dataset = load_from_disk(DS_PATH)
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.select(range(n))
    dataset = dataset.map(lambda x: {'source': [x['image_filename']]})
    dataset = dataset.select_columns(['source', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])
    return dataset

STAGE = 0
BATCH_SIZE = 256
'''tracks the current stage of the pipeline'''

def run_pipeline(config: Config):
    global STAGE
    random.seed(0)
    
    stages = config.stages
    # dataset = get_ds(2_000_000)
    dataset = get_ds(16)
    dataset = utils.remove_pdfs_from_dataset(dataset, EXCLUDE_PDFS)

    with Pipeline(
        name='kp_retrieval',
        description=(
            'Generate questions that ask the model to retrieve content from a page given a specific location or key.'
        ),
        cache_dir=Path(CACHE_DIR) / 'kp_retrieval',
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
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        ################## STAGE 1: GEN KEY AND POS RETRIEVAL QUESTIONS ##################
        STAGE += 1
        stage = stages[STAGE]

        lms = pipe_utils.make_lms(config, stage, use_cache=False)

        # Key Retrieval
        kr_branch = NoOp(name='kr_branch', input_batch_size=BATCH_SIZE)
        generate_key_extraction = [
            LMGenerationTask(
                name=f"key_extraction_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                # map to img_source was used when using a LM for the next step, but now it's just a function. keep for ease of use
                output_mappings={'system': 'key_extraction_system', 'model_name': 'key_extraction_model_name', 'extraction': 'key_extraction', 'source': 'img_source'},
                # invalidate_cache=True,
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
            if lm.lm_config.task_name == 'key_extraction'
        ]
        filter_key_extraction = FilterRows(
            name='filter_key_extraction',
            cols=['key_extraction'],
            condition=utils.logical_and_filters(
                # the model is good about responding with an empty string if the page is blank or requested extraction is n/a
                utils.not_empty_string,
                utils.generation_is_structured,
            ),
            input_batch_size=BATCH_SIZE,
        )
        random_key_slice = Map(
            name='random_key_slice',
            fn=slice_key,
            cols=['key_extraction'],
            output_cols=['key_selection'],
            input_batch_size=BATCH_SIZE,
            use_cache=True,
        )
        filter_key_selection = FilterRows(
            name='filter_key_selection',
            cols=['key_selection', 'key_extraction'],
            condition=utils.logical_and_filters(
                utils.not_empty_string,
                key_and_selection_reasonable,
                utils.generation_is_structured,
            ),
            input_batch_size=BATCH_SIZE,
        )

        # Pos Retrieval
        pr_branch = NoOp(name='pr_branch', cols=['source'], output_mappings={'source': 'img_source'}, input_batch_size=BATCH_SIZE)
        generate_pos_extraction = [
            LMGenerationTask(
                name=f"pos_extraction_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                input_mappings={'source': 'md'},
                output_mappings={'system': 'pos_extraction_system', 'model_name': 'pos_extraction_model_name', 'extraction': 'pos_extraction', 'source': 'md'},
                use_cache=True,
                # invalidate_cache=True,
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
            if lm.lm_config.task_name == 'pos_extraction'
        ]
        filter_pos_extraction = FilterRows(
            name='filter_pos_extraction',
            cols=['pos_extraction'],
            condition=utils.logical_and_filters(
                utils.not_empty_string,
                utils.generation_is_structured,
            ),
            input_batch_size=BATCH_SIZE,
        )

        join = ConcatenateBranches(
            name='join', 
            cols=[
                'key_extraction_system', 'pos_extraction_system',
                'key_selection', 'key_extraction', 'pos_extraction', 'md', 'img_source',
                'hard_negs_idx_img_img', 'hard_negs_idx_txt_img',
            ],
            input_batch_size=BATCH_SIZE,
        )

        img_to_source = NoOp(
            name='img_to_source',
            cols=['img_source'],
            output_mappings={'img_source': 'source'},
            input_batch_size=BATCH_SIZE,
        )

        ## Pipeline
        load_data >> generate_transcribe >> filter_transcribe >> [kr_branch, pr_branch]
        kr_branch >> generate_key_extraction >> filter_key_extraction >> random_key_slice >> filter_key_selection
        pr_branch >> generate_pos_extraction >> filter_pos_extraction
        [filter_key_selection, filter_pos_extraction] >> join >> img_to_source

    distiset = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [
                    load_data,
                    *generate_transcribe,
                    filter_transcribe,
                ],
                len(stage.available_gpus),
            ) +
            pipe_utils.steps_to_load_groups(
                [ 
                    kr_branch,
                    pr_branch,
                    *generate_key_extraction,
                    filter_key_extraction,
                    random_key_slice,
                    filter_key_selection,
                    *generate_pos_extraction,
                    filter_pos_extraction,
                    join,
                    img_to_source,
                ],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=False,
    )
    return distiset

if __name__ == '__main__':
    distiset: Dataset = run_pipeline(config)['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])  # don't need this for this pipeline
    distiset, images_ds = format_distiset(distiset)
    distiset.save_to_disk(Path(CACHE_DIR) / 'kp_retrieval_ds')
    images_ds.save_to_disk(Path(CACHE_DIR) / 'all_pdfs_images')
