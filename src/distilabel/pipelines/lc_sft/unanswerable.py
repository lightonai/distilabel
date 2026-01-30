from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
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

from distilabel.pipelines.lc_sft.single_page_q import (
    _add_pages_hn,
    _add_pages_doc,
    _add_adjacent_pages,
    _doc_eligible_batched,
)
from distilabel.pipelines.lc_sft.true_multi_page_q import (
    build_seeds,
)

from distilabel.configs.lc_sft.unanswerable import (
    config,
    EXCLUDE_PDFS,
    DS_PATH,
    PDF_ROOT,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

SEED = 7439
random.seed(SEED)

TARGET_COUNTS = {
    # 'true_multi_page_short_hn': 16,
    # 'true_multi_page_short_doc': 32,
    'recursive_hn': 384,
    'recursive_doc': 2688,
    # 'full_context_one_shot_hn': 16,
    # 'full_context_one_shot_doc': 32,
    # 'reasoning_hn': 16,
    # 'reasoning_doc': 32,
}

_resolve_path = partial(utils.resolve_path, path_substitution=config.path_substitution)

def get_ds(n: int, seed: int, use_source: bool = True, allow_doc_reuse: bool = False) -> Dataset:
    ds = load_from_disk(DS_PATH)
    ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    ds = ds.select(range(n))
    ds = ds.select_columns(['image_filename', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])

    # if doc reuse is not allowed, prune the dataset to only include one page from each doc (first occurrence)
    if not allow_doc_reuse:
        ds = utils.take_n_first_doc_occurrences(ds, _resolve_path)

    # Exclude benchmark PDFs
    ds = utils.remove_pdfs_from_dataset(
        ds,
        EXCLUDE_PDFS, 
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
        num_proc=1,
    )
    ds = utils.remove_pdfs_with_pages_(
        ds,
        PDF_ROOT,
        CACHE_DIR,
        more_than=336,
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
        num_proc=1,
    )
    if use_source:
        ds = ds.map(lambda x: {'source': [_resolve_path(x['image_filename'])]}, num_proc=1)
    return ds

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 64

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    
    stages = config.stages
    # source is 1 page only
    dataset = get_ds(32, SEED)

    # source is 2-5 pages for multi-page questions
    base_ds = get_ds(sum(TARGET_COUNTS.values()) * 2, SEED + 1, use_source=False)
    seeds = build_seeds(base_ds, TARGET_COUNTS)
    seed_ds = Dataset.from_list(seeds)

    dataset = concatenate_datasets([dataset, seed_ds])

    with Pipeline(
        name=PIPELINE_NAME,
        description="Ask the model to generate hallucination provoking questions and ideal answers",
        cache_dir=CACHE_DIR / 'unanswerable',
    ) as pipeline:
        ################## STAGE 0: GENERATE QA ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)  # cols: ['source', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=True, invalidate_cache=True)
        generate_qa = [
            LMGenerationTask(
                use_cache=True,
                invalidate_cache=True,
                name=f"qa_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'qa_system', 'model_name': 'qa_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['question', 'answer', 'key_ideas', 'key_details', 'qa_system', 'qa_model_name', ...]
        drop_none_qa = FilterRows(  # drop rows where the question is None (structured output failed)
            name="drop_none_qa",
            cols=['question', 'answer'],
            condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )  # cols: ['question', 'answer', ...] -> ['question', 'answer', ...]

        ## Pipeline
        (
            load_data >> data_router >> generate_qa >> drop_none_qa
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_qa, drop_none_qa],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    global FN_TO_PAGE_COUNT, IDX_TO_IFN_IMAGES_DS
    FN_TO_PAGE_COUNT = utils.count_all_pages(PDF_ROOT, CACHE_DIR)
    IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)

    distiset, cost_tracker = run_pipeline(config)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    split_names = [
        # 'distractors_short',
        # 'adj_short',
        # 'hn_short',
        'recursive_hn',
        'recursive_doc',
        # 'full_context_one_shot_hn',
        # 'full_context_one_shot_doc',
        # 'reasoning_hn',
        # 'reasoning_doc',
    ]
    split_sizes = [10] * len(split_names)
    ratio_to_fulfill = len(distiset) / sum(split_sizes)
    print(f"ratio of split_sizes that can be fulfilled: {ratio_to_fulfill}")
    split_sizes = [int(r * ratio_to_fulfill) for r in split_sizes]
    ds = DatasetDict()
    for split_name, start, end in zip(split_names, accumulate([0] + split_sizes[:-1]), accumulate(split_sizes)):
        ds[split_name] = distiset.select(range(start, end))
        ds[split_name] = utils.add_split_label_ds(ds[split_name], split_name)
        if split_name == 'distractors_short':
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(_add_pages_hn, max_total=5, after_k=32, top_k=256, idx_to_ifn_images_ds=IDX_TO_IFN_IMAGES_DS)), batched=True, num_proc=1)
            )
        elif split_name == 'adj_short':
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(_add_adjacent_pages, max_adjacent=4, fn_to_page_count=FN_TO_PAGE_COUNT)), batched=True, num_proc=1)
            )
        elif split_name == 'hn_short':
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(_add_pages_hn, max_total=5, top_k=32, idx_to_ifn_images_ds=IDX_TO_IFN_IMAGES_DS)), batched=True, num_proc=1)
            )
        elif 'short' not in split_name:
            if split_name.endswith('hn'):
                ds[split_name] = (
                    ds[split_name]
                    .map(utils.hf_batched(partial(_add_pages_hn, max_total=64, top_k=32, idx_to_ifn_images_ds=IDX_TO_IFN_IMAGES_DS)), batched=True, num_proc=1)
                )
            elif split_name.endswith('doc'):
                reasoning = 'reasoning' in split_name
                ds[split_name] = (
                    ds[split_name]
                    .filter(partial(_doc_eligible_batched, fn_to_page_count=FN_TO_PAGE_COUNT), batched=True, num_proc=1)
                    .map(utils.hf_batched(partial(_add_pages_doc, reasoning=reasoning, fn_to_page_count=FN_TO_PAGE_COUNT)), batched=True, num_proc=1)
                )
    ds = concatenate_datasets(ds.values())
    ds = ds.rename_column('qa_model_name', 'answer_model_name')
    ds = utils.format_distiset(
        ds, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=1,
    )
    ds.save_to_disk(CACHE_DIR / 'unanswerable_3k_vds')

