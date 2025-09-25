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

def _resolve_path(path: str) -> str:
    return path.replace(config.path_substitution[0], config.path_substitution[1])

def get_ds(n: int, front=True) -> Dataset:
    dataset = load_from_disk(DS_PATH)
    dataset = dataset.shuffle(seed=0)
    n = min(n, len(dataset))
    dataset = dataset.select(range(n) if front else range(len(dataset) - n, len(dataset)))
    dataset = dataset.map(lambda x: {'source': [_resolve_path(x['image_filename'])]}, num_proc=32)
    dataset = dataset.select_columns(['source', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])
    return dataset

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 128

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    
    stages = config.stages
    # dataset = get_ds(5_000_000, front=True)
    # dataset = get_ds(20_000, front=True)
    dataset = get_ds(4_000 * 2, front=True)
    dataset = utils.remove_pdfs_from_dataset(dataset, EXCLUDE_PDFS, row_to_ifn=lambda row: row['source'][0])
    dataset = utils.remove_pdfs_with_pages_(dataset, PDF_ROOT, CACHE_DIR, less_than=2, more_than=336, row_to_ifn=lambda row: row['source'][0])
    # dataset = get_ds(100)

    with Pipeline(
        name="single_page_q",
        description="Load mp synthetic data, sample system prompts and generate questions and answers",
        cache_dir=CACHE_DIR / 'single_page_q',
    ) as pipeline:
        ################## STAGE 0: GENERATE QUESTIONS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)  # cols: ['source', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        generate_questions = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f"question_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'question_system', 'model_name': 'question_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['questions', 'key_ideas', 'key_details', 'question_system', 'question_model_name', ...]
        questions_to_rows = ListToRows(  # expand the generated list of questions into separate rows
            name="questions_to_rows",
            input_col='questions',
            sample_n=1,  # only taking one question per page
            input_batch_size=BATCH_SIZE,
            output_mappings={'questions': 'question'},
            resources=StepResources(replicas=1),
        )  # cols: ['questions', ...] -> ['question', ...]
        drop_none_questions = FilterRows(  # drop rows where the question is None (structured output failed)
            name="drop_none_questions",
            cols=['question'],
            condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )  # cols: ['question', ...] -> ['question', ...]

        ## Pipeline
        (
            load_data >> data_router >> generate_questions >> questions_to_rows >> drop_none_questions
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_questions, questions_to_rows, drop_none_questions],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=True,
    )
    return distiset, cost_tracker

def _doc_eligible_batched(batch: dict[str, list[Any]], fn_to_page_count: dict[str, int]) -> list[bool]:
    # really only the n_pages >= 5 is new, already filtered for <= 336 when getting the dataset
    # in run_pipeline
    keep: list[bool] = []
    for src in batch['source']:
        anchor = src[0]
        pdf_path = utils.pdf_name(anchor)
        n = fn_to_page_count[pdf_path]
        keep.append(5 <= n <= 336)
    return keep

def _add_pages_hn(
    row: dict[str, Any],
    max_total: int,
    top_k: int,
    after_k: int = 0,
    idx_to_ifn_images_ds: dict[int, str] = {},
) -> dict[str, Any]:
    '''
    Add a random amount of hard negatives to the source from the top_k hard negatives for each type
    up to max_total total images (including the existing source pages)

    The source order is shuffled before returning.
    '''
    source = row['source']
    remaining = max(0, max_total - len(source))
    hni = (row.get('hard_negs_idx_img_img', []) or [])[after_k:after_k + top_k]
    hnt = (row.get('hard_negs_idx_txt_img', []) or [])[after_k:after_k + top_k]
    candidates = list(dict.fromkeys(hni + hnt))

    # add a random amount up to the max_total
    n = random.randint(0, remaining)
    add = random.sample(candidates, k=min(n, len(candidates))) if n > 0 else []
    negs = [idx_to_ifn_images_ds[i] for i in add]
    new_source = source + list(dict.fromkeys(negs))
    return {'source': random.sample(new_source, k=len(new_source))}

def _add_pages_doc(
    row: dict[str, Any],
    reasoning: bool = False,
    fn_to_page_count: dict[str, int] = {},
) -> dict[str, Any]:
    '''Add pages from the doc to the source.
    If reasoning, take a contiguous chunk of length up to 104 inclusive.
    Otherwise, take the whole doc.
    '''
    # all of these are from the same doc, so we can just use the first page to get the doc info
    anchor = row['source'][0]
    pdf_path = utils.pdf_name(anchor)
    n = fn_to_page_count[pdf_path]
    if reasoning and n > 104:
        # pick contiguous chunk of length up to 104 inclusive
        length = 104  # hopefully shorter than gpt-oss max length (128K) when converted to text
        start = random.randint(0, max(0, n - length))
        pages = list(range(start, min(n, start + length)))
    else:
        # take the whole doc
        pages = list(range(n))
    full = [utils.path_as_page(anchor, p) for p in pages]
    return {'source': full}

def _add_adjacent_pages(row: dict, max_adjacent: int = 4, fn_to_page_count: dict[str, int] = {}) -> dict:
    '''Select n_pages random pages adjacent to the anchor (row['image_filename']).
    Return a list of page filenames sorted by page number.

    n_pages will be the exact number of pages selected, document length permitting.
    '''
    anchor_ifn = row['source'][0]
    pdf_path = utils.pdf_name(anchor_ifn)
    anchor_page = utils.pdf_page(anchor_ifn)
    total = fn_to_page_count[pdf_path]
    n_select = min(random.randint(2, max_adjacent + 1), total)

    # Build a contiguous window of size n_select that includes the anchor,
    # centered when possible, and shifted to respect document bounds.
    # number of pages preceding the current page, sometimes preferring pages before, sometimes preferring pages after
    p = (n_select - 1) // 2 + ((n_select % 2 == 0) and random.choice([1,0]))
    left = anchor_page - p
    right = left + n_select
    if left < 0:  # if left border is out of bounds, shift right
        right += -left
        left = 0
    if right > total:  # if right border is out of bounds, shift left
        left -= right - total
        right = total
    if left < 0:  # if left border is out of bounds here, doc is too short, clip to 0
        left = 0

    pages = [_resolve_path(utils.path_as_page(anchor_ifn, i)) for i in range(left, right)]
    return {'source': pages}

def augment_into_splits(
    dataset: Dataset,
    split_sizes: list[int],
    split_names: list[str],
    fn_to_page_count: dict[str, int],
    idx_to_ifn_images_ds: dict[int, str],
    num_proc: int = 32,
) -> DatasetDict:
    '''
    Create long source cols based on various split types.
    '''
    ds = DatasetDict()
    for split_name, start, end in zip(split_names, accumulate([0] + split_sizes[:-1]), accumulate(split_sizes)):
        ds[split_name] = dataset.select(range(start, end))
        if split_name == 'distractors_short':
            ds[split_name].save_to_disk(CACHE_DIR / 'distractors_short_q')
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(
                    _add_pages_hn, 
                    max_total=5, 
                    after_k=32, 
                    top_k=256, 
                    idx_to_ifn_images_ds=idx_to_ifn_images_ds,
                )), batched=True, num_proc=num_proc)
            )
        elif split_name == 'adj_short':
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(
                    _add_adjacent_pages, 
                    max_adjacent=4, 
                    fn_to_page_count=fn_to_page_count,
                )), batched=True, num_proc=num_proc)
            )
        elif split_name == 'hn_short':
            ds[split_name] = (
                ds[split_name]
                .map(utils.hf_batched(partial(
                    _add_pages_hn, 
                    max_total=5, 
                    top_k=32, 
                    idx_to_ifn_images_ds=idx_to_ifn_images_ds,
                )), batched=True, num_proc=num_proc)
            )
        elif 'short' not in split_name:
            if split_name.endswith('hn'):
                ds[split_name] = (
                    ds[split_name]
                    .map(utils.hf_batched(partial(
                        _add_pages_hn, 
                        max_total=64, 
                        top_k=32, 
                        idx_to_ifn_images_ds=idx_to_ifn_images_ds,
                    )), batched=True, num_proc=num_proc)
                )
            elif split_name.endswith('doc'):
                reasoning = 'reasoning' in split_name
                ds[split_name] = (
                    ds[split_name]
                    .filter(partial(
                        _doc_eligible_batched, 
                        fn_to_page_count=fn_to_page_count,
                    ), batched=True, num_proc=num_proc)
                    .map(utils.hf_batched(partial(
                        _add_pages_doc, 
                        reasoning=reasoning, 
                        fn_to_page_count=fn_to_page_count,
                    )), batched=True, num_proc=num_proc)
                )
    return ds

if __name__ == "__main__":
    global FN_TO_PAGE_COUNT, IDX_TO_IFN_IMAGES_DS
    FN_TO_PAGE_COUNT = utils.count_all_pages(PDF_ROOT, CACHE_DIR)
    IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)

    distiset, cost_tracker = run_pipeline(config)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    # split_sizes = [1_000_000] * 3 + [300_000] * 6
    # split_sizes = [4_000] * 3 + [1_500] * 6
    split_sizes = [1800] * 3 + [600] * 6
    # split_sizes = [5] * 9
    ratio_to_fulfill = len(distiset) / sum(split_sizes)
    print(f"ratio of split_sizes that can be fulfilled: {ratio_to_fulfill}")
    split_sizes = [int(r * ratio_to_fulfill) for r in split_sizes]
    split_names = [
        'distractors_short',
        'adj_short',
        'hn_short',
        'recursive_hn',
        'recursive_doc',
        'full_context_one_shot_hn',
        'full_context_one_shot_doc',
        'reasoning_hn',
        'reasoning_doc',
    ]
    ds = augment_into_splits(distiset, split_sizes, split_names, FN_TO_PAGE_COUNT, IDX_TO_IFN_IMAGES_DS)
    ds.save_to_disk(CACHE_DIR / 'single_page_q_ds')

