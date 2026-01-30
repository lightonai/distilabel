from __future__ import annotations

import random
from collections import defaultdict
from typing import  Any, Tuple
from tqdm import tqdm
from functools import partial
from logging import getLogger
from pathlib import Path
from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, DatasetDict

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
    Split,
    Rejoin,
    NoOp,
    JoinParallelBranches,
)
from distilabel.steps.tasks import LMGenerationTask

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config

from distilabel.configs.lc_sft.true_multi_page_q import (
    config,
    EXCLUDE_PDFS,
    DS_PATH,
    IMAGES_DS_PATH,
    PDF_ROOT,
    CACHE_DIR,
    PIPELINE_NAME,
)

# ----------------------------------------------------------------------------
# Seed building helpers
# ----------------------------------------------------------------------------

_resolve_path = partial(utils.resolve_path, path_substitution=config.path_substitution)

SEED = 0
random.seed(SEED)

TARGET_COUNTS = {
    'true_multi_page_short_hn': 10_000,
    'true_multi_page_short_doc': 10_00,
    'recursive_hn': 50_000,
    'recursive_doc': 50_000,
    # 'full_context_one_shot_hn': 500,
    # 'full_context_one_shot_doc': 1_000,
    # 'reasoning_hn': 500,
    # 'reasoning_doc': 1_000,
}
TARGET_COUNTS2 = {
    'true_multi_page_short_hn': 20_000,
    'true_multi_page_short_doc': 20_000,
    'recursive_hn': 150_000,
    'recursive_doc': 150_000,
}

IDX_TO_IFN_IMAGES_DS = None  # filled in main
FN_TO_PAGE_COUNT = None  # filled in main


def _pdf_ok_for_doc(pdf_path: str) -> bool:
    global FN_TO_PAGE_COUNT
    n = FN_TO_PAGE_COUNT[pdf_path]
    return 3 <= n <= 336


def _sorted_by_page(paths: list[str]) -> list[str]:
    return sorted(paths, key=utils.pdf_page)


def _close_hn_seed(row: dict[str, Any], top_k: int, n_pages: int) -> list[str]:
    '''Select n_pages random negs from the top_k negs.
    Return a list of page filenames with the anchor (row['image_filename']) randomly shuffled.

    Thus, there will be n_pages + 1 pages in the seed (sufficient negs permitting).
    '''
    global IDX_TO_IFN_IMAGES_DS
    hni = (row.get('hard_negs_idx_img_img', []) or [])[:top_k]
    hnt = (row.get('hard_negs_idx_txt_img', []) or [])[:top_k]
    candidates = list(dict.fromkeys(hni + hnt))  # preserve order, unique
    candidates = [c for c in candidates if Path(utils.pdf_name(_resolve_path(IDX_TO_IFN_IMAGES_DS[c]))).exists()]
    if len(candidates) == 0:
        return []
    k = min(n_pages, len(candidates))
    chosen = random.sample(candidates, k)
    ifns = [row['image_filename']] + [IDX_TO_IFN_IMAGES_DS[i] for i in chosen if i in IDX_TO_IFN_IMAGES_DS]
    random.shuffle(ifns)
    return [_resolve_path(ifn) for ifn in ifns]


def _adjacent_pages_seed(anchor_ifn: str, n_pages: int) -> list[str]:
    '''Select n_pages random pages adjacent to the anchor (row['image_filename']).
    Return a list of page filenames sorted by page number.

    n_pages will be the exact number of pages selected, document length permitting.
    '''
    pdf_path = utils.pdf_name(anchor_ifn)
    if not _pdf_ok_for_doc(pdf_path):
        return []
    anchor_page = utils.pdf_page(anchor_ifn)
    total = FN_TO_PAGE_COUNT[pdf_path]
    n_select = min(max(2, min(3, n_pages)), total)

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
    return pages


def _random_doc_subsample_seed(anchor_ifn: str, n_pages: int) -> list[str]:
    '''Select n_pages random pages from the doc, in order. Include the anchor.'''
    pdf_path = utils.pdf_name(anchor_ifn)
    if not _pdf_ok_for_doc(pdf_path):
        return []
    total = FN_TO_PAGE_COUNT[pdf_path]
    n_select = max(2, min(3, n_pages))
    pool = list(range(total))
    # Ensure we include anchor
    anchor_page = utils.pdf_page(anchor_ifn)
    pool.remove(anchor_page) if anchor_page in pool else None
    extra = random.sample(pool, k=max(0, n_select - 1))
    pages = [anchor_page] + extra
    return _sorted_by_page([_resolve_path(utils.path_as_page(anchor_ifn, p)) for p in pages])


def get_ds(n: int, front: bool = False) -> Dataset:
    ds = load_from_disk(DS_PATH)
    ds = ds.shuffle(seed=SEED)
    if front:
        ds = ds.select(range(n))
    else:
        ds = ds.select(range(len(ds) - n, len(ds)))
    # Keep anchor and negs
    ds = ds.select_columns(['image_filename', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])
    # Exclude benchmark PDFs
    ds = utils.remove_pdfs_from_dataset(
        ds, 
        EXCLUDE_PDFS, 
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
        num_proc=64,
    )
    ds = utils.remove_pdfs_with_pages_(
        ds, 
        PDF_ROOT, 
        CACHE_DIR, 
        less_than=2, 
        more_than=336, 
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
    )
    return ds


def structured_and_requires_multiple_pages(row: dict, cols: list[str]) -> bool:
    '''
    Check the 'question_fully_answered' column, all must be False for the question to require multiple pages

    Also check that the cols are structured (not None)
    '''
    structured = True
    for col in cols:
        if (
            (isinstance(row[col], list) and any([generation is None for generation in row[col]]))
            or row[col] is None
        ):
            structured = False
    return not any(row['question_fully_answered']) and structured


def build_seeds(ds: Dataset, target_counts: dict[str, int], allow_doc_reuse: bool = False) -> Dataset:
    """Build initial 2+ page seeds for all 8 target splits."""
    ds_cache_path = CACHE_DIR / f'.true_multi_page_q_seeds_{utils.hash_structure_with_images(target_counts)}'
    if ds_cache_path.exists():
        return load_from_disk(ds_cache_path)

    # Precompute mapping and page counts
    global IDX_TO_IFN_IMAGES_DS, FN_TO_PAGE_COUNT
    IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)
    FN_TO_PAGE_COUNT = utils.count_all_pages(PDF_ROOT, CACHE_DIR)

    # Distribute types per split
    need = target_counts.copy()
    seeds: dict[str, list[dict[str, Any]]] = {k: [] for k in need}

    # Helpers for type assignment
    def want(split: str) -> bool:
        '''are more rows needed for this split?'''
        if split not in need:
            return False
        return len(seeds[split]) < need[split]

    def add_to_seed(split: str, src: list[str], row: dict[str, Any]):
        '''add a source (list of pages) to the seeds dict, including the hard negs'''
        if len(src) < 2:
            return
        seeds[split].append({
            'source': src,
            'split': split,
            # carry negs to allow HN augmentation later
            'hard_negs_idx_img_img': row.get('hard_negs_idx_img_img', []),
            'hard_negs_idx_txt_img': row.get('hard_negs_idx_txt_img', []),
        })

    # Iterate rows and fill splits uniformly by type where applicable
    idx_doc = 0

    docs_used = set()
    ds = list(ds)
    for row in tqdm(ds, desc='Building seeds'):
        # if all target splits are filled, stop
        if all(len(seeds[k]) >= need[k] for k in need):
            break
            
        anchor = _resolve_path(row['image_filename'])
        # if doc reuse is not allowed, skip if the doc has already been used
        pdf_path = utils.pdf_name(anchor)
        if not allow_doc_reuse and pdf_path in docs_used:
            continue
        docs_used.add(pdf_path)

        # true_multi_page_short_hn: only close hard negs, top 8
        if want('true_multi_page_short_hn'):
            src = _close_hn_seed(row, top_k=32, n_pages=random.randint(1, 16))
            add_to_seed('true_multi_page_short_hn', src, row)
            continue

        # true_multi_page_short_doc: only adjacent pages, doc in [3,336]
        if want('true_multi_page_short_doc'):
            src = _adjacent_pages_seed(anchor, n_pages=random.randint(2, 16))
            add_to_seed('true_multi_page_short_doc', src, row)
            continue

        # HN splits, pulls from top 8–24 negs of each type (img_img and txt_img)
        anchor_used = False
        for split in ['recursive_hn', 'full_context_one_shot_hn', 'reasoning_hn']:
            if not want(split):
                continue
            # random top_k means it will sometimes pull from closest negs, sometimes slightly further
            # closest negs are probably more coherent together and thus better for making 
            # questions that require each of them. But should be diverse
            src = _close_hn_seed(row, top_k=random.randint(8, 48), n_pages=random.randint(1, 16))
            add_to_seed(split, src, row)
            anchor_used = True
            break
        if anchor_used:
            continue
        
        # Doc splits, half the time a contiguous set of pages, half the time random selection from the doc
        for split in ['recursive_doc', 'full_context_one_shot_doc', 'reasoning_doc']:
            if not want(split):
                continue
            idx_doc = (idx_doc + 1) % 2
            if idx_doc == 0:
                src = _adjacent_pages_seed(anchor, n_pages=random.randint(2, 16))
            else:
                src = _random_doc_subsample_seed(anchor, n_pages=random.randint(2, 16))
            add_to_seed(split, src, row)
            break

    # Flatten
    ds_list = [item for split in seeds for item in seeds[split]]
    ds_list = [row for row in ds_list if all(Path(utils.pdf_name(_resolve_path(ifn))).exists() for ifn in row['source'])]
    all_seeds = Dataset.from_list(ds_list)
    all_seeds.save_to_disk(ds_cache_path)
    return all_seeds

def suitable_for_questions(row: dict, cols: list[str]) -> bool:
    return not row['not_suitable_for_questions']

STAGE = 0
BATCH_SIZE = 256

def run_pipeline_for_questions(seed_ds: Dataset, config: Config):
    global STAGE
    STAGE = 0

    with Pipeline(
        name=PIPELINE_NAME,
        description='Generate multi-page questions with filtering (stages 0–2 of true_multi_page).',
        cache_dir=CACHE_DIR / 'true_multi_page_q',
        disable_output_queue_timeout=True,
    ) as pipeline:
        # Stage 0: initial questions
        stage = config.stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=seed_ds, batch_size=BATCH_SIZE)

        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )

        lms = pipe_utils.make_lms(config, stage, use_cache=True, invalidate_cache=False)
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
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'question_system', 'model_name': 'question_model_name', 'analysis': 'question_analysis'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['questions', 'question_system', 'question_model_name', ...]

        filter_not_suitable_for_questions = FilterRows(
            name='filter_not_suitable_for_questions',
            cols=['not_suitable_for_questions'],
            condition=utils.logical_and_filters(
                utils.generation_is_structured, 
                suitable_for_questions,
            ),
            input_batch_size=BATCH_SIZE,
        )

        questions_to_rows = ListToRows(
            name="questions_to_rows",
            input_col='questions',
            input_batch_size=BATCH_SIZE,
            output_mappings={'questions': 'question'},
            resources=StepResources(replicas=1),
        )

        drop_none_questions = FilterRows(
            name="drop_none_questions",
            cols=['question'],
            condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )

        # Stage 1: single page answers + question requirements
        STAGE = 1
        stage = config.stages[STAGE]

        lms = pipe_utils.make_lms(config, stage, use_cache=True, invalidate_cache=False)
        sp_lms = [lm for lm in lms if lm.lm_config.task_name == 'single_page_answer']
        q_req_lms = [lm for lm in lms if lm.lm_config.task_name == 'question_requirements']

        split_pages = {
            branch: Split(
                name=f'split_pages_{branch}',
                input_col='source',
                keep_as_list=True, # keep source as a list of strings (format distinguishing pages from text directly for input)
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=1),
            ) for branch in ['sp_branch', 'q_req_branch']
        }

        q_req_branch = NoOp(
            name='q_req_branch',
            cols=['source'],
            output_mappings={'source': 'page_source'},
            input_batch_size=BATCH_SIZE,
        )
        q_req_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in q_req_lms]
        )
        generate_question_req = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'question_reqs_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['page_source'],
                input_mappings={'source': 'question'},  # don't want the pages as context, just question
                output_mappings={
                    'system': 'question_requirements_system',
                    'model_name': 'question_requirements_model_name',
                    'page_source': 'source',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(q_req_lms)
        ]  # cols: ['question', ...] -> ['question_requirements', 'question_requirements_system', 'question_requirements_model_name', ...]

        filter_question_requirements = FilterRows(
            name='filter_question_requirements',
            cols=['question_requirements'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        sp_answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in sp_lms]
        )
        generate_single_page_answers = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'single_page_answers_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question'],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['split', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'],
                output_mappings={'system': 'sp_answer_system', 'model_name': 'sp_answer_model_name', 'generation': 'sp_answer'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(sp_lms)
        ]  # cols: ['source', 'question', ...] -> ['sp_answer', 'sp_answer_system', 'sp_answer_model_name', ...]
        drop_none_sp_answers = FilterRows(
            name='drop_none_sp_answers',
            cols=['sp_answer'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        collect_sp_answers_and_q_req = JoinParallelBranches(
            name='collect_sp_answers_and_q_req',
            join_on_cols=['source', 'question'],  # the pair (source, question) is mostly unique for each branch
            input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'page_source'},
        )

        # global step must take and output all current rows, so that will all get routed together
        # we use the no op to break that into batches
        break_up_collect_sp_answers_and_q_req = NoOp(name='break_up_collect_sp_answers_and_q_req', input_batch_size=BATCH_SIZE)

        # Stage 2: judge and filter
        STAGE = 2
        stage = config.stages[STAGE]

        judge_answers_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )

        lms = pipe_utils.make_lms(config, stage, use_cache=True, invalidate_cache=False)
        judge_answers = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'answer_judge_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question_requirements', 'sp_answer'],
                lm_input_col_prefixes=['question requirements: ', 'answer: '],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['page_source'],
                input_mappings={'source': 'question'},
                output_mappings={
                    'system': 'judge_system',
                    'model_name': 'judge_model_name',
                    'page_source': 'source',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['question', 'question_requirements', 'sp_answer', ...] -> ['question_requirements_met', 'question_fully_answered', 'judge_system', 'judge_model_name', ...]

        rejoin_pages = Rejoin(
            name='rejoin_pages',
            input_col='source',
            drop_incomplete_rows=True,
            duplicates_cols={
                'question', 'question_analysis', 'question_system', 'question_model_name',
                'question_requirements', 'question_requirements_system', 'question_requirements_model_name',
                'split', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img',
            },
            input_batch_size=BATCH_SIZE,
        )

        drop_poor_questions = FilterRows(
            name='drop_poor_questions',
            cols=['question_fully_answered'],
            condition=structured_and_requires_multiple_pages,
            input_batch_size=BATCH_SIZE,
        )

        # Graph
        # Stage 0: initial questions
        load_data >> data_router >> generate_questions >> filter_not_suitable_for_questions >> questions_to_rows >> drop_none_questions

        # Stage 1: single page answers + question requirements
        drop_none_questions >> [q_req_branch, split_pages['sp_branch']]

        # q_req_branch (split pages at the end in this branch because the lm response is per question)
        q_req_branch >> q_req_router >> generate_question_req >> filter_question_requirements >> split_pages['q_req_branch']

        # sp_branch
        split_pages['sp_branch'] >> sp_answer_router >> generate_single_page_answers >> drop_none_sp_answers

        # join branches
        [split_pages['q_req_branch'], drop_none_sp_answers] >> collect_sp_answers_and_q_req

        # Stage 2: judge and filter
        (collect_sp_answers_and_q_req >> break_up_collect_sp_answers_and_q_req >>
         judge_answers_router >> judge_answers >> rejoin_pages >> drop_poor_questions)

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_questions, filter_not_suitable_for_questions, questions_to_rows, drop_none_questions],
                len(config.stages[0].available_gpus),
            ) +
            pipe_utils.steps_to_load_groups(
                [q_req_branch, *generate_question_req, filter_question_requirements, split_pages['q_req_branch'],
                 split_pages['sp_branch'], *generate_single_page_answers, drop_none_sp_answers],
                len(config.stages[1].available_gpus),
            ) +
            [[collect_sp_answers_and_q_req.name]] + # global step goes on its own
            pipe_utils.steps_to_load_groups(
                [break_up_collect_sp_answers_and_q_req, *judge_answers],
                len(config.stages[2].available_gpus),
            ) +
            [[rejoin_pages.name]] + # global step goes on its own
            pipe_utils.steps_to_load_groups(
                [drop_poor_questions],
                len(config.stages[2].available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker


# ----------------------------------------------------------------------------
# Add rest of the pages to make full LC examples
# ----------------------------------------------------------------------------


def _add_pages_hn(
    row: dict[str, Any],
    max_total: int,
    top_k: int,
) -> dict[str, Any]:
    '''
    Add a random amount of hard negatives to the source from the top_k hard negatives for each type
    up to max_total total images (including the existing source pages)

    The source order is shuffled before returning.
    '''
    global IDX_TO_IFN_IMAGES_DS
    source = row['source']
    remaining = max(0, max_total - len(source))
    hni = (row.get('hard_negs_idx_img_img', []) or [])[:top_k]
    hnt = (row.get('hard_negs_idx_txt_img', []) or [])[:top_k]
    candidates = list(dict.fromkeys(hni + hnt))
    candidates = [c for c in candidates if Path(utils.pdf_name(_resolve_path(IDX_TO_IFN_IMAGES_DS[c]))).exists()]

    # add a random amount up to the max_total
    n = random.randint(0, remaining)
    add = random.sample(candidates, k=min(n, len(candidates))) if n > 0 else []
    negs = [IDX_TO_IFN_IMAGES_DS[i] for i in add]
    new_source = source + list(dict.fromkeys(negs))
    return {'source': random.sample(new_source, k=len(new_source))}


def _doc_eligible_batched(batch: dict[str, list[Any]]) -> list[bool]:
    global FN_TO_PAGE_COUNT
    keep: list[bool] = []
    for src in batch['source']:
        anchor = src[0]
        pdf_path = utils.pdf_name(anchor)
        n = FN_TO_PAGE_COUNT[pdf_path]
        keep.append(3 <= n <= 336)
    return keep


def _add_pages_doc(
    row: dict[str, Any],
    reasoning: bool = False,
) -> dict[str, Any]:
    '''Add pages from the doc to the source.
    If reasoning, take a contiguous chunk of length up to 104 inclusive.
    Otherwise, take the whole doc.
    '''
    global FN_TO_PAGE_COUNT
    # all of these are from the same doc, so we can just use the first page to get the doc info
    anchor = row['source'][0]
    pdf_path = utils.pdf_name(anchor)
    n = FN_TO_PAGE_COUNT[pdf_path]
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


def augment_into_splits(distiset: Dataset) -> DatasetDict:
    '''Add additional pages to the source to make full LC examples
    Using the split col to determine whether to add HN pages or pages
    from the doc.
    '''
    global TARGET_COUNTS
    global IDX_TO_IFN_IMAGES_DS, FN_TO_PAGE_COUNT
    if IDX_TO_IFN_IMAGES_DS is None:
        IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)
    if FN_TO_PAGE_COUNT is None:
        FN_TO_PAGE_COUNT = utils.count_all_pages(PDF_ROOT, CACHE_DIR)

    # Group by target split
    by_split = {
        split: distiset.filter(utils.hf_batched(lambda row: row['split'] == split), batched=True, num_proc=4)
        for split in TARGET_COUNTS.keys()
    }

    out = {}
    cols_to_keep = ['source', 'question', 'question_model_name']

    # true multi page short doesn't get additional pages
    if 'true_multi_page_short_hn' in by_split:
        out['true_multi_page_short_hn'] = by_split['true_multi_page_short_hn'].select_columns(cols_to_keep)
    if 'true_multi_page_short_doc' in by_split:
        out['true_multi_page_short_doc'] = by_split['true_multi_page_short_doc'].select_columns(cols_to_keep)

    # HNs
    for name in ['recursive_hn', 'full_context_one_shot_hn', 'reasoning_hn']:
        if name in by_split:
            out[name] = (
                by_split[name]
                .map(utils.hf_batched(partial(_add_pages_hn, max_total=96, top_k=48)), batched=True, num_proc=4)
                .select_columns(cols_to_keep)
            )
    # Docs
    for name in ['recursive_doc', 'full_context_one_shot_doc', 'reasoning_doc']:
        reasoning = 'reasoning' in name
        if name in by_split:
            out[name] = (
                by_split[name]
                .filter(partial(_doc_eligible_batched), batched=True, num_proc=4)
                .map(utils.hf_batched(partial(_add_pages_doc, reasoning=reasoning)), batched=True, num_proc=4)
                .select_columns(cols_to_keep)
            )

    return DatasetDict(out)


if __name__ == '__main__':
    total_needed = sum(TARGET_COUNTS.values())
    base_ds = get_ds(4_000_000, front=False)
    seed_ds1 = build_seeds(base_ds, TARGET_COUNTS, allow_doc_reuse=True)  # allow doc reuse to get more examples
    from tqdm import tqdm
    seed1_fns = {tuple(sorted(fns)) for fns in tqdm(seed_ds1['source'])}
    
    seed_ds2 = build_seeds(base_ds, TARGET_COUNTS2, allow_doc_reuse=True) 
    seed2_fns = {tuple(sorted(fns)): idx for idx, fns in enumerate(tqdm(seed_ds2['source']))}
    seed_ds = seed_ds2.select([idx for fns, idx in seed2_fns.items() if fns not in seed1_fns])
    seed_ds = seed_ds.select(range(140_000, len(seed_ds))).flatten_indices(num_proc=64) 

    questions_ds, cost_tracker = run_pipeline_for_questions(seed_ds, config)
    print(f"Cost: {dict(cost_tracker)}")
    questions_ds = questions_ds['default']['train']
    questions_ds = questions_ds.remove_columns(['distilabel_metadata'])

    questions_ds = questions_ds.shuffle(seed=0)
    questions_ds = utils.take_n_first_doc_occurrences(
        questions_ds, 
        row_to_ifn=lambda row: _resolve_path(row['source'][0]),
        _resolve_path=_resolve_path,
        n=4,
    )

    ds_dict = augment_into_splits(questions_ds)
    ds_dict.save_to_disk(CACHE_DIR / 'true_multi_page_q_ds_2')

    init_row_to_src = {
        utils.hash_structure_with_images({k: v for k, v in row.items() if k in {'question', 'question_model_name'}}): row['source']
        for row in questions_ds
    }
    final_rows = [row for split in ds_dict.values() for row in split]

    row_to_init_src = {}
    for row in final_rows:
        row_id = utils.hash_structure_with_images({k: v for k, v in row.items() if k in {'question', 'question_model_name'}})
        if set(init_row_to_src[row_id]).issubset(set(row['source'])):
            row_to_init_src[row_id] = init_row_to_src[row_id]
    utils.save_json(CACHE_DIR / 'true_multi_page_q_ds_2' / 'row_to_init_src.json', row_to_init_src)


