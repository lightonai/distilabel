from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from logging import getLogger
from itertools import accumulate
import random
from typing import Any
from functools import partial
import re
from copy import deepcopy

from distilabel.steps import (
    StepResources, 
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
    Split,
    Map,
    Rejoin,
    ConcatenateBranches,
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
    augment_into_splits,
)
from distilabel.pipelines.lc_sft.true_multi_page_q import (
    build_seeds,
)
from collections import defaultdict
from distilabel.pipelines.lc_sft.synthetic_cot import (
    _some_relevant,
    _combine_evidence,
    _set_lc_mm_source,
)

from distilabel.configs.lc_sft.multi_turn import (
    config,
    EXCLUDE_PDFS,
    DS_PATH,
    PDF_ROOT,
    CACHE_DIR,
    IMAGES_DS_PATH,
    TOP_K_PAGES,
    MAX_TURNS,
    PIPELINE_NAME,
)
import json

SEED = 7439
random.seed(SEED)

def _resolve_path(path: str) -> str:
    return path.replace(config.path_substitution[0], config.path_substitution[1])

def get_ds(n: int, seed: int) -> Dataset:
    if (CACHE_DIR / 'multi_turn_input_ds').exists():
        return load_from_disk(CACHE_DIR / 'multi_turn_input_ds')
    num_proc = 1
    ds = load_from_disk(DS_PATH)
    ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    ds = ds.select(range(n))
    ds = ds.select_columns(['image_filename', 'hard_negs_idx_img_img', 'hard_negs_idx_txt_img'])
    ds = utils.remove_pdfs_with_pages_(
        ds,
        PDF_ROOT,
        CACHE_DIR,
        more_than=336,
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
        num_proc=num_proc,
    )
    ds = ds.map(lambda x: {'source': [_resolve_path(x['image_filename'])]}, num_proc=num_proc)

    # just need to make various types of source combinations
    # hard negatives, adjacent pages or the whole doc
    split_names = [
        'distractors_short',
        'adj_short',
        'hn_short',
        'hn',
        'doc',
    ]
    split_weights = [1, 1, 1, 3, 6]
    split_sizes = [((i * n) // sum(split_weights)) for i in split_weights]
    fn_to_page_count = utils.count_all_pages(PDF_ROOT, CACHE_DIR)
    idx_to_ifn_images_ds = utils.get_idx_to_filename(IMAGES_DS_PATH)
    ds_dict = augment_into_splits(
        dataset=ds, 
        split_sizes=split_sizes, 
        split_names=split_names, 
        fn_to_page_count=fn_to_page_count, 
        idx_to_ifn_images_ds=idx_to_ifn_images_ds, 
        num_proc=num_proc,
    )
    ds = concatenate_datasets(list(ds_dict.values()))
    # Exclude benchmark PDFs
    ds = utils.remove_pdfs_from_dataset(
        ds,
        EXCLUDE_PDFS, 
        row_to_ifn=lambda row: _resolve_path(row['image_filename']),
        num_proc=num_proc,
    )
    ds = ds.add_column('conversation', [[] for _ in range(len(ds))])
    ds.save_to_disk(CACHE_DIR / 'multi_turn_input_ds')
    return ds

def _lm_generation_task(
    name: str,
    lm: OpenAILM, 
    stage: Stage,
    lm_input_cols: list[str],
    extra_cols: list[str] = [],
    lm_input_col_prefixes: list[str] = [],
    output_mappings: dict[str, str] = {},
    input_mappings: dict[str, str] = {},
    use_cache: bool = True,
    invalidate_cache: bool = False,
) -> LMGenerationTask:
    return LMGenerationTask(
        use_cache=use_cache,
        invalidate_cache=invalidate_cache,
        name=name,
        stage=stage,
        llm=lm,
        lm_config=lm.lm_config,
        input_formatter=lm.format_input,
        parallel_input_formatter=lm.parallel_format_inputs,
        lm_input_cols=lm_input_cols,
        lm_input_col_prefixes=lm_input_col_prefixes,
        extra_cols=extra_cols,
        input_batch_size=BATCH_SIZE,
        resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size, oversubscribe=lm.lm_config.replicas_per_vllm_server),
        output_mappings=output_mappings,
        input_mappings=input_mappings,
        **lm.lm_config.task_kwargs,
    )

def _drop_none_str(name: str, cols: list[str]) -> FilterRows:
    return FilterRows(
        name=name,
        cols=cols,
        condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
        input_batch_size=BATCH_SIZE,
    )

def _single_page(source: list[str], rng: random.Random) -> tuple[list[str], list[int]]:
    indices = [rng.choice(range(len(source)))]
    return [source[indices[0]]], indices

def _random_subset(source: list[str], rng: random.Random, max_pages: int = 5) -> tuple[list[str], list[int]]:
    '''Select up to max_pages random pages from the source in order.'''
    n_select = min(rng.randint(2, max_pages), len(source))
    indices = sorted(rng.sample(range(len(source)), k=n_select))
    return [source[i] for i in indices], indices

def _adjacent_subset(source: list[str], rng: random.Random, max_pages: int = 5) -> tuple[list[str], list[int]]:
    '''Select up to max_pages adjacent pages from the source in order.'''
    anchor_page = rng.choice(range(len(source)))
    n_select = min(rng.randint(2, max_pages), len(source))
    # Build a contiguous window of size n_select that includes the anchor,
    # centered when possible, and shifted to respect document bounds.
    # number of pages preceding the current page, sometimes preferring pages before, sometimes preferring pages after
    p = (n_select - 1) // 2 + ((n_select % 2 == 0) and rng.choice([1,0]))
    left = anchor_page - p
    right = left + n_select
    if left < 0:  # if left border is out of bounds, shift right
        right += -left
        left = 0
    if right > len(source):  # if right border is out of bounds, shift left
        left -= right - len(source)
        right = len(source)
    if left < 0:  # if left border is out of bounds here, doc is too short, clip to 0
        left = 0
    indices = sorted(list(range(left, right)))
    return [source[i] for i in indices], indices

def _sample_source(source: list[str], conversation: list[dict], loop_idx: int, **kwargs) -> dict:
    """
    Sample from the source a single page, a random set of pages, or a set of adjacent pages
    """
    rng = random.Random(utils.hash_structure_with_images(source) + str(loop_idx))
    conversation_pages = [msg['pages'] for msg in conversation if 'pages' in msg]
    if rng.random() < 0.5 and len(conversation_pages) > 0:
        indices = conversation_pages[-1]
        partial_source = [source[i] for i in indices]
    else:
        r = rng.random()
        if r < 0.33 and len(source) > 1:
            partial_source, indices = _random_subset(source, rng)
        elif r < 0.66 and len(source) > 1:
            partial_source, indices = _adjacent_subset(source, rng)
        else:
            partial_source, indices = _single_page(source, rng)
    return {'partial_source': partial_source, 'partial_source_indices': indices}

def _sample_conversation(
    source: list[str], 
    conversation: list[dict], 
    loop_idx: int,
    **kwargs
) -> dict:
    def _drop_none_values(conversation: list[dict]) -> list[dict]:
        # None values will appear when pytable formatting is used,
        # don't want to distract the LM with these
        return [
            {k: v for k, v in msg.items() if v is not None}
            for msg in conversation
        ]

    conversation = _drop_none_values(conversation)
    out = _sample_source(source, conversation, loop_idx)
    out['conversation'] = conversation + [{'pages': out.pop('partial_source_indices')}]
    rng = random.Random(utils.hash_structure_with_images(source) + str(loop_idx))

    out['conversation_history'] = json.dumps(out['conversation'], indent=2, ensure_ascii=False)
    if len(conversation) == 0:
        out['followup_on_question'] = '''This is the beginning of the conversation, please ask the first question.'''
    else:
        out['followup_on_question'] = rng.choice([msg['content'] for msg in conversation if msg.get('role') == 'user'])
    return out

def _map_to_conversation(question: str, answer: str, conversation: list[dict], **kwargs) -> dict:
    conversation.extend(
        [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ]
    )
    return {'conversation': conversation}

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 256

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    
    stages = config.stages
    dataset = get_ds(20, SEED)

    distiset = dataset
    cost_tracker = defaultdict(int)
    rng = random.Random(SEED)
    loop_distisets = []
    for loop_idx in range(MAX_TURNS):
        with Pipeline(
            name=f"{PIPELINE_NAME}_{loop_idx}",
            description="Simulate various scenarios in a loop for multi-turn QA.",
            cache_dir=CACHE_DIR / 'multi_turn' / f'loop_{loop_idx}',
        ) as pipeline:
            # ---------------------- Stage 0: Generate followup questions ----------------------
            STAGE = 0
            stage = stages[STAGE]
            if loop_idx != 0:
                # half of the conversations stop at their current number of turns every loop
                selection = sorted(rng.sample(range(len(distiset)), k=len(distiset) // 2))
                loop_distisets.append((
                    distiset
                    .select(set(range(len(distiset))) - set(selection))
                    .flatten_indices(num_proc=1)
                ))
                distiset = (
                    distiset
                    .select(selection)
                    .flatten_indices(num_proc=1)
                )
                distiset = distiset.select_columns(['source', 'conversation'])
            if len(distiset) == 1:
                loop_distisets.append(distiset)
                break
            load_data = LoadDataFromDataset(name="load_data", dataset=distiset, batch_size=BATCH_SIZE)  # cols: ['source', ...]
            lms = pipe_utils.make_lms(config, stage, use_cache=True)

            followup_sample_conversation = Map(
                name='followup_sample_conversation',
                fn=partial(_sample_conversation, loop_idx=loop_idx),
                cols=['source', 'conversation'],
                output_cols=['partial_source', 'conversation_history', 'followup_on_question', 'conversation'],
                input_batch_size=BATCH_SIZE,
            )
            followup_question_router = pipe_utils.data_router(step_distribution=[lm.lm_config.data_ratio for lm in lms])
            followup_question = [
                _lm_generation_task(
                    name=f"followup_question_{i}",
                    lm=lm,
                    use_cache=True,
                    # invalidate_cache=True,
                    stage=stage,
                    lm_input_cols=['conversation_history', 'followup_on_question'],
                    lm_input_col_prefixes=[
                        '\nThis is the current conversation: ',
                        '\nThis is your reference question from earlier in the conversation: ',
                    ],
                    input_mappings={'source': 'partial_source'},
                    output_mappings={'system': 'question_system', 'model_name': 'question_model_name'},
                )
                for i, lm in enumerate(lms)
            ]  # cols: ['source', 'conversation_history', 'followup_on_question'] -> ['question', 'question_model_name', 'question_system', ...]
            drop_none_q = _drop_none_str(name="drop_none_q", cols=['question'])  # cols: ['question', ...] -> ['question', ...]

            # ---------------------- Stage 1: Evidence extraction ----------------------
            STAGE = 1
            stage = config.stages[STAGE]
            lms = pipe_utils.make_lms(config, stage, use_cache=True)

            # Chunk source into 1-page chunks per row
            split_chunks = Split(
                name='split_chunks',
                input_col='source',
                chunk_size=1,
                input_batch_size=BATCH_SIZE,
            )

            evidence_router = pipe_utils.data_router(
                step_distribution=[lm.lm_config.data_ratio for lm in lms]
            )
            extract_evidence = [
                _lm_generation_task(
                    name=f'evidence_in_chunks_{i}',
                    use_cache=True,
                    # invalidate_cache=True,
                    lm=lm,
                    stage=stage,
                    lm_input_cols=['question'],
                    lm_input_col_prefixes=['Given question: '],
                    output_mappings={'system': 'evidence_system', 'model_name': 'evidence_model_name'},
                )
                for i, lm in enumerate(lms)
            ]  # cols: ['source', 'question', ...] -> ['evidence', 'relevant', 'evidence_system', 'evidence_model_name', ...]

            filter_evidence = FilterRows(
                name='filter_evidence',
                cols=['evidence'],
                condition=utils.generation_is_structured,
                input_batch_size=BATCH_SIZE,
            )

            # Rejoin all chunks for each row (global step), restoring original source
            rejoin_chunks = Rejoin(
                name='rejoin_chunks',
                input_col='source',
                duplicates_cols=[
                    'question', 
                    'question_model_name', 
                    'split', 
                    'evidence_system',
                    'question_system',
                    'hard_negs_idx_img_img',
                    'hard_negs_idx_txt_img',
                    'followup_on_question',
                    'conversation_history',
                    'conversation',
                    'analysis',
                    'partial_source',
                ],
                input_batch_size=BATCH_SIZE,
            )

            # Combine evidence text from chunks
            combine_evidence = Map(
                name='combine_evidence',
                fn=_combine_evidence,
                cols=['evidence', 'relevant', 'source'],
                output_cols=['combined_evidence'],
                input_batch_size=BATCH_SIZE,
                output_mappings={'source': 'page_source'},
            )  # cols: ['source', 'evidence', 'relevant'] -> ['page_source', 'combined_evidence']

            filter_relevant = FilterRows(
                name='filter_relevant',
                cols=['relevant'],
                condition=utils.logical_and_filters(_some_relevant, utils.generation_is_structured),
                input_batch_size=BATCH_SIZE,
            )

            # ---------------------- Stage 2: answer using evidence ----------------------
            STAGE = 2
            stage = config.stages[STAGE]

            lms = pipe_utils.make_lms(config, stage, use_cache=True)
            lc_mm_lms = [lm for lm in lms if lm.lm_config.task_name == 'overall_answer_lc_mm']

            # for the LC MM models, we want to use some top K most relevant pages in addition to the extracted
            # text context because the images will help ground the models and the models selected for this branch
            # should be strong enough to use the context effectively
            set_lc_mm_source = Map(
                name='set_lc_mm_source',
                fn=partial(_set_lc_mm_source, K=TOP_K_PAGES),
                cols=['relevance_score', 'page_source'],
                output_cols=['source'],
                input_batch_size=BATCH_SIZE,
            )

            lc_mm_answer_router = pipe_utils.data_router(
                step_distribution=[lm.lm_config.data_ratio for lm in lc_mm_lms]
            )
            generate_answers_lc_mm = [
                _lm_generation_task(
                    use_cache=True,
                    # invalidate_cache=True,
                    name=f'answer_generation_lc_mm_{i}',
                    lm=lm,
                    stage=stage,
                    lm_input_cols=['combined_evidence', 'question'],
                    lm_input_col_prefixes=['Per-page relevant context and your current chain of thought (feel free to correct your previous mistakes): ', ''],
                    extra_cols=['combined_evidence'],
                    output_mappings={
                        'generation': 'answer',
                        'system': 'answer_system', 
                        'model_name': 'answer_model_name', 
                        'source': 'top_k_pages',
                        'combined_evidence': 'evidence',
                    },
                )
                for i, lm in enumerate(lc_mm_lms)
            ]  # cols: ['source', 'combined_evidence', 'question', ...] -> ['top_k_pages', 'answer', 'evidence', 'answer_system', 'answer_model_name', ...]

            filter_answers = FilterRows(
                name='filter_answers',
                cols=['answer'],
                condition=utils.generation_is_structured,  # will simply check not None
                input_batch_size=BATCH_SIZE,
            )  # cols: ['page_source'] -> ['source'] # Restore original set of pages (e.g. the full document) as source for output
            map_to_conversation = Map(
                name='map_to_conversation',
                fn=_map_to_conversation,
                cols=['question', 'answer'],
                output_cols=['conversation'],
                input_batch_size=BATCH_SIZE,
            )
            reset_source = NoOp(name='reset_source', cols=['page_source'], output_mappings={'page_source': 'source'}, input_batch_size=BATCH_SIZE)

            ## Pipeline
            (
                load_data >> followup_sample_conversation >> followup_question_router >> followup_question >> drop_none_q >> 
                
                split_chunks >> evidence_router >> extract_evidence >> filter_evidence >> rejoin_chunks >> combine_evidence >> filter_relevant >> 
                
                set_lc_mm_source >> lc_mm_answer_router >> generate_answers_lc_mm >> filter_answers >> map_to_conversation >> reset_source
            )
        
        distiset, loop_cost_tracker = pipeline.run(
            load_groups=(
                pipe_utils.steps_to_load_groups(
                    [load_data, followup_sample_conversation, *followup_question, drop_none_q, split_chunks],
                    len(stage.available_gpus),
                ) + pipe_utils.steps_to_load_groups(
                    [*extract_evidence, filter_evidence],
                    len(stage.available_gpus),
                ) + pipe_utils.steps_to_load_groups(  # global step by itself
                    [rejoin_chunks],
                    len(stage.available_gpus),
                ) + pipe_utils.steps_to_load_groups(
                    [combine_evidence, filter_relevant],
                    len(stage.available_gpus),
                ) + pipe_utils.steps_to_load_groups(
                    [set_lc_mm_source, *generate_answers_lc_mm, filter_answers, map_to_conversation, reset_source],
                    len(stage.available_gpus),
                )
            ),
            use_cache=True,
            # invalidate_distiset=True,
        )
        distiset = distiset['default']['train']
        for k, v in loop_cost_tracker.items():
            cost_tracker[k] += v
    return concatenate_datasets(loop_distisets), cost_tracker

fn_to_idx: dict[str, int] | None = None

def convert_to_vision(row: dict, path_substitution: tuple[str, str] | None = None, **kwargs) -> dict:
    '''
    Convert the row to vision format
    '''
    global fn_to_idx
    image_indices = [fn_to_idx[ifn.replace(path_substitution[0], path_substitution[1])] for ifn in row['source']]

    conversation = [{k: v for k, v in msg.items() if v is not None} for msg in row['conversation'] if 'pages' not in msg]
    user_content = (
        ''.join([f'<IMG_{i}>' for i in range(len(image_indices))])
        + conversation[0]['content']
    )
    messages = [
        {'role': 'user', 'content': user_content},
    ]
    for msg in conversation[1:]:
        if msg['role'] == 'user':
            messages.append({'role': 'user', 'content': msg['content']})
        else:
            messages.append({'role': 'assistant', 'content': msg['content']})

    return {
        'images': image_indices,
        'messages': messages,
        'n_images': len(image_indices),
    }

_IMG_TAG_PATTERN = re.compile(r"<IMG_(\d+)>")

def _shift_image_tags(content: str, offset: int) -> str:
    if offset == 0:
        return content
    return _IMG_TAG_PATTERN.sub(lambda match: f"<IMG_{int(match.group(1)) + offset}>", content)


def _merge_metadata_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    '''
    Metadata is values from keys not in ['messages', 'images', 'n_images']
    These fields are concatenated into a list and if they are all the same,
    only the first value is used
    '''
    metadata: dict[str, Any] = {}
    keys = {
        key
        for row in rows
        for key in row.keys()
        if key not in {'messages', 'images', 'n_images'}
    }
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        first_value = values[0]
        if all(value == first_value for value in values):
            metadata[key] = first_value
        else:
            metadata[key] = values
    return metadata


def _merge_same_source_rows(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    '''
    Merge rows that have the same source, concatenating the conversation in a random order. 
    The images are only seen once at the beginning
    '''
    shuffled_rows = rows[:]
    rng.shuffle(shuffled_rows)

    merged_row = _merge_metadata_rows(shuffled_rows)
    merged_row['images'] = list(shuffled_rows[0]['images'])
    merged_row['n_images'] = shuffled_rows[0]['n_images']

    merged_messages: list[dict[str, Any]] = []
    seen_system_messages: set[str] = set()
    is_first_user_message = True

    for row in shuffled_rows:
        for message in row['messages']:
            msg_copy = message.copy()
            role = msg_copy.get('role')
            # for system prompts, only add unseen ones to the messages
            if role == 'system':
                content = msg_copy.get('content')
                if content is not None and content not in seen_system_messages:
                    merged_messages.append(msg_copy)
                    seen_system_messages.add(content)
                continue

            # for user messages, the first one keeps the tags, otherwise they are dropped
            if role == 'user':
                content = msg_copy.get('content', '')
                if not is_first_user_message:
                    content = _IMG_TAG_PATTERN.sub('', content).strip()
                else:
                    is_first_user_message = False
                if content:
                    msg_copy['content'] = content
                    merged_messages.append(msg_copy)
                continue

            # assistant messages are just added
            merged_messages.append(msg_copy)

    merged_row['messages'] = merged_messages
    return merged_row


def _merge_different_source_rows(
    rows: list[dict[str, Any]], 
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shuffled_rows = rows[:]
    rng.shuffle(shuffled_rows)

    merged_row = _merge_metadata_rows(shuffled_rows)
    merged_images: list[int] = []
    merged_messages: list[dict[str, Any]] = []
    seen_system_messages: set[str] = set()
    image_offset = 0
    unused_rows: list[dict[str, Any]] = []

    max_n_images = rng.randint(5, 336)
    for row in shuffled_rows:
        if len(merged_images) + len(row['images']) > max_n_images:
            unused_rows.append(row)
            continue
        merged_images.extend(row['images'])
        row_first_user = True
        for message in row['messages']:
            msg_copy = message.copy()
            role = msg_copy.get('role')
            if role == 'system':
                content = msg_copy.get('content')
                if content is not None and content not in seen_system_messages:
                    merged_messages.append(msg_copy)
                    seen_system_messages.add(content)
                continue

            content = msg_copy.get('content')
            if isinstance(content, str):
                if role == 'user' and row_first_user:
                    row_first_user = False  # first user message keeps the tags
                else:
                    msg_copy['content'] = _shift_image_tags(content, image_offset)

            merged_messages.append(msg_copy)
        image_offset += row['n_images']

    merged_row['images'] = merged_images
    merged_row['n_images'] = len(merged_images)
    merged_row['messages'] = merged_messages
    return merged_row, unused_rows


def concatenate_conversations(
    dataset: Dataset,
    same_source_target_images: int | None = None,
    different_source_target_images: int | None = None,
    seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Create extended conversations grouped by image sources.

    Returns a dict with ``same_source`` and ``different_source`` lists containing
    merged examples. Generation for each list stops once the cumulative number of
    images meets the provided target (if any).
    """

    if len(dataset) == 0:
        return {'same_source': [], 'different_source': []}

    grouped_rows: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset.to_list():
        grouped_rows[tuple(row['images'])].append(row)

    rng = random.Random(seed)

    aggregated_rows = [
        _merge_same_source_rows(rows, rng)
        for rows in grouped_rows.values()
        if len(rows) >= 2 
    ]
    single_occurrence_rows = [
        row
        for rows in grouped_rows.values()
        if len(rows) == 1
    ]

    # Same-source combinations
    same_source_examples: list[dict[str, Any]] = aggregated_rows
    if same_source_target_images is not None:
        same_source_examples: list[dict[str, Any]] = []
        same_source_image_total = 0
        for aggregated_row in aggregated_rows:
            example = deepcopy(aggregated_row)
            same_source_examples.append(example)
            same_source_image_total += example['n_images']
            if same_source_image_total >= same_source_target_images:
                break

    # Different-source combinations
    different_source_examples: list[dict[str, Any]] = []
    different_source_image_total = 0
    if (len(aggregated_rows) + len(single_occurrence_rows)) >= 2:
        diff_candidates = aggregated_rows + single_occurrence_rows
        rng.shuffle(diff_candidates)
        idx = 0
        step_size = MAX_TURNS
        while idx + 1 < len(diff_candidates):
            chunk_rows = [deepcopy(row) for row in diff_candidates[idx: idx + step_size]]
            idx += step_size

            example, unused_rows = _merge_different_source_rows(chunk_rows, rng)
            diff_candidates.extend(unused_rows)
            different_source_examples.append(example)
            different_source_image_total += example['n_images']

            if (
                different_source_target_images is not None
                and different_source_image_total >= different_source_target_images
            ):
                break

    return {
        'same_source': Dataset.from_list(same_source_examples),
        'different_source': Dataset.from_list(different_source_examples),
    }

if __name__ == "__main__":
    distiset, cost_tracker = run_pipeline(config)
    print(f"Cost: {dict(cost_tracker)}")

    # format to vision generic
    images_ds = load_from_disk(IMAGES_DS_PATH)
    fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename', config.path_substitution)

    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=1,
    )

    distiset_n_images = sum(distiset['n_images'])

    synthetic_cot_ds = load_from_disk(CACHE_DIR / 'synthetic_cot_for_multi_turn_vds')
    mp_short_hn_ds = load_from_disk(CACHE_DIR / 'true_multi_page_short_hn_for_multi_turn_vds')
    mp_short_doc_ds = load_from_disk(CACHE_DIR / 'true_multi_page_short_doc_for_multi_turn_vds')
    
    cat_conversations = concatenate_conversations(
        distiset,
        same_source_target_images=distiset_n_images,
        different_source_target_images=distiset_n_images,
        seed=SEED,
    )
    distiset = concatenate_datasets([
        distiset, 
        cat_conversations['same_source'], 
        cat_conversations['different_source'],
    ])

    distiset.save_to_disk(CACHE_DIR / 'multi_turn_vds')

