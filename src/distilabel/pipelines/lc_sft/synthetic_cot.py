from functools import partial
import numpy as np
import random
from distilabel.pipeline import Pipeline
from datasets import load_from_disk, concatenate_datasets, Dataset
from logging import getLogger
import re

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    Split,
    Rejoin,
    NoOp,
    Map,
    FilterRows,
    ConcatenateBranches,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config

from distilabel.configs.lc_sft.synthetic_cot import (
    config,
    SP_DS_PATH,
    MP_DS_PATH,
    MP_DS_PATH_PT2,
    CACHE_DIR,
    TOP_K_PAGES,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

STAGE = 0
BATCH_SIZE = 256
MIN_RELEVANCE_SCORE = 1.0


def _prompt_question_source_as_relevant(
    source: list[str],
    question_source: list[str],
    **kwargs
) -> dict:
    '''If the single page of the source is part of the question source, prompt the model that it is explicitly relevant'''
    return {
        'explicitly_relevant': (
            (
                'Note, this page you are given is marked as explicitly relevant to the question ' 
                'because the question was based on a set of up to 16 pages that included this one. ' 
                'Thus, your relevance_score should be between 6.0 and 10.0, '
                'but you are free to adjust within this range to reflect your own judgement.'
            )
            if source[0] in set(question_source) 
            else None
        )
    }


def _some_relevant(row: dict, cols: list[str]) -> bool:
    # row['relevant'] should be a list of bools
    return any(row['relevant'])


def _combine_evidence(
    evidence: list[str], 
    relevant: list[bool], 
    relevance_score: list[float], 
    explicitly_relevant: list[str | None],
    **kwargs
) -> dict:
    """
    Build a single evidence string from page-level evidence
    """
    assert len(explicitly_relevant) == len(relevance_score)
    relevance_score = [
        score if exp_rel is None else max(score, 6.0)
        for score, exp_rel in zip(relevance_score, explicitly_relevant)
    ]

    # parts uses up to the top topk
    parts: list[str] = []
    asort = np.argsort(relevance_score)
    top_k_pages = asort[np.array(relevance_score)[asort] > MIN_RELEVANCE_SCORE][-TOP_K_PAGES:] + 1  # +1 because page numbers are 1-indexed
    # reversed orders it so that the highest relevance score is first
    # I think this will be interesting because the model will see different page orders,
    # forcing it to both consider the relevancy of pages and sort them and will give it
    # the flexibility to go back and forth between pages, not being stuck to a single order 
    # where if it skips something, it now is sure it isn't relevant. I think having diverse orders
    # as well will help the model learn to think more, because it is more challenging to LM that as 
    # opposed to the single order.
    for i in reversed(top_k_pages):
        ev_str = evidence[i - 1].strip()
        if ev_str:
            parts.append(f"Page {i}:\n{ev_str}")

    # verbose uses all relevant evidence
    verbose_parts: list[str] = []
    for i, (ev, rel) in enumerate(zip(evidence, relevant), start=1):
        ev_str = ev.strip()
        if ev_str and rel:
            verbose_parts.append(f"Page {i}:\n{ev_str}")
    
    combined = "\n\n".join(parts)
    verbose_combined = "\n\n".join(verbose_parts)
    return {'distilled_evidence': combined, 'combined_evidence': verbose_combined}


def _set_lc_mm_source(
    relevance_score: list[float], 
    page_source: list[str], 
    K: int, 
    min_relevance_score: float = MIN_RELEVANCE_SCORE, 
    **kwargs
) -> dict:
    '''Takes the top K most relevant pages and sets them as the source'''
    assert len(page_source) == len(relevance_score)
    top_k_pages = np.argsort(relevance_score)
    top_k_pages = top_k_pages[np.array(relevance_score)[top_k_pages] >= min_relevance_score][-K:]
    # reversed orders it so that the highest relevance score is first
    return {'source': [page_source[i] for i in reversed(top_k_pages)]}


def run_pipeline(config: Config, dataset: Dataset):
    global STAGE, BATCH_SIZE

    with Pipeline(
        name=PIPELINE_NAME,
        description='Extract evidence from each page, then combine and use this to answer the question.',
        cache_dir=CACHE_DIR / 'synthetic_cot',
    ) as pipeline:
        # ---------------------- Stage 0: evidence extraction ----------------------
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

        prompt_question_source_as_relevant = Map(
            name='prompt_question_source_as_relevant',
            fn=_prompt_question_source_as_relevant,
            cols=['source', 'question_source'],
            output_cols=['explicitly_relevant'],
            input_batch_size=BATCH_SIZE,
        )

        evidence_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage0.lm_configs]
        )

        lms0 = pipe_utils.make_lms(config, stage0, use_cache=True)
        extract_evidence = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'evidence_in_chunks_{i}',
                stage=stage0,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question', 'explicitly_relevant'],
                lm_input_col_prefixes=['Given question: ', ''],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'evidence_system', 'model_name': 'evidence_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms0)
        ]  # cols: ['source', 'question', ...] -> ['evidence', 'relevant', 'evidence_system', 'evidence_model_name', ...]

        filter_evidence = FilterRows(
            # use_cache=True,
            name='filter_evidence',
            cols=['evidence'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        # Rejoin all chunks for each row (global step), restoring original source
        rejoin_chunks = Rejoin(
            # use_cache=True,
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
                'partial_source',
                'question_source',
            ],
            input_batch_size=BATCH_SIZE,
        )

        # Combine evidence text from chunks
        combine_evidence = Map(
            # use_cache=True,
            name='combine_evidence',
            fn=_combine_evidence,
            cols=['evidence', 'relevant', 'relevance_score', 'source'],
            output_cols=['combined_evidence', 'distilled_evidence'],
            input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'page_source'},
        )  # cols: ['source', 'evidence', 'relevant'] -> ['page_source', 'combined_evidence']

        filter_relevant = FilterRows(
            # use_cache=True,
            name='filter_relevant',
            cols=['relevant'],
            condition=utils.logical_and_filters(_some_relevant, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )

        # ---------------------- Stage 1: answer using evidence ----------------------
        STAGE = 1
        stage1 = config.stages[STAGE]

        lms1 = pipe_utils.make_lms(config, stage1, use_cache=True)
        lc_mm_lms = [lm for lm in lms1 if lm.lm_config.task_name == 'overall_answer_lc_mm']
        text_only_lms = [lm for lm in lms1 if lm.lm_config.task_name == 'overall_answer_text_only']

        lc_mm_lm_configs = [lm.lm_config for lm in lc_mm_lms]
        text_only_lm_configs = [lm.lm_config for lm in text_only_lms]

        branch_router = pipe_utils.data_router(
            step_distribution=[
                sum(lm_config.data_ratio for lm_config in lc_mm_lm_configs),
                sum(lm_config.data_ratio for lm_config in text_only_lm_configs),
            ],
        )

        lc_mm_branch = NoOp(name='lc_mm_branch', input_batch_size=BATCH_SIZE)
        # the goal is for the extracted evidence to be sufficient for the models in this branch to answer the question
        text_only_branch = NoOp(name='text_only_branch', input_batch_size=BATCH_SIZE)

        # for the LC MM models, we want to use some top K most relevant pages in addition to the extracted
        # text context because the images will help ground the models and the models selected for this branch
        # should be strong enough to use the context effectively
        set_lc_mm_source = Map(
            # use_cache=True,
            name='set_lc_mm_source',
            fn=partial(_set_lc_mm_source, K=TOP_K_PAGES, min_relevance_score=MIN_RELEVANCE_SCORE),
            cols=['relevance_score', 'page_source'],
            output_cols=['source'],
            input_batch_size=BATCH_SIZE,
        )

        lc_mm_answer_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in lc_mm_lm_configs]
        )
        generate_answers_lc_mm = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'answer_generation_lc_mm_{i}',
                stage=stage1,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                lm_input_cols=['question'],
                lm_input_col_prefixes=[''],
                system_col='default_system',
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['distilled_evidence'],
                output_mappings={
                    'generation': 'answer',
                    'system': 'answer_system', 
                    'model_name': 'answer_model_name', 
                    'source': 'top_k_pages',
                    'distilled_evidence': 'evidence',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lc_mm_lms)
        ]  # cols: ['source', 'combined_evidence', 'question', ...] -> ['top_k_pages', 'answer', 'evidence', 'answer_system', 'answer_model_name', ...]
        end_lc_mm_branch = NoOp(name='end_lc_mm_branch', input_batch_size=BATCH_SIZE)

        # For text-only branch, prepare inputs for stage 1: preserve original pages as page_source and set combined evidence as source
        set_evidence_as_source = NoOp(
            # use_cache=True,
            # invalidate_cache=True,
            name='set_evidence_as_source',
            cols=['source', 'distilled_evidence'],
            output_mappings={'distilled_evidence': 'source'},
            input_batch_size=BATCH_SIZE,
        )  # cols: ['distilled_evidence'] -> ['source']

        text_only_answer_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in text_only_lm_configs]
        )
        generate_answers_text_only = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'answer_generation_text_only_{i}',
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
                    'source': 'evidence',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(text_only_lms)
        ]  # cols: ['source', 'question', ...] -> ['evidence', 'answer', 'answer_system', 'answer_model_name', ...]
        end_text_only_branch = NoOp(name='end_text_only_branch', input_batch_size=BATCH_SIZE)

        join_mm_and_text_only = ConcatenateBranches(
            # use_cache=True,
            name='join_mm_and_text_only',
            col_factories={'top_k_pages': list},
            output_mappings={'source': 'drop'},
            input_batch_size=BATCH_SIZE,
        )

        # Restore original set of pages (e.g. the full document) as source for output
        restore_pages = NoOp(
            # use_cache=True,
            name='restore_pages',
            cols=['page_source'],
            output_mappings={'page_source': 'source'},
            input_batch_size=BATCH_SIZE,
        )  # cols: ['page_source'] -> ['source']

        filter_answers = FilterRows(
            # use_cache=True,
            name='filter_answers',
            cols=['answer'],
            condition=utils.generation_is_structured,  # will simply check not None
            input_batch_size=BATCH_SIZE,
        )

        # ---------------------- Pipeline ----------------------
        (
            load_data >> split_chunks >> prompt_question_source_as_relevant >> 
            evidence_router >> extract_evidence >> filter_evidence >> rejoin_chunks >> combine_evidence >> filter_relevant >> branch_router
        )
        
        branch_router >> [lc_mm_branch, text_only_branch]

        # LC MM branch
        lc_mm_branch >> set_lc_mm_source >> lc_mm_answer_router >> generate_answers_lc_mm >> end_lc_mm_branch

        # Text only branch
        text_only_branch >> set_evidence_as_source >> text_only_answer_router >> generate_answers_text_only >> end_text_only_branch

        # Join MM and text only branches and filter
        [end_lc_mm_branch, end_text_only_branch] >> join_mm_and_text_only >> restore_pages >> filter_answers

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, split_chunks, prompt_question_source_as_relevant, *extract_evidence, filter_evidence],
                len(config.stages[0].available_gpus),
            )
            + [[rejoin_chunks.name]]  # global step on its own
            + pipe_utils.steps_to_load_groups(
                [
                    combine_evidence, filter_relevant, 
                    lc_mm_branch, set_lc_mm_source, *generate_answers_lc_mm, end_lc_mm_branch,
                    text_only_branch, set_evidence_as_source, *generate_answers_text_only, end_text_only_branch,
                    join_mm_and_text_only, restore_pages, filter_answers,
                ],
                len(config.stages[1].available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker


fn_to_idx: dict[str, int] | None = None

def convert_to_vision(row: dict, path_substitution: tuple[str | re.Pattern, str] | None = None, **kwargs) -> dict:
    '''
    Convert the row to vision format
    '''
    global fn_to_idx
    image_indices = [
        fn_to_idx[
             (
                path_substitution[0].sub(path_substitution[1], ifn) 
                if isinstance(path_substitution[0], re.Pattern)
                else ifn.replace(path_substitution[0], path_substitution[1])
            )
            if path_substitution else ifn
        ] for ifn in row['source']
    ]

    user_content = (
        ''.join([f'<IMG_{i}>' for i in range(len(image_indices))])
        + row['question']
    )
    assistant_content = f'<think>{row['evidence']}</think>\n{row['answer']}'
    messages = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content}
    ]

    control_token = '<cot>'
    r = random.random()
    if r < 0.5:
        if random.random() < 0.5:
            user_content = f'{control_token} {user_content}'
        else:
            user_content = f'{user_content} {control_token}'
        messages[0]['content'] = user_content
    elif r < 0.95:
        messages.insert(0, {'role': 'system', 'content': control_token})
    else:
        # no cot, teaches the model that the control token implies <think></think>
        messages[-1]['content'] = row['answer']

    return {
        'images': image_indices,
        'messages': messages,
        'n_images': len(image_indices),
    }


def _add_question_source(ds: Dataset, full_to_init_src: dict[str, list[str]]) -> Dataset:
    keys = {k for k in ds.column_names if k != 'source'}
    return Dataset.from_list(
        [
            row | {
                'question_source': full_to_init_src[
                    utils.hash_structure_with_images({k: v for k, v in row.items() if k in keys})
                ]
            }
            for row in ds
        ]
    )


if __name__ == '__main__':
    # Load only the desired splits and add a split label
    cols_to_keep = ['source', 'question', 'split', 'question_model_name']
    sp_ds_dict = load_from_disk(SP_DS_PATH)
    mp_ds_dict = load_from_disk(MP_DS_PATH)
    mp_ds_dict_pt2 = load_from_disk(MP_DS_PATH_PT2)

    sp_splits = [
        'distractors_short',
        'adj_short',
        'hn_short',
        'recursive_hn',
        'recursive_doc',

        # 'full_context_one_shot_hn',
        # 'full_context_one_shot_doc',
        # 'reasoning_hn',
        # 'reasoning_doc',
    ]
    mp_splits = [
        'true_multi_page_short_hn',
        'true_multi_page_short_doc',
        'recursive_hn',
        'recursive_doc',

        # 'full_context_one_shot_hn',
        # 'full_context_one_shot_doc',
        # 'reasoning_hn',
        # 'reasoning_doc',
    ]

    sp_full_to_init_src = utils.load_json(SP_DS_PATH / 'row_to_init_src.json')
    mp_full_to_init_src = utils.load_json(MP_DS_PATH / 'row_to_init_src.json')
    mp_full_to_init_src_pt2 = utils.load_json(MP_DS_PATH_PT2 / 'row_to_init_src.json')

    datasets: list[Dataset] = []
    for split in sp_splits:
        datasets.append(
            utils.add_split_label_ds(
                _add_question_source(sp_ds_dict[split], sp_full_to_init_src),
                f'sp_{split}'
            )
        )
    for split in mp_splits:
        datasets.append(
            utils.add_split_label_ds(
                _add_question_source(mp_ds_dict[split], mp_full_to_init_src),
                f'mp_{split}'
            )
        )
        datasets.append(
            utils.add_split_label_ds(
                _add_question_source(mp_ds_dict_pt2[split], mp_full_to_init_src_pt2),
                f'mp_{split}_pt2'
            )
        )
    dataset = concatenate_datasets(datasets).select_columns(cols_to_keep + ['question_source']).shuffle(seed=0)
    # dataset = dataset.select(range(50_000)).flatten_indices(num_proc=2)
    dataset = dataset.select(range(50_000, 100_000)).flatten_indices(num_proc=2)

    distiset, cost_tracker = run_pipeline(config, dataset)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    # format to vision generic
    images_ds = load_from_disk(IMAGES_DS_PATH)
    fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename', config.path_substitution)

    distiset = utils.format_distiset(
        distiset, 
        convert_to_vision, 
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split', 'question_source'], 
        n_workers=16,
    )

    distiset.save_to_disk(CACHE_DIR / 'synthetic_cot_vds_pt2')
    # distiset = distiset.shuffle(seed=0)
    # mt = distiset.select(range(1000))
    # mt.save_to_disk(CACHE_DIR / 'for_multi_turn' / 'synthetic_cot_vds')
    # distiset = distiset.select(range(1000, len(distiset)))
    # distiset.save_to_disk(CACHE_DIR / 'synthetic_cot_vds')

    # hn = distiset.filter(utils.hf_batched(lambda row: 'hn' in row['split']), batched=True, num_proc=16).remove_columns(['split'])
    # doc = distiset.filter(utils.hf_batched(lambda row: 'doc' in row['split']), batched=True, num_proc=16).remove_columns(['split'])

    # hn.save_to_disk(CACHE_DIR / 'synthetic_cot_hn_vds')
    # doc.save_to_disk(CACHE_DIR / 'synthetic_cot_doc_vds')

