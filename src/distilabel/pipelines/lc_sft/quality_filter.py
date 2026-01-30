import re
import numpy as np
from functools import partial
from distilabel.pipeline import Pipeline
from datasets import load_from_disk, concatenate_datasets, Dataset
from logging import getLogger
import json
from pathlib import Path
from collections import defaultdict

from itertools import chain
from typing import TYPE_CHECKING
from distilabel.steps import StepInput, GlobalStep

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    Split,
    NoOp,
    Map,
    FilterRows,
    Rejoin,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, CheckClaims

from distilabel.pipelines.lc_sft.synthetic_cot import (
    _some_relevant,
    _combine_evidence,
    _set_lc_mm_source,
)

from distilabel.configs.lc_sft.quality_filter import (
    config,
    CLAIMS_SUPPORTED_THRESHOLD,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
    TOP_K_PAGES,
)

# class LMCorrections(LMGenerationTask):
#     def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
#         no_correction_needed = [
#             row | {'generation': None, 'reasoning': None}
#             for row in inputs 
#             if all(row['claims_supported'])
#         ]
#         correction_needed = [
#             row for row in inputs if not all(row['claims_supported'])
#         ]
#         corrections = []
#         if len(correction_needed) > 0:
#             corrections = next(super().process(correction_needed))
#         yield no_correction_needed + corrections

def _validate_corrections(corrected_answer: str, claims_supported: list[bool], page_source: list[str], **kwargs) -> dict:
    '''Enforce the corrected answer is only provided if it was needed'''
    return {
        'corrected_answer': corrected_answer if not all(claims_supported) else None, 
        'source': page_source,
    }

def _filter_check_claims(row: dict, cols: list[str]) -> bool:
    '''
    Verify that a corrected answer is provided if at least one claim was not supported
    and additionally that the claims supported are above a threshold
    '''
    if row['claims_supported'] is None:
        return False
    if all(row['claims_supported']):
        return True
    return (
        # the reason the corrected_answer check is here and not in _validate_corrections is that we actually want to drop 
        # rows where a correction is needed but no correction is provided
        row['corrected_answer'] is not None
        # for ones that are too contentious (i.e. the original answer claims are entirely disputed) it is more likely one or both sides are wrong
        # so we discard these below a threshold
        and (sum(row['claims_supported']) / len(row['claims_supported'])) >= CLAIMS_SUPPORTED_THRESHOLD
    )

STAGE = 0
BATCH_SIZE = 256

IMG_TAG_PATTERN = re.compile(r"<IMG_\d+>")
PAGE_REF_PATTERN = re.compile(r"\bpage\s*:?\s*\d+\b(?:\s*[-–]\s*\d+\b)?", re.IGNORECASE)

def vds_to_distilabel(row: dict) -> Dataset:
    global IDX_TO_IFN_IMAGES_DS
    assert row['messages'][-2]['role'] == 'user'
    assert row['messages'][-1]['role'] == 'assistant'
    question = row['messages'][-2]['content']
    answer = row['messages'][-1]['content']

    question = re.sub(IMG_TAG_PATTERN, "", question)
    # question = re.sub(PAGE_REF_PATTERN, "", question)
    question = re.sub(r"\s{2,}", " ", question).strip()
    source = [utils.resolve_path(IDX_TO_IFN_IMAGES_DS[i]) for i in row['images']]
    if '</think>' in answer:
        answer = answer.split('</think>')[1]
    # answer = re.sub(PAGE_REF_PATTERN, "", answer)
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    
    return {
        'source': source,
        'question': question,
        'answer': answer,
    }

def run_pipeline(config: Config, dataset: Dataset, pipeline_name: str):
    global STAGE, BATCH_SIZE
 
    with Pipeline(
        name=pipeline_name,
        description='Fact check answers against the source.',
        cache_dir=CACHE_DIR / pipeline_name,
        disable_output_queue_timeout=True,
    ) as pipeline:
        # ---------------------- Stage 0: check answer language matches question language ----------------------
        STAGE = 0
        stage = config.stages[STAGE]
        load_data = LoadDataFromDataset(name='load_data', dataset=dataset, batch_size=BATCH_SIZE)

        check_language_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        check_language = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'check_language_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                lm_input_cols=['answer'],
                lm_input_col_prefixes=['answer: '],
                input_mappings={'source': 'question'},
                output_mappings={'system': 'check_language_system', 'model_name': 'check_language_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['question', 'answer', ...] -> ['check_language_system', 'check_language_model_name', ...]

        filter_check_language = FilterRows(
            name='filter_check_language',
            cols=['answer_language_matches_question_language'],
            condition=utils.logical_and_filters(
                utils.generation_is_structured, utils.cols_true
            ),
            input_batch_size=BATCH_SIZE,
        )

        # ---------------------- Stage 1: answer to claims ----------------------
        STAGE = 1
        stage = config.stages[STAGE]

        to_claims_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        to_claims = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'to_claims_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                input_mappings={'source': 'answer'},
                output_mappings={'system': 'to_claims_system', 'model_name': 'to_claims_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['answer', ...] -> ['answer_claims', 'to_claims_system', 'to_claims_model_name', ...]

        filter_claims = FilterRows(
            name='filter_claims',
            cols=['answer_claims'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        # ---------------------- Stage 2: use synthetic cot pipeline to check claims against the source ----------------------
        STAGE = 2
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
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'evidence_in_chunks_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                lm_input_cols=['answer_claims'],
                lm_input_col_prefixes=['A breakdown of the fundamental answer claims:\n'],
                output_mappings={'system': 'evidence_system', 'model_name': 'evidence_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]

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
                'question', 'question_model_name', 'idxs', 'split', 'answer', 'answer_model_name', 'answer_claims',
                'to_claims_system', 'to_claims_model_name', 'check_language_system', 'check_language_model_name',
                'answer_language_matches_question_language', 'evidence_system', 'images', 'n_images', 'messages',
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

        # ---------------------- Stage 3: check claims supported by the source ----------------------
        STAGE = 3
        stage = config.stages[STAGE]
        lms = pipe_utils.make_lms(config, stage, use_cache=True)

        # for the LC MM models, we want to use some top K most relevant pages in addition to the extracted
        # text context because the images will help ground the models and the models selected for this branch
        # should be strong enough to use the context effectively
        set_lc_mm_source = Map(
            name='set_lc_mm_source',
            fn=partial(_set_lc_mm_source, K=TOP_K_PAGES, min_relevance_score=1.0),
            cols=['relevance_score', 'page_source'],
            output_cols=['source', 'distilled_evidence', 'combined_evidence'],
            input_batch_size=BATCH_SIZE,
        )

        lc_mm_check_claims_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in lms]
        )
        check_claims_lc_mm = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f'check_claims_lc_mm_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE // 2,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['distilled_evidence', 'combined_evidence'],
                lm_input_cols=['question', 'distilled_evidence', 'answer', 'answer_claims'],
                lm_input_col_prefixes=[
                    'Original question:\n', 
                    'Per-page relevant context and your current chain of thought (feel free to correct your previous mistakes):\n',
                    'Given answer:\n', 
                    'A breakdown of the fundamental answer claims:\n',
                ],
                output_mappings={
                    'system': 'answer_system', 
                    'model_name': 'answer_model_name', 
                    'source': 'top_k_pages',
                    'combined_evidence': 'evidence',
                    'analysis': 'claims_analysis',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]

        filter_check_claims = FilterRows(
            name='filter_check_claims',
            cols=['claims_supported'],
            condition=utils.logical_and_filters(
                utils.generation_is_structured,
                _filter_check_claims,
            ),
            input_batch_size=BATCH_SIZE // 2,
        )
        validate_corrections = Map(
            name='validate_corrections',
            fn=_validate_corrections,
            cols=['corrected_answer', 'claims_supported', 'page_source'],
            output_cols=['corrected_answer', 'source'],
            input_batch_size=BATCH_SIZE,
        )  # also restore the full source to the source column

        # ---------------------- Pipeline ----------------------
        (
            # stage 0 - check answer language matches question language
            load_data >> check_language_router >> check_language >> filter_check_language

            # stage 1 - answer to claims
            >> to_claims_router >> to_claims >> filter_claims

            # stage 2 - per-page check claims
            >> split_chunks >> evidence_router >> extract_evidence >> filter_evidence >> rejoin_chunks >> combine_evidence >> filter_relevant

            # stage 3 - check claims supported by the source
            >> set_lc_mm_source >> lc_mm_check_claims_router >> check_claims_lc_mm >> filter_check_claims >> validate_corrections
        )

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *check_language, filter_check_language],
                len(config.stages[0].available_gpus),
            )
            + pipe_utils.steps_to_load_groups(
                [*to_claims, filter_claims],
                len(config.stages[1].available_gpus),
            )
            + pipe_utils.steps_to_load_groups(
                [split_chunks, *extract_evidence, filter_evidence],
                len(config.stages[2].available_gpus),
            )
            + [[rejoin_chunks.name]]  # global step on its own
            + pipe_utils.steps_to_load_groups(
                [
                    combine_evidence, filter_relevant,
                    set_lc_mm_source, *check_claims_lc_mm, filter_check_claims, validate_corrections
                ],
                len(config.stages[3].available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker


def _rm_duplicate_questions(dataset: list[dict]) -> list[dict]:
    '''
    If two rows have the same question, keep the first occurrence
    '''
    questions = set()
    images = set()
    ds = []
    for row in dataset:
        images.add(tuple(row['images']))
        if row['question'] not in questions or 'summary' in row['question']:
            questions.add(row['question'])
            ds.append(row)
    print(f"Removed {len(dataset) - len(ds)} / {len(dataset)} rows due to duplicate questions")
    return ds


def split_thinking(content: str) -> tuple[str, str]:
    """Return content without thinking, thinking content"""
    if "</think>" in content:
        matches = re.search(r"(.*</think>)(.*)", content, flags=re.DOTALL)
        thinking = matches.group(1).strip()
        final_content = matches.group(2).strip()
    else:
        thinking = ""
        final_content = content
    return final_content, thinking


def apply_verified_answer(ds: list[dict]) -> list[dict]:
    '''Use the corrected answer if corrections were made'''
    verified_ds = []
    for row in ds:
        if row['corrected_answer'] is not None:
            answer, thinking = split_thinking(row['messages'][-1]['content'])
            if thinking == '':
                verified_answer = row['corrected_answer']
            else:
                verified_answer = f'{thinking}\n{row['corrected_answer']}'
            verified_ds.append(row | {'messages': row['messages'][:-1] + [{'role': 'assistant', 'content': verified_answer}]})
        else:
            verified_ds.append(row)
    return verified_ds


def make_synthetic_cot_w_reflection(distiset: list[dict]) -> list[dict]:
    '''Use the quality verification and corrected answer to make a synthetic cot dataset with reflection'''
    ds = []
    for row in distiset:
        og_answer = row['messages'][-1]['content']
        final_answer, thinking = split_thinking(og_answer)
        # use the corrected answer if corrections were made
        verified_answer = row['corrected_answer'] if row['corrected_answer'] is not None else final_answer
        # if no thinking, just use the verified answer
        if thinking == '':
            ds.append(row | {'messages': row['messages'][:-1] + [{'role': 'assistant', 'content': f'{verified_answer}'}]})
            continue
        # otherwise, use the thinking, claims, and claims analysis to make a reflection:
        # draft, analysis of the draft against the source, verified answer
        w_reflection = '\n\n'.join([
            thinking.replace('</think>', '').strip(),
            f'Draft Claims:\n{json.dumps(row["answer_claims"], indent=2)}',
            f'Claims Analysis:\n{row["claims_analysis"]}',
            f'Claims Supported:\n{row["claims_supported"]}',
        ])
        if all(row['claims_supported']):
            w_reflection = f'{w_reflection}\nAll claims supported, no corrections needed, providing verified answer:'
        else:
            w_reflection = f'{w_reflection}\nSome claims not supported, correcting and providing verified answer:'
        ds.append(
            row | {
                'messages': row['messages'][:-1] + [
                    {'role': 'assistant', 'content': f'{w_reflection}</think>\n{verified_answer}'}
                ]
            }
        )
    return ds


def make_quality_filter_task(distiset: list[dict], synthetic_cot: bool = False) -> list[dict]:
    '''Make a quality filter task from the distiset rows'''
    ds = []
    system = Path('distilabel/prompts/lc_sft/check_claims_minimal.txt').read_text()
    for row in distiset:
        ds.append(
            {
                'images': row['images'],
                'messages': [
                    {'role': 'system', 'content': f'{'<cot>\n' if synthetic_cot else ''}{system}'},
                    {
                        'role': 'user', 
                        'content': (
                            ''.join([f'<IMG_{i}>' for i in range(len(row['images']))])
                            + f'Original question:\n{row['question']}\n\nGiven answer:\n{row['answer']}\n\nExtracted claims:\n{json.dumps(row['answer_claims'], indent=2)}'
                        )
                    },
                    {
                        'role': 'assistant',
                        'content': (
                            f'<think>{row["distilled_evidence"]}</think>\n'
                            if synthetic_cot
                            else ''
                        )
                        + CheckClaims(
                            analysis=row['claims_analysis'],
                            claims_supported=row['claims_supported'],
                            corrected_answer=row['corrected_answer'],
                        ).model_dump_json(indent=2)
                    }
                ],
                'n_images': len(row['images']),
                'answer_model_name': row['answer_model_name'],
            }
        )
    return ds


if __name__ == '__main__':
    cols_to_keep = [
        'source', 'question', 'question_model_name', 
        'answer', 'answer_model_name', 'split', 'idxs'
    ]
    dataset_names = [
            'adj_short_vds',
            'rag_short_vds',
            'distractors_short_vds',
            'true_multi_page_short_vds',
            'full_context_one_shot_vds',
            ### 'recursive_vds',  # recursive is a subset of synthetic
            'reasoning_vds',
            'synthetic_cot_vds',
            'unanswerable_vds',
            ### 'for_multi_turn/multi_turn_no_think_vds',  # too small
            # 'unanswerable_3k_vds'
        ]
    cached_ds_path = CACHE_DIR / PIPELINE_NAME / 'quality_filter_input_ds'
    if not cached_ds_path.exists():
        global IDX_TO_IFN_IMAGES_DS
        IDX_TO_IFN_IMAGES_DS = utils.get_idx_to_filename(IMAGES_DS_PATH)

        datasets_ = [load_from_disk(CACHE_DIR / ds) for ds in dataset_names]
        datasets = []
        for ds, ds_name in zip(datasets_, dataset_names):
            ds = ds.add_column('idxs', list(range(len(ds))))
            ds = ds.map(
                utils.hf_batched(
                    lambda row: row | 
                    ({'answer_model_name': [row.get('answer_model_name')]}
                    if isinstance(row.get('answer_model_name'), str) else {})
                ), 
                batched=True,
            )
            datasets.append(utils.add_split_label_ds(ds, ds_name))
        dataset = concatenate_datasets(datasets)
        dataset = dataset.map(utils.hf_batched(vds_to_distilabel), batched=True)#, num_proc=4)
        dataset.save_to_disk(cached_ds_path)
    else:
        dataset = load_from_disk(cached_ds_path)

    distiset, cost_tracker = run_pipeline(config, dataset, PIPELINE_NAME)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    synthetic_cot_hashes = set()
    cols_to_keep = ['images', 'messages', 'n_images', 'answer_model_name', 'idxs']
    distiset_list = distiset.to_list()
    ds_dict = defaultdict(list)
    for row in distiset_list:
        ds_dict[row['split']].append(row)
    ds_dict = {
        k: _rm_duplicate_questions(v) for k, v in ds_dict.items()
    }
    for ds_name in dataset_names:
        ds = ds_dict[ds_name]
        # build the synthetic cot with reflection vds
        if ds_name == 'synthetic_cot_vds':
            synthetic_cot_hashes = {
                utils.hash_structure_with_images(
                    (
                        row['images'],
                        [msg['content'].replace('<cot>', '').strip() for msg in row['messages'] if msg['role'] == 'user'][0],
                    )
                ) 
                for row in ds
            }
            reflection_vds = Dataset.from_list(make_synthetic_cot_w_reflection(ds))
            reflection_vds = reflection_vds.select_columns(cols_to_keep)
            reflection_vds = reflection_vds.remove_columns(['idxs'])
            reflection_vds.save_to_disk(CACHE_DIR / f'{ds_name}/synthetic_cot_w_reflection_vds')

        # use the verified answer (changed if corrections were made)
        ds = Dataset.from_list(apply_verified_answer(ds))
        vds = ds.select_columns(cols_to_keep)
        vds = vds.remove_columns(['idxs'])
        vds.save_to_disk(CACHE_DIR / f'{ds_name}/quality_filtered_vds')

    non_think = []
    synthetic_cot = []
    for k, v in ds_dict.items():
        for row in v:
            if 'synthetic_cot' in k:
                synthetic_cot.append(row)
            elif 'reasoning' not in k:
                non_think.append(row)
    non_think = Dataset.from_list(make_quality_filter_task(non_think, synthetic_cot=False))
    synthetic_cot = Dataset.from_list(make_quality_filter_task(synthetic_cot, synthetic_cot=True))
    non_think.save_to_disk(CACHE_DIR / 'non_think_qf_task_vds')
    non_think.shuffle(seed=0).select(range(min(5_000, len(non_think)))).save_to_disk(CACHE_DIR / 'non_think_qf_task_5k_vds')
    synthetic_cot.save_to_disk(CACHE_DIR / 'synthetic_cot_qf_task_vds')
    synthetic_cot.shuffle(seed=0).select(range(min(5_000, len(synthetic_cot)))).save_to_disk(CACHE_DIR / 'synthetic_cot_qf_task_5k_vds')

    # recursive is a subset of synthetic, so use the filtered idxs from synthetic
    recursive_vds = load_from_disk(CACHE_DIR / 'recursive_vds')
    filtered_recursive_vds = recursive_vds.filter(
        utils.hf_batched(
            lambda row: utils.hash_structure_with_images((row['images'], row['messages'][0]['content'])) in synthetic_cot_hashes
        ), batched=True#, num_proc=4
    )
    filtered_recursive_vds.save_to_disk(CACHE_DIR / 'recursive_vds/quality_filtered_vds')

