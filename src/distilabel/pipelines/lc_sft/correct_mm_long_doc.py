import re
import numpy as np
from functools import partial
from distilabel.pipeline import Pipeline
from datasets import load_from_disk, concatenate_datasets, Dataset, load_dataset
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

from distilabel.configs.lc_sft.correct_mm_long_doc import (
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

def _resolve_path(path: str) -> str:
    return path.replace(config.path_substitution[0], config.path_substitution[1])

def mmlongdoc_to_distilabel(row: dict) -> dict:
    global IDX_TO_IFN_IMAGES_DS
    root = Path('/mnt/nfs/hf_home/hub/datasets--yubo2333--MMLongBench-Doc/snapshots/38bceac8784469e70ad783dbf26c0b6ff08e0a9a/documents/')
    pdf_path = root / row['doc_id']
    if not pdf_path.exists():
        return row['doc_id']
    _, page_count = utils.get_page_count(pdf_path)
    source = [
        utils.page_path(str(pdf_path), i) for i in range(page_count)
    ]
    if len(source) == 0:
        return row['doc_id']
    return {
        'source': source,
        'question': row['question'],
        'answer': row['answer'],
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

        # ---------------------- Stage 2: use synthetic cot pipeline to check claims against the source ----------------------
        STAGE = 1
        stage = config.stages[STAGE]
        lms = pipe_utils.make_lms(config, stage, use_cache=False)

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
                invalidate_cache=True,
                name=f'evidence_in_chunks_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                lm_input_cols=['question', 'answer'],
                lm_input_col_prefixes=['Given question for reference:\n', 'Given answer for reference:\n'],
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
                'question', 'question_model_name', 'idxs', 'split', 'answer', 'answer_model_name',
                'check_language_system', 'check_language_model_name',
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
        STAGE = 2
        stage = config.stages[STAGE]
        lms = pipe_utils.make_lms(config, stage, use_cache=False)

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
                invalidate_cache=True,
                name=f'check_claims_lc_mm_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE // 2,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                extra_cols=['distilled_evidence', 'combined_evidence'],
                lm_input_cols=['question', 'distilled_evidence', 'answer'],
                lm_input_col_prefixes=[
                    'Original question:\n', 
                    'Per-page relevant context and your current chain of thought (feel free to correct your previous mistakes):\n',
                    'Given answer:\n', 
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
                [split_chunks, *extract_evidence, filter_evidence],
                len(config.stages[1].available_gpus),
            )
            + [[rejoin_chunks.name]]  # global step on its own
            + pipe_utils.steps_to_load_groups(
                [
                    combine_evidence, filter_relevant,
                    set_lc_mm_source, *check_claims_lc_mm, filter_check_claims, validate_corrections
                ],
                len(config.stages[2].available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker


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


if __name__ == '__main__':
    cols_to_keep = [
        'source', 'question', 'question_model_name', 
        'answer', 'answer_model_name', 'split', 'idxs'
    ]
    dataset = load_dataset('yubo2333/MMLongBench-Doc', split='train')
    distilabel_dataset = []
    pdfs_not_found = []
    for row in dataset:
        distilabel_row = mmlongdoc_to_distilabel(row)
        if isinstance(distilabel_row, str):
            pdfs_not_found.append(distilabel_row)
            continue
        distilabel_dataset.append(distilabel_row)
    distilabel_dataset = Dataset.from_list(distilabel_dataset)

    distiset, cost_tracker = run_pipeline(config, distilabel_dataset, PIPELINE_NAME)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    # distiset.save_to_disk(CACHE_DIR / 'distiset')

    distiset_questions = set(row['question'] for row in distiset)
    missing = [row for row in dataset if row['question'] not in distiset_questions]

    import random
    from pathlib import Path
    from typing import Optional, List
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import json
    from glob import glob
    from collections import Counter

    from datasets import Dataset, load_from_disk as lds
    from distilabel import utils
    from distilabel.utils import get_image

    import numpy as np
    from PIL import Image
    import random
    import os
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import io
    from PyPDF2 import PdfMerger
    import tempfile

    IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
    image_ds = lds(IMAGES_DS_PATH)


    def extract_images(inp):
        images = []
        for msg in inp:
            if isinstance(msg['content'], list):
                images.extend([utils.b64_decode_image(m['image_url']['url'][22:]) for m in msg['content'] if 'image_url' in m])
        return images

    def get_images(image_ds_idxs: list[int]) -> list[Image.Image]:
        global image_ds
        return [get_image(image_ds, idx, path_substitution=('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/'), pdf_backend='pdf2image') for idx in image_ds_idxs]

    def print_conv(messages: list[dict], strip_think=False):
        out = '=' * 50 + '\n'
        def render_msg(msg, out):
            if msg.get('type') == 'image_url':
                out += 'IMAGE,'
                return out
            if 'role' in msg:
                out += str(msg['role']).upper() + '\n'
            if msg.get('type') == 'text':
                t = msg['text']
                if strip_think and '</think>' in t:
                    t = t.split('</think>')[1]
                out += str(t) + '\n\n'
                return out
            t = msg['content']
            if strip_think and '</think>' in t:
                t = t.split('</think>')[1]
            out += t + '\n\n'
            return out
        
        def render_msg_list(msg_list, out):
            for msg in msg_list:
                out = render_msg(msg, out)
            return out
        
        for msg in messages:
            if isinstance(msg['content'], list):
                out = render_msg_list(msg['content'], out)
            else:
                out = render_msg(msg, out)
        out += '=' * 50
        print(out)

    def images_to_pdf(images: list[Image.Image], output_dir='visualizing_generated_data'):
        os.makedirs(output_dir, exist_ok=True)
        merger = PdfMerger()
        for i, img in enumerate(images):
            hn_pdf_path = os.path.join(output_dir, f"hn_{i}.pdf")
            create_pdf_page(img, hn_pdf_path)
            merger.append(hn_pdf_path)
        output_path = os.path.join(output_dir, f"visualization.pdf")
        merger.write(output_path)
        merger.close()
        print(f"Saved PDF to {output_path}")

    def visualize_images(ds, idx, images_ds, output_dir='visualizing_generated_data'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a PDF merger
        merger = PdfMerger()

        # Create a temporary directory for individual page PDFs
        with tempfile.TemporaryDirectory() as temp_dir:
            for images_ds_idx in ds[idx]['images']:
                hn_img = get_image(images_ds, images_ds_idx, path_substitution=('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/'))
                hn_pdf_path = os.path.join(temp_dir, f"hn_{images_ds_idx}.pdf")
                create_pdf_page(hn_img, hn_pdf_path)
                merger.append(hn_pdf_path)

            # Save the merged PDF
            output_path = os.path.join(output_dir, f'visualization.pdf')
            merger.write(output_path)
            merger.close()

        print(f"Saved PDF to {output_path}")

    def create_pdf_page(img, output_path, header_text='', subheader_text=''):
        # Get image dimensions
        img_width, img_height = img.size

        # Create a PDF with the same aspect ratio as the image
        c = canvas.Canvas(output_path, pagesize=(img_width, img_height))

        # Convert PIL image to bytes for ReportLab
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Use reportlab's ImageReader
        from reportlab.lib.utils import ImageReader
        img_reader = ImageReader(img_bytes)

        # Draw the image on the canvas
        c.drawImage(img_reader, 0, 0, width=img_width, height=img_height)

        # Add semi-transparent overlay for text at the top
        c.setFillColorRGB(0, 0, 0, 0.7)
        c.rect(0, img_height-40, img_width, 40, fill=1, stroke=0)

        # Add header text
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(10, img_height-15, header_text)

        # Add subheader text
        c.setFont("Helvetica", 10)
        c.drawString(10, img_height-30, subheader_text)

        c.save()

    corrected = [(i, row) for i, row in enumerate(distiset) if (row['corrected_answer'] and row['corrected_answer'] != row['answer']) or (row['corrected_question'] and row['corrected_question'] != row['question'])]
    if Path('distilabel/out/correct_mm_long_doc/annotations.json').exists():
        serializable_distiset = utils.load_json('distilabel/out/correct_mm_long_doc/annotations.json')
    else:
        serializable_distiset = distiset.to_list()

    def n(i):
        serializable_distiset[i]['judgement'] = 'fine'
        utils.save_json('distilabel/out/correct_mm_long_doc/annotations.json', serializable_distiset)
    def c(i, judgement = None, question = None, answer = None):
        serializable_distiset[i]['judgement'] = judgement or 'change'
        serializable_distiset[i]['final_question'] = question or serializable_distiset[i]['corrected_question'] or serializable_distiset[i]['question']
        serializable_distiset[i]['final_answer'] = answer or serializable_distiset[i]['corrected_answer'] or serializable_distiset[i]['answer']
        utils.save_json('distilabel/out/correct_mm_long_doc/annotations.json', serializable_distiset)
    pass
    # format as corrections to the original dataset
    missing[34] = missing[34] | {'judgement': 'change', 'final_question': missing[34]['question'], 'final_answer': 'Not answerable'}
    missing_to_idx = {row['question']: i for i, row in enumerate(missing)}
    sd_to_idx = {row['question']: i for i, row in enumerate(serializable_distiset)}
    og_to_idx = {row['question']: i for i, row in enumerate(dataset)}
    final = []
    for row in dataset:
        if row['question'] in sd_to_idx:
            final_row = serializable_distiset[sd_to_idx[row['question']]]
            og_row = dataset[og_to_idx[row['question']]]
            res_row = og_row | {'question': final_row.get('final_question') or final_row['question'], 'answer': final_row.get('final_answer') or final_row['answer'], 'judgement': final_row.get('judgement') or 'unchecked', 'og_question': og_row['question'], 'og_answer': og_row['answer'], 'pipeline_analysis': final_row['claims_analysis'], 'pipeline_evidence_pages': str([i for i in range(len(final_row['relevance_score'])) if final_row['relevance_score'][i] > 0])}
            final.append(res_row)
        else:
            final_row = missing[missing_to_idx[row['question']]]
            og_row = dataset[og_to_idx[row['question']]]
            res_row = og_row | {'question': final_row.get('final_question') or final_row['question'], 'answer': final_row.get('final_answer') or final_row['answer'], 'judgement': final_row.get('judgement') or 'unchecked', 'og_question': og_row['question'], 'og_answer': og_row['answer']}
            final.append(res_row)
    final = [
        row | {'question': row['question'].replace('percentage difference', 'difference in percent (not relative)')}
        for row in final
        if 'drop' not in row.get('judgement', '')
    ]
    final = [
        row | (
            {'judgement': 'change'} if (row['question'] != row['og_question'] or row['answer'] != row['og_answer']) else {'judgement': row['judgement']}
        )
        for row in final
    ]
    # utils.save_json('distilabel/out/correct_mm_long_doc/final.json', final)
    # final = Dataset.from_list(final)
    # final.save_to_disk('/mnt/nfs/austin_shared/data/corrected_mm_long_bench_doc')

'''
for judgements, allow 0 to count for not answerable

i, row = corrected[135]
images_to_pdf([get_image(None, src) for src in row['source']])
print(f"Evidence:\n{row['distilled_evidence']}\n\nAnalysis:\n{row['claims_analysis']}\n\nCorrected Question:\n{row['corrected_question']}\n\nCorrected Answer:\n{row['corrected_answer']}\n\nQuestion:\n{row['question']}\n\nAnswer:\n{row['answer']}")
print(f'i:[{i}]', [row['evidence_pages'] for row in dataset if row['question'] == distiset[i]['question']][0])
'''

