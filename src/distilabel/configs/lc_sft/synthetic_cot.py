import dotenv
from pathlib import Path
import re
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config,
    Stage,
    LMConfig,
    PromptSamplerConfig,
    CategoricalDist,
)

# vllm serve Qwen/Qwen2.5-VL-32B-Instruct -tp 2 --port 41256 --quantization fp8 --limit-mm-per-prompt '{"images": 336}'
# vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 -tp 2 -pp 2 --port 41256 --max-model-len 160000

import os
os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

EXCLUDE_PDFS = set(Path('/home/austin_veselka/lc/data/bench_pdfs.txt').read_text().splitlines())
CACHE_DIR = Path('/home/austin_veselka/lc/distilabel/out/lc_sft/v2')
SP_DS_PATH = Path(CACHE_DIR / 'single_page_q_ds')
MP_DS_PATH = Path(CACHE_DIR / 'true_multi_page_q_ds')
MP_DS_PATH_PT2 = Path(CACHE_DIR / 'true_multi_page_q_ds_2')
IMAGES_DS_PATH = Path('/home/austin_veselka/lc/data/all_pdfs_images_ds')
PDF_ROOT = Path('/home/austin_veselka/lc/data/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = (re.compile(r'^(/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/|/mnt/nfs/pdfs/)'), '/home/austin_veselka/lc/data/pdfs/')

TOP_K_PAGES = 24

PIPELINE_NAME = 'synthetic_cot_v2'

lc_mm_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(
            choices=[(f' and they have ranked the pages by relevance for you. You are given up to the top {TOP_K_PAGES} pages in addition to the extracted evidence.', 1)],
        ),
    },
)
text_only_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(choices=[(f'. You are given extracted evidence from up to the top {TOP_K_PAGES} pages.', 1)]),
    },
)

def lc_mm_overall_answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1),
):
    vllm_kwargs = {
        'limit-mm-per-prompt': "'{\"image\": top_k, \"video\": 0}'".replace('top_k', str(TOP_K_PAGES)),
        'gpu-memory-utilization': 0.9,
        'mm-processor-cache-gb': '0',
    }
    if 'FP8' not in path:
        vllm_kwargs |= {'quantization': 'fp8'}
    if 'Qwen3-VL' in path:
        vllm_kwargs |= {'enable-expert-parallel': None}
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='overall_answer_lc_mm',
        temperature=temperature,
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs=vllm_kwargs,
        out_model=None,
        system_template_path=None,
        prompt_sampler_config=PromptSamplerConfig(),
        # system_template_path='distilabel/prompts/lc_sft/minimal_combine_evidence_chunks.txt',
        # prompt_sampler_config=lc_mm_prompt_sampler_config,
    )

stages = [
    # Stage 0: collect evidence in chunks
    Stage(
        lm_configs=[ # 72b
            LMConfig(
                path='Qwen/Qwen3-VL-32B-Instruct-FP8',
                data_ratio=1.0,
                task_name='evidence_in_chunks',
                temperature=0.7,
                max_new_tokens=16384,
                # replicas=1,
                # tp_size=None,
                # replicas_per_vllm_server=1,
                replicas=16,
                tp_size=1,
                replicas_per_vllm_server=2,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1, \"video\": 0}'",
                    'max-model-len': '65536',
                    'gpu-memory-utilization': 0.9,
                    'mm-processor-cache-gb': '0',
                },
                out_model='EvidenceInChunks',
                system_template_path='distilabel/prompts/lc_sft/evidence_in_chunks.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1400, 1400),
    ),

    # Stage 1: overall answer
    # Qwen/Qwen3-235B-A22B-Instruct-2507-FP8, gemini flash, gpt 5 mini (must have temperature=1)
    Stage(
        lm_configs=[
            # gemini flash, gpt 5 mini
            # LC MM models
            # lc_mm_overall_answer_lm_config('gemini-2.5-flash', data_ratio=0.5, gpu_mesh=(1, None, 1)),
            # lc_mm_overall_answer_lm_config('gemini-2.5-flash-lite', data_ratio=0.5, gpu_mesh=(1, None, 1)),
            # lc_mm_overall_answer_lm_config('gemini-2.5-pro', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            lc_mm_overall_answer_lm_config('Qwen/Qwen3-VL-235B-A22B-Instruct-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),

            # qwen 235 instruct
            # text only models
            LMConfig(
                path='Qwen/Qwen3-235B-A22B-Instruct-2507-FP8',
                data_ratio=1.0,
                task_name='overall_answer_text_only',
                temperature=0.7,
                max_new_tokens=16384,
                replicas=4,
                tp_size=4,  # tp_size 8 not functional for this model
                # pp_size=2,
                replicas_per_vllm_server=2,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.9,
                    'enable-expert-parallel': None,
                },
                out_model=None,
                system_template_path='distilabel/prompts/lc_sft/minimal_combine_evidence_chunks.txt',
                prompt_sampler_config=text_only_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1400, 1400),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)

# Expecting $3 / 18K question = 437K images = 390M tok (though plus the cot, maybe more like 450M tok)