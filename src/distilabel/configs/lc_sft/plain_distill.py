import dotenv
from pathlib import Path
import re
import os
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
)

os.environ['HF_HUB_OFFLINE'] = '1'

EXCLUDE_PDFS = set(Path('/home/austin_veselka/lc/data/bench_pdfs.txt').read_text().splitlines())
IMAGES_DS_PATH = Path('/home/austin_veselka/lc/data/all_pdfs_images_ds')
CACHE_DIR = Path('/home/austin_veselka/lc/distilabel/out/lc_sft/v2')
# SP_DS_PATH = Path(CACHE_DIR / 'single_page_q_ds')
# MP_DS_PATH = Path(CACHE_DIR / 'true_multi_page_q_ds')
DS_PATH = Path(CACHE_DIR / 'longpo_q_ds_50k')
PDF_ROOT = Path('/home/austin_veselka/lc/data/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = (re.compile(r'^(/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/|/mnt/nfs/pdfs/)'), '/home/austin_veselka/lc/data/pdfs/')

PIPELINE_NAME = 'plain_distill_longpo_q_ds_50k_question_source'
# PIPELINE_NAME = 'plain_distill_longpo_ds_50k_full_source'

def answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1)
):
    vllm_kwargs = {
        'limit-mm-per-prompt': "'{\"image\": 32, \"video\": 0}'",  # change when you switch to full source
        'gpu-memory-utilization': 0.9,
        'mm-processor-cache-gb': '0',
    }
    if 'Qwen3-VL' in path:
        vllm_kwargs |= {'enable-expert-parallel': None}
    if 'FP8' not in path:
        vllm_kwargs |= {'quantization': 'fp8'}
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='answer_generation',
        temperature=temperature,
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs=vllm_kwargs,
        out_model=None,
        system_template_path=None,
        prompt_sampler_config=PromptSamplerConfig(),
    )

stages = [
    Stage(
        lm_configs=[ # qwen 3 vl 235b
            answer_lm_config('Qwen/Qwen3-VL-235B-A22B-Instruct-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
