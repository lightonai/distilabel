import dotenv
from pathlib import Path
import re
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
)

EXCLUDE_PDFS = set(Path('/home/austin_veselka/lc/data/bench_pdfs.txt').read_text().splitlines())
CACHE_DIR = Path('/home/austin_veselka/lc/distilabel/out/lc_sft/v2')
DS_PATH = Path(CACHE_DIR / 'single_page_q_ds')
IMAGES_DS_PATH = Path('/home/austin_veselka/lc/data/all_pdfs_images_ds')
PDF_ROOT = Path('/home/austin_veselka/lc/data/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = (re.compile(r'^(/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/|/mnt/nfs/pdfs/)'), '/home/austin_veselka/lc/data/pdfs/')

PIPELINE_NAME = 'multi_page_a_v2'

def answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1)
):
    vllm_kwargs = {
        'limit-mm-per-prompt': "'{\"image\": 32, \"video\": 0}'",
        'max-model-len': '65536',
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
        task_name='answer_generation',
        temperature=temperature,
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs=vllm_kwargs,
        out_model=None,
        system_template_path=None,  # use default system prompt
        prompt_sampler_config=PromptSamplerConfig(), 
    )

stages = [
    Stage(
        lm_configs=[ # 72b, gemini flash, gpt 5 mini, Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
            # answer_lm_config('gemini-2.5-flash', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            # answer_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            answer_lm_config('Qwen/Qwen3-VL-235B-A22B-Instruct-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1400, 1400),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
