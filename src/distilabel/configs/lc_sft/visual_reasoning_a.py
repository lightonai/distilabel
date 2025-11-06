import dotenv
from pathlib import Path
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
)

# import os
# os.environ['HF_HOME'] = '/opt/home/shared'

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/prompt_sampler_fixed')
SP_DS_PATH = Path(CACHE_DIR / 'single_page_q_ds')
MP_DS_PATH = Path(CACHE_DIR / 'true_multi_page_q_ds')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

PIPELINE_NAME = 'visual_reasoning_a_v1'

def answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1)
):
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='answer_generation',
        temperature=temperature,
        max_new_tokens=65536,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 336, \"video\": 0}'",
            'gpu-memory-utilization': 0.9,
            'max-num-batched-tokens': '4096',
            'max-num-seqs': '48',
            'enable-expert-parallel': None,
            'mm-processor-cache-gb': '0',
        },
        out_model=None,
        system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
        prompt_sampler_config=PromptSamplerConfig(),
    )

stages = [
    Stage(
        lm_configs=[ # qwen 3 vl 235b
            answer_lm_config('Qwen/Qwen3-VL-235B-A22B-Thinking-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
