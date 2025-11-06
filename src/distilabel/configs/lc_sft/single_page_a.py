import dotenv
from pathlib import Path
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
)

# Use simple prompt sampler (no distributions needed for answers)
answer_prompt_sampler_config = PromptSamplerConfig()

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/prompt_sampler_fixed')
DS_PATH = Path(CACHE_DIR / 'distractors_short_q')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

PIPELINE_NAME = 'single_page_a_v1'

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
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 1, \"video\": 0}'",
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.92,
        } | ({'quantization': 'fp8'} if 'FP8-Dynamic' not in path else {
            'max-num-batched-tokens': '4096',
            'max-num-seqs': '128',
            'enable-expert-parallel': None,
            'mm-processor-cache-gb': '0',
        }),
        out_model=None,
        system_template_path='distilabel/prompts/rag_focused_answer.txt',
        prompt_sampler_config=answer_prompt_sampler_config,
    )

stages = [
    Stage(
        lm_configs=[ # 72b, gemini flash, gpt 5 mini, RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic
            answer_lm_config('gemini-2.5-flash', data_ratio=0.5, gpu_mesh=(1, None, 1)),
            answer_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            # answer_lm_config('gpt-5-mini', data_ratio=0.1, gpu_mesh=(1, None)),
            # answer_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            answer_lm_config('RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic', data_ratio=2.0, gpu_mesh=(4, 4, 2)),
            # answer_lm_config('Qwen/Qwen2.5-VL-32B-Instruct', data_ratio=2.0, gpu_mesh=(2, 1, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
