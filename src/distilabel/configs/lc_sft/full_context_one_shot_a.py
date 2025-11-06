import dotenv
from pathlib import Path
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config,
    Stage,
    LMConfig,
    PromptSamplerConfig,
)

# vllm serve Qwen/Qwen2.5-VL-32B-Instruct -tp 2 --port 41256 --quantization fp8 --limit-mm-per-prompt '{"images": 336}'

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/prompt_sampler_fixed')
SP_DS_PATH = Path(CACHE_DIR / 'single_page_q_ds')
MP_DS_PATH = Path(CACHE_DIR / 'true_multi_page_q_ds')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

PIPELINE_NAME = 'full_context_one_shot_a_v1'

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
        task_name='answer',
        temperature=temperature,
        max_new_tokens=65536,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs={
            'gpu-memory-utilization': 0.94,
        } | ({'quantization': 'fp8'} if 'FP8' not in path else {
            'max-model-len': '240000',
            'max-num-seqs': '64',
            'max-num-batched-tokens': '4096',
            'enable-expert-parallel': None,
        }),
        out_model=None,
        system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
        prompt_sampler_config=PromptSamplerConfig(),
    )

stages = [
    # Stage 0: transcribe
    Stage(
        lm_configs=[ # 72b
            LMConfig(
                path='Qwen/Qwen2.5-VL-72B-Instruct',
                data_ratio=1.0,
                task_name='transcribe',
                temperature=0.2,
                max_new_tokens=4096,
                replicas=4,
                tp_size=2,
                replicas_per_vllm_server=2,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1, \"video\": 0}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                    'max-num-seqs': '96',
                },
                out_model=None,
                system_template_path='distilabel/prompts/transcribe.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 1: full text context answer
    # Qwen/Qwen3-235B-A22B-Instruct-2507-FP8, gemini flash, gpt 5 mini (temperature=1)
    Stage(
        lm_configs=[
            answer_lm_config('Qwen/Qwen3-235B-A22B-Instruct-2507-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),
            answer_lm_config('gemini-2.5-flash', data_ratio=1.0, gpu_mesh=(1, None, 1)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
