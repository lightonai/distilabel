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
SP_DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/single_page_q_ds')
MP_DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/true_multi_page_q_ds')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

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
                tp_size=1,
                replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.95,
                    'quantization': 'fp8',
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
            LMConfig(
                # path='Qwen/Qwen2.5-VL-32B-Instruct',
                path='Qwen/Qwen3-235B-A22B-Instruct-2507-FP8',
                data_ratio=1.0,
                task_name='answer',
                temperature=0.3,
                max_new_tokens=16384,
                tp_size=4,
                replicas=1,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.95,
                    'max-model-len': '220000',
                },
                out_model=None,
                system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
