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
DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/distractors_short_q')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

stages = [
    Stage(
        lm_configs=[ # 72b, gemini flash, gpt 5 mini, Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
            LMConfig(
                path='Qwen/Qwen2.5-VL-32B-Instruct',
                # path='gpt-5-nano',
                data_ratio=1.0,
                task_name='answer_generation',
                temperature=1.0,
                max_new_tokens=4096,
                tp_size=1,
                replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1}'",
                    'quantization': 'fp8',
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.95,
                },
                out_model=None,
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=answer_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
