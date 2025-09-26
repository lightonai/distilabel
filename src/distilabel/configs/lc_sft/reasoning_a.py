import dotenv
from pathlib import Path
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config,
    Stage,
    LMConfig,
    PromptSamplerConfig,
    CategoricalDist,
)

# simply inserts the full prompt after 'Reasoning: high'
gpt_oss_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'system_prompt': CategoricalDist(
            choices=[
                ('/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/lc_sft/full_context_answer.txt', 1.0)
            ],
        ),
    },
)

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
                tp_size=2,
                replicas=2,
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
    # Qwen/Qwen3-235B-A22B-Thinking-2507-FP8, gpt-oss (remember to set system prompt to high reasoning)
    Stage(
        lm_configs=[
            LMConfig(
                path='openai/gpt-oss-120b',
                data_ratio=1.0,
                task_name='answer',
                temperature=1.0,
                max_new_tokens=65536,
                tp_size=None,
                replicas=1,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.92,
                },
                out_model=None,
                # system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
                system_template_path='distilabel/prompts/high_reasoning.txt',
                prompt_sampler_config=gpt_oss_prompt_sampler_config,
            ),
            LMConfig(
                path='Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
                data_ratio=4.0,
                task_name='answer',
                temperature=1.0,
                max_new_tokens=65536,
                tp_size=4,
                replicas=1,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.95,
                    'max-model-len': '220000',
                },
                out_model=None,
                # system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
                system_template_path='distilabel/prompts/high_reasoning.txt',
                prompt_sampler_config=gpt_oss_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
