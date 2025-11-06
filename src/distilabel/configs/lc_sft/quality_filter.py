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
from distilabel.utils import add_index_badge_to_image

# simply inserts the full prompt after 'Reasoning: high'
gpt_oss_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'system_prompt': CategoricalDist(
            choices=[
                ('/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/lc_sft/answer_to_claims.txt', 1.0)
            ],
        ),
    },
)

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/prompt_sampler_fixed')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

TOP_K_PAGES = 16
# for ones that are too contentious (i.e. the original answer claims are entirely disputed) it is more likely one or both sides are wrong
# so we discard these below a threshold
CLAIMS_SUPPORTED_THRESHOLD = 0.6
PIPELINE_NAME = 'quality_filter_v1'

lc_mm_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(
            choices=[(f'\nYou are given the top {TOP_K_PAGES} most relevant pages directly, in addition to the relevant context.\n', 1)],
        ),
    },
)

stages = [
    # Stage 0: check answer language matches question language
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2.5-VL-32B-Instruct',
                # path='RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic',
                # path='Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
                data_ratio=1.0,
                task_name='check_language',
                temperature=0.2,
                max_new_tokens=4096,
                replicas=16,
                tp_size=1,
                replicas_per_vllm_server=2,
                # tp_size=None,
                # replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 0, \"video\": 0}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                },
                out_model='CheckAnswerLanguage',
                system_template_path='distilabel/prompts/lc_sft/check_language.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 1: answer to claims
    # gpt oss 120b
    Stage(
        lm_configs=[
            LMConfig(
                path='openai/gpt-oss-120b',
                # path='Qwen/Qwen2.5-VL-72B-Instruct',
                # path='RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic',
                # path='Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
                data_ratio=1.0,
                task_name='to_claims',
                temperature=1.0,
                max_new_tokens=65536,
                replicas=8,
                tp_size=2,
                replicas_per_vllm_server=2,
                # tp_size=None,
                # replicas=1,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.92,
                },
                out_model='AnswerToClaims',
                # system_template_path='distilabel/prompts/lc_sft/full_context_answer.txt',
                system_template_path='distilabel/prompts/high_reasoning.txt',
                prompt_sampler_config=gpt_oss_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 2: per-page extract evidence for claims
    # qwen 2.5 vl 72b
    Stage(
        lm_configs=[
            LMConfig(
                # path='Qwen/Qwen2.5-VL-72B-Instruct',
                path='Qwen/Qwen2.5-VL-72B-Instruct',
                # path='RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic',
                # path='Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
                data_ratio=1.0,
                task_name='extract_evidence_for_claims',
                temperature=1.0,
                max_new_tokens=16384,
                replicas=8,
                tp_size=2,
                replicas_per_vllm_server=2,
                # tp_size=None,
                # replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1, \"video\": 0}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                },
                out_model='EvidenceInChunks',
                system_template_path='distilabel/prompts/lc_sft/claims_evidence_in_chunks.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1344, 1344),
    ),

    # Stage 3: check claims supported by the source and provide corrections if needed
    # qwen3 vl
    Stage(
        lm_configs=[
            LMConfig(
                path='RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic',
                # path='Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
                # path='Qwen/Qwen2.5-VL-72B-Instruct',
                data_ratio=1.0,
                task_name='check_claims_supported',
                temperature=1.0,
                max_new_tokens=32768,
                replicas=4,
                tp_size=4,
                # pp_size=2,
                replicas_per_vllm_server=2,
                # tp_size=None,
                # replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 336, \"video\": 0}'",
                    'max-model-len': '240000',
                    'gpu-memory-utilization': 0.9,
                    'max-num-batched-tokens': '4096',
                    'max-num-seqs': '48',
                    'enable-expert-parallel': None,
                    'mm-processor-cache-gb': '0',
                },
                out_model='CheckClaims',
                system_template_path='distilabel/prompts/lc_sft/check_claims.txt',
                prompt_sampler_config=lc_mm_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1280, 1280),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)
