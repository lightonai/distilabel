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

# vllm serve Qwen/Qwen2.5-VL-32B-Instruct -tp 2 --port 41256 --quantization fp8 --limit-mm-per-prompt '{"images": 336}'
# vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 -tp 2 -pp 2 --port 41256 --max-model-len 160000

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
SP_DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/single_page_q_ds')
MP_DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft/true_multi_page_q_ds')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

TOP_K_PAGES = 8

PIPELINE_NAME = 'synthetic_cot_v0'

lc_mm_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(
            choices=[(f'\nYou are given the top {TOP_K_PAGES} most relevant pages directly, in addition to the relevant context.\n', 1)],
        ),
    },
)
text_only_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(choices=[('', 1)]),
    },
)

def lc_mm_overall_answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1),
):
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
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": top_k}'".replace('top_k', str(TOP_K_PAGES)),
            'max-model-len': '240000',
            'gpu-memory-utilization': 0.9,
        } | ({'quantization': 'fp8'} if 'FP8-Dynamic' not in path else {
            'max-num-batched-tokens': '8192',
            'max-num-seqs': '64',
            'enable-expert-parallel': None,
            'mm-processor-cache-gb': '0',
        }),
        out_model=None,
        system_template_path='distilabel/prompts/lc_sft/combine_evidence_chunks.txt',
        prompt_sampler_config=lc_mm_prompt_sampler_config,
    )

stages = [
    # Stage 0: collect evidence in chunks
    Stage(
        lm_configs=[ # 72b
            LMConfig(
                path='Qwen/Qwen2.5-VL-72B-Instruct',
                data_ratio=1.0,
                task_name='evidence_in_chunks',
                temperature=0.7,
                max_new_tokens=4096,
                replicas=8,
                tp_size=2,
                replicas_per_vllm_server=2,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                    'max-num-seqs': '64',
                },
                out_model='EvidenceInChunks',
                system_template_path='distilabel/prompts/lc_sft/evidence_in_chunks.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 1: overall answer
    # Qwen/Qwen3-235B-A22B-Instruct-2507-FP8, gemini flash, gpt 5 mini (must have temperature=1)
    Stage(
        lm_configs=[
            # gemini flash, gpt 5 mini
            # LC MM models
            # lc_mm_overall_answer_lm_config('gpt-5-mini', data_ratio=0.1, gpu_mesh=(1, None)),
            # lc_mm_overall_answer_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            lc_mm_overall_answer_lm_config('gemini-2.5-flash', data_ratio=0.5, gpu_mesh=(1, None, 1)),
            lc_mm_overall_answer_lm_config('gemini-2.5-flash-lite', data_ratio=0.5, gpu_mesh=(1, None, 1)),
            lc_mm_overall_answer_lm_config('RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic', data_ratio=1.0, gpu_mesh=(2, 8, 2)),

            # qwen 235 instruct
            # text only models
            LMConfig(
                path='Qwen/Qwen3-235B-A22B-Instruct-2507-FP8',
                data_ratio=1.0,
                task_name='overall_answer_text_only',
                temperature=0.7,
                max_new_tokens=16384,
                replicas=2,
                tp_size=4,  # tp_size 8 not functional for this model
                pp_size=2,
                replicas_per_vllm_server=2,
                vllm_kwargs={
                    'gpu-memory-utilization': 0.90,
                    'max-model-len': '240000',
                    # 'max-num-seqs': '64',
                },
                out_model=None,
                system_template_path='distilabel/prompts/lc_sft/combine_evidence_chunks.txt',
                prompt_sampler_config=text_only_prompt_sampler_config,
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)

# Expecting $3 / 18K question = 437K images = 390M tok (though plus the cot, maybe more like 450M tok)