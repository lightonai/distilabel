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
from distilabel import utils

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
# DS_PATH = Path('/mnt/nfs/austin_shared/data/scraped_and_pdfa')
DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/make_ds_from_bench/mp')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

stages = [
    Stage(
        lm_configs=[
            LMConfig(
                # path='google/gemma-3-27b-it',
                path='Qwen/Qwen2.5-VL-32B-Instruct', 
                # path='Qwen/Qwen2.5-VL-7B-Instruct', 
                # path='openai/gpt-oss-120b',
                data_ratio=1.0, 
                task_name='make_persona',
                temperature=0.2,
                max_new_tokens=16384,
                tp_size=None,
                replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 336}'",
                    'quantization': 'fp8',
                    'max-model-len': '96000',
                    'gpu-memory-utilization': 0.95,
                },
                out_model='Persona',
                system_template_path='distilabel/prompts/make_persona.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True, path_substitution=PATH_SUBSTITUTION)

