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

lm_configs=[
    LMConfig(
        # path='google/gemma-3-27b-it',
        # path='Qwen/Qwen2.5-VL-32B-Instruct', 
        path='Qwen/Qwen2.5-VL-7B-Instruct', 
        data_ratio=1.0, 
        task_name='transcribe',
        temperature=0.0,
        max_new_tokens=4096,
        tp_size=1,
        replicas=1,
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 64}'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model=None,
        system_template_path='distilabel/prompts/transcribe.txt',
        prompt_sampler_config=PromptSamplerConfig(),
    ),
]

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
DS_PATH = '/mnt/nfs/austin_shared/mp_data_gen/out/scraped_v0.3_with_txt_img_neg_references_filtered'
IMAGES_DS_PATH = '/mnt/nfs/austin_shared/mp_data_gen/out/all_pdfs_images'
PDF_ROOT = '/mnt/nfs/pdfs'
CACHE_DIR = 'out'
AVAILABLE_GPUS = [4, 5, 6, 7]
stages = [
    Stage(
        lm_configs=lm_configs,
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True)

