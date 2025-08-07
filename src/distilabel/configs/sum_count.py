import dotenv
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
    CategoricalDist,
)
from distilabel import utils
from pathlib import Path

sum_count_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'object': CategoricalDist(
            choices=[
                ('sentences (separated specifically by periods)', 1),
                ('paragraphs (defined by a group of text that begins with a header, an indentation that is not under another paragraph (i.e. a single indentation) or a gap between groups of multiple sentences on the page)', 1),
                ('tables (with a caption like "Table ...")', 1),
                ('figures (with a caption like "Figure ..." or "Fig ...")', 1),
                ('equations (numbered equations with dedicated space, not inline)', 1),
                ('footnotes (standard bottom of the page footnotes, these must be a note which clarifies something, referred to by a small superscript number)', 1),
                ('captions (these must be about the entity (a figure, table, diagram, image, etc.) they are below, not just nearby text, a label, or a document title)', 1),
            ],
        ),
    }
)

lm_configs=[
    LMConfig(
        # path='google/gemma-3-27b-it',
        path='Qwen/Qwen2.5-VL-32B-Instruct', 
        # path='Qwen/Qwen2.5-VL-7B-Instruct', 
        data_ratio=1.0, 
        task_name='count',
        temperature=0.0,
        max_new_tokens=4096,
        tp_size=None,
        replicas=16,
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 64}'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model='ThinkingCount',
        prompt_sampler_config=PromptSamplerConfig(),
        path_substitution=('/mnt/nfs/pdfs/', '/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/'),
    ),
]

work_dir = Path('/lustre/fswork/projects/rech/eya/uzj46do/')
scratch_dir = Path('/lustre/fsn1/projects/rech/eya/uzj46do/')

SHUFFLE_FACTOR = 3
EXCLUDE_PDFS = set((work_dir / 'distilabel/bench_pdfs.txt').read_text().splitlines())
DS_PATH = work_dir / 'data' / 'scraped_and_pdfa'
IMAGES_DS_PATH = work_dir / 'data' / 'all_pdfs_images_ds'
PDF_ROOT = scratch_dir / 'pdfs'
CACHE_DIR = scratch_dir / 'distilabel/out'
AVAILABLE_GPUS = [0, 1, 2, 3]

stages = [
    Stage(
        lm_configs=lm_configs,
        default_system_template_path='distilabel/prompts/sum_count.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True)

