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

# can take a value from a table, plot or diagram

pos_extraction_ps = PromptSamplerConfig(
    distributions={
        'extraction_goal': CategoricalDist(
            choices=[
                ('first sentence (not including headers and separated specifically by periods)', 2),
                ('second sentence (not including headers and separated specifically by periods)', 2),
                ('third sentence (not including headers and separated specifically by periods)', 2),
                ('last sentence (not including headers and separated specifically by periods)', 2),
                ('second from last (meaning the one right before the last one) sentence (not including headers and separated specifically by periods)', 2),
                ('first sentence (not including headers and separated specifically by periods) of the first paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('second sentence (not including headers and separated specifically by periods) of the first paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('third sentence (not including headers and separated specifically by periods) of the first paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('last sentence (not including headers and separated specifically by periods) of the first paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('first sentence (not including headers and separated specifically by periods) of the second paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('second sentence (not including headers and separated specifically by periods) of the second paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('third sentence (not including headers and separated specifically by periods) of the second paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('last sentence (not including headers and separated specifically by periods) of the second paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('first sentence (not including headers and separated specifically by periods) of the last paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('second sentence (not including headers and separated specifically by periods) of the last paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('third sentence (not including headers and separated specifically by periods) of the last paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('last sentence (not including headers and separated specifically by periods) of the last paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('first sentence (not including headers and separated specifically by periods) of the second from last (meaning the one right before the last one) paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('second sentence (not including headers and separated specifically by periods) of the second from last (meaning the one right before the last one) paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('last sentence (not including headers and separated specifically by periods) of the second from last (meaning the one right before the last one) paragraph (paragraphs separated by \n\n and excluding headers)', 2),
                ('first paragraph (paragraphs separated by \n\n and excluding headers)', 4),
                ('second paragraph (paragraphs separated by \n\n and excluding headers)', 4),
                ('last paragraph (paragraphs separated by \n\n and excluding headers)', 4),
                ('second from last (meaning the one right before the last one) paragraph (paragraphs separated by \n\n and excluding headers)', 4),
            ],
        ),
    }
)

key_extraction_ps = PromptSamplerConfig(
    distributions={
        'sentence_or_paragraph': CategoricalDist(
            choices=[
                ('sentence', 1),
                ('paragraph', 1),
            ],
        ),
        'height_percent': CategoricalDist(
            choices=[
                ('10%', 1),
                ('20%', 1),
                ('30%', 1),
                ('40%', 1),
                ('50%', 1),
                ('60%', 1),
                ('70%', 1),
                ('80%', 1),
                ('90%', 1),
            ],
        ),
    }
)

lm_configs_0=[
    # Stage 0: Transcribe the page
    LMConfig(
        # path='google/gemma-3-27b-it',
        path='Qwen/Qwen2.5-VL-32B-Instruct', 
        # path='Qwen/Qwen2.5-VL-7B-Instruct', 
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

lm_configs_1=[
    # Stage 1: Key and Pos Retrieval
    LMConfig(  # key selection
        # path='google/gemma-3-27b-it',
        path='Qwen/Qwen2.5-VL-32B-Instruct', 
        # path='Qwen/Qwen2.5-VL-7B-Instruct', 
        data_ratio=1.0, 
        task_name='key_extraction',
        temperature=0.2,
        max_new_tokens=4096,
        tp_size=1,
        replicas=1,
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 64}'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model='KeyExtraction',
        system_template_path='distilabel/prompts/key_extraction.txt',
        prompt_sampler_config=key_extraction_ps,
    ),
    LMConfig(  # pos extraction
        # path='google/gemma-3-27b-it',
        path='Qwen/Qwen2.5-VL-32B-Instruct', 
        # path='Qwen/Qwen2.5-VL-7B-Instruct', 
        data_ratio=1.0, 
        task_name='pos_extraction',
        temperature=0.2,
        max_new_tokens=4096,
        tp_size=1,
        replicas=1,
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 64}'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model='PosExtraction',
        system_template_path='distilabel/prompts/pos_extraction.txt',
        prompt_sampler_config=pos_extraction_ps,
    ),
]

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
AVAILABLE_GPUS = [2, 3]
DS_PATH = '/mnt/nfs/austin_shared/mp_data_gen/out/scraped_v0.3_with_txt_img_neg_references_filtered'
IMAGES_DS_PATH = '/mnt/nfs/austin_shared/mp_data_gen/out/all_pdfs_images'
PDF_ROOT = '/mnt/nfs/pdfs'
CACHE_DIR = 'out'

stages = [
    # Stage 0: Transcribe the page
    Stage(
        lm_configs=lm_configs_0,
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
    Stage(
        lm_configs=lm_configs_1,
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True)

