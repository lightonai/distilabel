import dotenv
import re
import os
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

os.environ['HF_HUB_OFFLINE'] = '1'


MAGPIE_QWEN3_VL_CHAT_TEMPLATE = r"""
{%- set image_count = namespace(value=0) %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>user\n' }}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {%- for content in message.content %}
                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}
                    {%- set image_count.value = image_count.value + 1 %}
                    <|vision_start|><|image_pad|><|vision_end|>
                {%- elif 'text' in content %}
                    {{- content.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{{- '<|im_start|>user\n' }}
"""


EXCLUDE_PDFS = set(Path('/home/austin_veselka/lc/data/bench_pdfs.txt').read_text().splitlines())
DS_PATH = Path('/home/austin_veselka/lc/data/scraped_and_pdfa_long_upsampled_sqrt_multinomial')
IMAGES_DS_PATH = Path('/home/austin_veselka/lc/data/all_pdfs_images_ds')
PDF_ROOT = Path('/home/austin_veselka/lc/data/pdfs')
CACHE_DIR = Path('/home/austin_veselka/lc/distilabel/out/lc_sft/v2')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = (re.compile(r'^(/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/|/mnt/nfs/pdfs/)'), '/home/austin_veselka/lc/data/pdfs/')

PIPELINE_NAME = 'magpie_q'

def get_lm_config(
    path: str, 
    chat_template: str,
    data_ratio: float = 1.0, 
    temperature: float = 1.5,
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1),
):
    vllm_kwargs = {
        'limit-mm-per-prompt': "'{\"image\": 32, \"video\": 0}'",
        'max-model-len': '65536',
        'gpu-memory-utilization': 0.9,
        'mm-processor-cache-gb': '0',
        'trust-request-chat-template': None,
    }
    if 'FP8' not in path:
        vllm_kwargs |= {'quantization': 'fp8'}
    return LMConfig(
        path=path, 
        data_ratio=data_ratio, 
        task_name='question_generation',
        temperature=temperature,
        max_new_tokens=512,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs=vllm_kwargs,
        api_call_extra_body={'min_tokens': 10, 'chat_template': chat_template, 'top_k': -1},
        out_model=None,
        system_template_path=None,
        prompt_sampler_config=PromptSamplerConfig(),
    )

def answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1)
):
    vllm_kwargs = {
        'limit-mm-per-prompt': "'{\"image\": 336, \"video\": 0}'",
        'gpu-memory-utilization': 0.9,
        'mm-processor-cache-gb': '0',
    }
    if 'Qwen3-VL' in path:
        vllm_kwargs |= {'enable-expert-parallel': None}
    if 'FP8' not in path:
        vllm_kwargs |= {'quantization': 'fp8'}
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='answer_generation',
        temperature=temperature,
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs=vllm_kwargs,
        out_model=None,
        system_template_path=None,
        prompt_sampler_config=PromptSamplerConfig(),
    )

stages = [
    Stage(
        lm_configs=[
            get_lm_config('Qwen/Qwen3-VL-32B-Instruct-FP8', data_ratio=1.0, gpu_mesh=(8, 2, 2), chat_template=MAGPIE_QWEN3_VL_CHAT_TEMPLATE, temperature=1.5),
            get_lm_config('Qwen/Qwen2.5-VL-72B-Instruct', data_ratio=0.1, gpu_mesh=(8, 2, 2), chat_template=MAGPIE_QWEN3_VL_CHAT_TEMPLATE, temperature=0.7),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1344, 1344),
    ),

    Stage(
        lm_configs=[ # qwen 3 vl 235b
            answer_lm_config('Qwen/Qwen3-VL-235B-A22B-Instruct-FP8', data_ratio=1.0, gpu_mesh=(4, 4, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)

# ds of 500, 0.25 for gpt-5-nano, 0.25 for gemini flash lite, openai reports $0.09. Gemini likely the same.

# Expecting $1.44 / 7K question = 145K images = 130M tok
