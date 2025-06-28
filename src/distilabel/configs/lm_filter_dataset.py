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

lm_configs=[
    LMConfig(
        # path='Qwen/Qwen2.5-VL-32B-Instruct', 
        path='Qwen/Qwen2.5-VL-7B-Instruct',
        data_ratio=1.0, 
        task_name='label_reference_pages',
        temperature=0.1,
        max_new_tokens=128,
        tp_size=1,
        replicas=4,
        vllm_kwargs={
            'limit-mm-per-prompt': "'image=64'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model='Metalabel',
        prompt_sampler_config=PromptSamplerConfig(),
    ),
]

AVAILABLE_GPUS = [4, 5, 6, 7]
stages = [
    Stage(
        lm_configs=lm_configs,
        default_system_template_path='distilabel/prompts/label_reference_pages.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(640, 640),
    ),
]

config = Config(stages=stages, debug_with_running_vllm=True)

