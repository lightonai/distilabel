import dotenv
dotenv.load_dotenv()

from distilabel.pydantics import (
    Config, 
    Stage, 
    LMConfig, 
    PromptSamplerConfig,
)

lm_configs=[
    LMConfig(
        path='Qwen/Qwen2.5-VL-32B-Instruct', 
        data_ratio=1.0, 
        task_name='count_numbered_pages',
        temperature=0.0,
        max_new_tokens=128,
        tp_size=1,
        replicas=1,
        vllm_kwargs={
            'limit-mm-per-prompt': "'image=64'", 
            'quantization': 'fp8',
            'max-model-len': '96000',
            'gpu-memory-utilization': 0.95,
        },
        out_model='CountNumberedPages',
        prompt_sampler_config=PromptSamplerConfig(),
    ),
]

AVAILABLE_GPUS = [0]
stages = [
    Stage(
        lm_configs=lm_configs,
        default_system_template_path='distilabel/prompts/count_numbered_pages.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(640, 640),
    ),
]

config = Config(stages=stages, use_running_vllm=True)
