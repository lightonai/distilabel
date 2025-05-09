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

answer_prompt_sampler_config = PromptSamplerConfig()

AVAILABLE_GPUS = [4, 5]
task_name = 'answer_generation'
data_ratios = utils.normalize_distribution([1] * 1)
stages = [
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name=task_name,
                temperature=0.1,
                max_new_tokens=4096,
                tp_size=1,
                replicas=2,
                out_model=None,  # no structured generation, just the output
                prompt_sampler_config=answer_prompt_sampler_config,
            ),
        ],
        default_system_template_path='distilabel/prompts/simple_answer.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
]

config = Config(stages=stages)

