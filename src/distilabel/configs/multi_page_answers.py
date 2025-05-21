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

lm_configs=[
    LMConfig(
        path='Qwen/Qwen2.5-VL-72B-Instruct', 
        data_ratio=0.12, 
        task_name='answer_generation',
        temperature=0.1,
        max_new_tokens=4096,
        tp_size=4,
        replicas=1,
        vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
    LMConfig(
        path='OpenGVLab/InternVL3-78B', 
        data_ratio=0.12, 
        task_name='answer_generation',
        temperature=0.1,
        max_new_tokens=4096,
        tp_size=4,
        replicas=1,
        vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
    LMConfig(
        path='mistralai/Mistral-Small-3.1-24B-Instruct-2503', 
        data_ratio=0.11, 
        task_name='answer_generation',
        temperature=0.1,
        max_new_tokens=4096,
        tp_size=2,
        replicas=2,
        vllm_kwargs={ 
            'tokenizer-mode': 'mistral', 
            'config-format': 'mistral', 
            'load-format': 'mistral', 
            'tool-call-parser': 'mistral', 
            'limit-mm-per-prompt': "'image=64'"
        },
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
    LMConfig(
        path='gpt-4.1-mini', 
        data_ratio=0.25,
        task_name='answer_generation',
        temperature=0.1, 
        max_new_tokens=4096,
        replicas=1,
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
    LMConfig(
        path='gemini-2.5-flash-preview-04-17',
        data_ratio=0.25,
        task_name='answer_generation',
        temperature=0.1,
        max_new_tokens=4096,
        replicas=1,
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
    LMConfig(
        path='claude-3-5-haiku-latest', 
        data_ratio=0.15,
        task_name='answer_generation',
        temperature=0.1, 
        max_new_tokens=4096,
        replicas=1,
        out_model=None,
        prompt_sampler_config=answer_prompt_sampler_config,
    ),
]

AVAILABLE_GPUS = [0, 1, 2, 3]
stages = [
    Stage(
        lm_configs=lm_configs,
        default_system_template_path='distilabel/prompts/rag_focused_answer.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    ),
]

config = Config(stages=stages, debug_with_running_vllm=False)

