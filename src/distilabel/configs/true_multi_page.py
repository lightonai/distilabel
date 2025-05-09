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

# Configuration for generating multi-page questions
question_prompt_sampler_config = PromptSamplerConfig(
    samples_per_prompt_kwarg='n_questions',
    distributions={
        'n_questions': CategoricalDist(
            choices=[('2', 1)]  # ['1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 1)]
        ),
    }
)

AVAILABLE_GPUS = [4, 5]
data_ratios = utils.normalize_distribution([1] * 1)

stages = [
    # Stage 0: Generate questions that require multiple pages to answer
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name='question_generation',
                temperature=0.6,  # Slightly higher temperature for more creative questions
                max_new_tokens=4096,
                replicas=1,
                out_model='MultiPageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
        ],
        default_system_template_path='distilabel/prompts/multi_page_questions.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
    
    # Stage 1: Generate single page answers and break down question into requirements
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name='single_page_answer',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=1,
                replicas=1,
                out_model=None,  # No structured output needed
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name='question_requirements',
                temperature=0.3,
                max_new_tokens=1024,
                replicas=1,
                out_model='QuestionRequirements',
                system_template_path='distilabel/prompts/question_requirements.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
    
    # Stage 2: Judge answers and filter out questions that are not multi-page
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name='answer_judge',
                temperature=0.1,  # Low temperature for consistent judgments
                max_new_tokens=2048,
                replicas=1,
                out_model='SatisfactoryAnswer',  # Structured output for boolean judgment
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        default_system_template_path='distilabel/prompts/satisfied_user.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),

    # Stage 3: Generate reference answers with full context
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2-VL-7B-Instruct', 
                data_ratio=data_ratios[0], 
                task_name='answer_generation',
                temperature=0.2,  # Lower temperature for more consistent answers
                max_new_tokens=4096,
                replicas=1,
                out_model=None,  # No structured output needed
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        default_system_template_path='distilabel/prompts/rag_focused_answer.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
]

config = Config(stages=stages, debug_with_running_vllm=True) 