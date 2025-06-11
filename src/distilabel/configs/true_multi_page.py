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
            choices=[('1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 1)],
            # choices=[('1', 2)]
        ),
    }
)

AVAILABLE_GPUS = [4, 5, 6, 7]

stages = [
    # Stage 0: Ask for questions that require multiple pages to answer
    Stage(
        lm_configs=[
            LMConfig(
                path='gpt-4.1-mini', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.3,
                temperature=0.1, 
                task_name='question_generation',
                max_new_tokens=4096,
                replicas=1,
                out_model='MultiPageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
            LMConfig(
                path='gemini-2.5-flash-preview-04-17',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.4,
                temperature=0.1,
                task_name='question_generation',
                max_new_tokens=4096,
                replicas=1,
                out_model='MultiPageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
            LMConfig(
                path='claude-3-5-haiku-latest', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.3,
                temperature=0.1, 
                task_name='question_generation',
                max_new_tokens=4096,
                replicas=1,
                out_model='MultiPageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
        ],
        default_system_template_path='distilabel/prompts/multi_page_questions.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    ),
    
    # Stage 1: Generate single page answers and break down question into requirements
    Stage(
        lm_configs=[
            # single page answer models
            LMConfig(
                path='Qwen/Qwen2.5-VL-32B-Instruct',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
                task_name='single_page_answer',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=2,
                replicas=1,
                vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
                out_model=None,  # No structured output needed
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='OpenGVLab/InternVL3-38B', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
                task_name='single_page_answer',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=2,
                replicas=1,
                vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
                out_model=None,
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='mistralai/Mistral-Small-3.1-24B-Instruct-2503', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
                task_name='single_page_answer',
                temperature=0.2,
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
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),

            # question requirements models
            LMConfig(
                path='gpt-4.1-nano', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1,
                temperature=0.1, 
                task_name='question_requirements',
                max_new_tokens=1024,
                replicas=1,
                out_model='QuestionRequirements',
                system_template_path='distilabel/prompts/question_requirements.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='gemini-2.0-flash-lite',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1,
                temperature=0.1,
                task_name='question_requirements',
                max_new_tokens=1024,
                replicas=1,
                out_model='QuestionRequirements',
                system_template_path='distilabel/prompts/question_requirements.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),

        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    ),
    
    # Stage 2: Judge answers for meeting requirements and filter out questions that can meet requirements with single page answers
    Stage(
        lm_configs=[
            # note: can use grok-3-mini-beta for this task, since it is a text only task
            LMConfig(
                # path='grok-3-mini-beta', 
                path='gpt-4.1-mini',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
                task_name='answer_judge',
                temperature=0.1,  # Low temperature for consistent judgments
                max_new_tokens=2048,
                replicas=1,
                out_model='SatisfactoryAnswer',  # Structured output for boolean judgment
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='gemini-2.5-flash-preview-04-17',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
                task_name='answer_judge',
                temperature=0.1,  # Low temperature for consistent judgments
                max_new_tokens=2048,
                replicas=1,
                out_model='SatisfactoryAnswer',  # Structured output for boolean judgment
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='claude-3-5-haiku-latest',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=1, 
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
        max_dims=(1024, 1024),
    ),

    # Stage 3: Generate reference answers with full context
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2.5-VL-72B-Instruct', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.12, 
                task_name='answer_generation',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=4,
                replicas=1,
                vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
                out_model=None,
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='OpenGVLab/InternVL3-78B', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.12, 
                task_name='answer_generation',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=4,
                replicas=1,
                vllm_kwargs={'limit-mm-per-prompt': "'image=64'"},
                out_model=None,
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='mistralai/Mistral-Small-3.1-24B-Instruct-2503', 
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.11, 
                task_name='answer_generation',
                temperature=0.2,
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
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='gpt-4.1-mini',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.25, 
                task_name='answer_generation',
                temperature=0.2,  # Lower temperature for more consistent answers
                max_new_tokens=4096,
                replicas=1,
                out_model=None,  # No structured output needed
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='gemini-2.5-flash-preview-04-17',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.25, 
                task_name='answer_generation',
                temperature=0.2,
                max_new_tokens=4096,
                replicas=1,
                out_model=None,
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            LMConfig(
                path='claude-3-5-haiku-latest',
                # path='Qwen/Qwen2-VL-7B-Instruct',
                data_ratio=0.15, 
                task_name='answer_generation',
                temperature=0.2,
                max_new_tokens=4096,
                replicas=1,
                out_model=None,
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        default_system_template_path='distilabel/prompts/rag_focused_answer.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    ),
]

config = Config(stages=stages, debug_with_running_vllm=False) 
