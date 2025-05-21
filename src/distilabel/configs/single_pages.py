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

question_words = [
    'What', 
    'Who', 
    'Where', 
    'When', 
    'Why', 
    'How', 
    'Which', 
    'Do', 
    'Does', 
    'Is', 
    'Are', 
    'Has', 
    'Have', 
    'Will', 
    'Would', 
    'Can', 
    'Should',
]
start_question_with = [
    (f'begin your question with "{word}" (translated into the language of the page)', 1) 
    for word in question_words
]

question_prompt_sampler_config = PromptSamplerConfig(
    samples_per_prompt_kwarg='n_questions',
    distributions={
        'n_questions': CategoricalDist(
            choices=[('1', 1), ('2', 1), ('3', 1), ('4', 1), ('5', 1)]
        ),
        'question_spec': CategoricalDist(
            choices=[
                ("pick a section of the page and ask for a summary of that section", 1),
                ("ask for a summary of the entire page", 1),
                ("ask a question requiring comprehension of a specific section of the page", 1),
                ("ask a question requiring comprehension of the entire page and ask for a detailed response", 1),
                ("ask a question requiring multi-step reasoning about the page and ask for the model's thought process", 1),
                ("ask a question that requires an open ended answer and ask for a detailed response", 1),
                ("request a specific piece of information from the page", 1)
            ],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'start_question_with': CategoricalDist(
            choices=start_question_with + [("", len(start_question_with))],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'side_by_side_prefix': CategoricalDist(
            choices=[
                ("Given the number of questions to generate, the following are guidelines for each question in particular:", 1.0)
            ],
        )
    }
)
answer_prompt_sampler_config = PromptSamplerConfig()

def get_lm_configs(
    task_name: str, 
    prompt_sampler_config: PromptSamplerConfig, 
    temperature: float, 
    max_new_tokens: int,
    out_model: str,
):
    return [
        LMConfig(
            path='Qwen/Qwen2.5-VL-72B-Instruct', 
            data_ratio=0.12, 
            task_name=task_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tp_size=4,
            replicas=1,
            vllm_kwargs={'limit-mm-per-prompt': "'image=64'", 'enforce-eager': None},
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        LMConfig(
            path='OpenGVLab/InternVL3-78B', 
            data_ratio=0.12, 
            task_name=task_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tp_size=4,
            replicas=1,
            vllm_kwargs={'limit-mm-per-prompt': "'image=64'", 'enforce-eager': None},
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        LMConfig(
            path='mistralai/Mistral-Small-3.1-24B-Instruct-2503', 
            data_ratio=0.11, 
            task_name=task_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tp_size=2,
            replicas=2,
            vllm_kwargs={
                'tokenizer-mode': 'mistral', 
                'config-format': 'mistral', 
                'load-format': 'mistral', 
                'tool-call-parser': 'mistral', 
                'limit-mm-per-prompt': "'image=64'",
                'enforce-eager': None,
            },
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        LMConfig(
            path='gpt-4.1-mini', 
            data_ratio=0.25,
            temperature=temperature, 
            task_name=task_name,
            max_new_tokens=max_new_tokens,
            replicas=1,
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        LMConfig(
            path='gemini-2.5-flash-preview-04-17',
            data_ratio=0.25,
            temperature=temperature,
            task_name=task_name,
            max_new_tokens=max_new_tokens,
            replicas=1,
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        LMConfig(
            path='claude-3-5-haiku-latest', 
            data_ratio=0.15,
            temperature=temperature, 
            task_name=task_name,
            max_new_tokens=max_new_tokens,
            replicas=1,
            out_model=out_model,
            prompt_sampler_config=prompt_sampler_config,
        ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct', 
        #     data_ratio=0.12, 
        #     task_name=task_name,
        #     temperature=temperature,
        #     max_new_tokens=max_new_tokens,
        #     tp_size=4,
        #     replicas=1,
        #     vllm_kwargs={'limit-mm-per-prompt': "'image=64'", 'enforce-eager': None},
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct', 
        #     data_ratio=0.12, 
        #     task_name=task_name,
        #     temperature=temperature,
        #     max_new_tokens=max_new_tokens,
        #     tp_size=4,
        #     replicas=1,
        #     vllm_kwargs={'limit-mm-per-prompt': "'image=64'", 'enforce-eager': None},
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct', 
        #     data_ratio=0.11, 
        #     task_name=task_name,
        #     temperature=temperature,
        #     max_new_tokens=max_new_tokens,
        #     tp_size=2,
        #     replicas=2,
        #     vllm_kwargs={
        #         'tokenizer-mode': 'mistral', 
        #         'config-format': 'mistral', 
        #         'load-format': 'mistral', 
        #         'tool-call-parser': 'mistral', 
        #         'limit-mm-per-prompt': "'image=64'",
        #         'enforce-eager': None,
        #     },
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct', 
        #     data_ratio=0.25,
        #     temperature=temperature, 
        #     task_name=task_name,
        #     max_new_tokens=max_new_tokens,
        #     replicas=1,
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct',
        #     data_ratio=0.25,
        #     temperature=temperature,
        #     task_name=task_name,
        #     max_new_tokens=max_new_tokens,
        #     replicas=1,
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
        # LMConfig(
        #     path='Qwen/Qwen2.5-VL-7B-Instruct', 
        #     data_ratio=0.15,
        #     temperature=temperature, 
        #     task_name=task_name,
        #     max_new_tokens=max_new_tokens,
        #     replicas=1,
        #     out_model=out_model,
        #     prompt_sampler_config=prompt_sampler_config,
        # ),
    ]

OVERWRITE_SPLITS = {'distractors_short', 'hard_negs_short', 'adjacent_pages_short'}
'''
splits to overwrite on disk with the new questions and answers
'''
AVAILABLE_GPUS = [0, 1, 2, 3]
stages = [
    # Stage 0: Generate questions
    Stage(
        lm_configs=get_lm_configs(
            'question_generation', 
            question_prompt_sampler_config,
            temperature=0.6,
            max_new_tokens=2048,
            out_model='SinglePageQuestions',
        ),
        default_system_template_path='distilabel/prompts/single_page_questions.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    ),

    # Stage 1: Generate answers
    Stage(
        lm_configs=get_lm_configs(
            'answer_generation', 
            answer_prompt_sampler_config,
            temperature=0.1,
            max_new_tokens=4096,
            out_model=None,
        ),
        default_system_template_path='distilabel/prompts/single_page_answer.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1024, 1024),
    )
]

config = Config(stages=stages, debug_with_running_vllm=False)

