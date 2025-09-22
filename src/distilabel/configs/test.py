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
            choices=start_question_with + [("", 16)],
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

OVERWRITE_SPLITS = {'distractors_short', 'hard_negs_short', 'adjacent_pages_short'}
'''
splits to overwrite on disk with the new questions and answers
'''
AVAILABLE_GPUS = [4, 5]
task_name = 'question_generation'
data_ratios = utils.normalize_distribution([1] * 1)
stages = [
    Stage(
        lm_configs=[
            LMConfig(
                path='Qwen/Qwen2.5-VL-7B-Instruct', 
                data_ratio=1, 
                task_name=task_name,
                temperature=0.6,
                max_new_tokens=2048,
                tp_size=2,
                replicas=1,
                out_model='SinglePageQuestions',
                prompt_sampler_config=question_prompt_sampler_config,
            ),
            # LMConfig(
            #     path='OpenGVLab/InternVL3-38B', 
            #     data_ratio=1, 
            #     task_name=task_name,
            #     temperature=0.6,
            #     max_new_tokens=2048,
            #     tp_size=2,
            #     replicas=1,
            #     out_model='SinglePageQuestions',
            #     prompt_sampler_config=question_prompt_sampler_config,
            # ),
        ],
        default_system_template_path='distilabel/prompts/single_page_questions.txt',
        available_gpus=AVAILABLE_GPUS,
        max_dims=(512, 512),
    ),
]

config = Config(stages=stages, use_running_vllm=False)

