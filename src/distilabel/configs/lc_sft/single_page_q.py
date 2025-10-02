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
                ("pick a section of the context and ask for a summary of that section", 1),
                ("ask for a summary of the entire context", 1),
                ("ask a question requiring comprehension of a specific section of the context", 1),
                ("ask a question requiring comprehension of the entire context and ask for a detailed response", 1),
                ("ask a question requiring multi-step reasoning about the page and ask for the model's thought process", 1),
                ("ask a question that requires an open ended answer and ask for a detailed response", 1),
                ("request a specific piece of information from the context", 1),
                ("ask a question requiring math", 1),
                ("ask a question regarding a table, graph, chart, diagram or other visual element. This should challenge the model to the utmost at precisely reading table rows, columns, specific values or sets of values, performing math operations, computing related mathematical/financial values, reasoning about the table data, reading elements from graphs, charts, diagrams, etc., extrapolating or interpolating graphs and charts, performing calculations based on graphs/charts/etc., answering questions conditional on values in one graph or table using values from another (e.g. what is the Q2 performance of the company with the highest Q1 performance in table/chart 12?), finding visual elements related to a specific topic, tracking/counting/finding entities from multiple pages and more. Be creative and come up with new types of questions that put the model to the test. (if no visual elements are present, this question spec should be an empty string)", 1),
            ],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'start_question_with': CategoricalDist(
            choices=start_question_with + [("", len(start_question_with) * 2)],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'question_word_count': CategoricalDist(
            choices=[
                ("", 5),
                ("the question should be less than or equal to 8 words, you are a lazy user who doesn't want to type a long question", 1),
            ],
            samples_per_prompt=None,
            side_by_side=True,
        ),
        'side_by_side_prefix': CategoricalDist(
            choices=[
                ("Given the number of questions to generate, the following are guidelines for each question in particular. The guidelines are organized as a list corresponding to each question. Each list denotes one or multiple requirements, where empty strings can be ignored:", 1.0)
            ],
        )
    }
)

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
DS_PATH = Path('/mnt/nfs/austin_shared/data/scraped_and_pdfa')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

PIPELINE_NAME = 'single_page_q_v0'

def get_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None] = (1, 1),
):
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path, 
        data_ratio=data_ratio, 
        task_name='question_generation',
        temperature=temperature,
        max_new_tokens=16384,
        tp_size=gpu_mesh[1],
        replicas=gpu_mesh[0],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 1}'",
            'quantization': 'fp8',
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.95,
        },
        out_model='SinglePageQuestions',
        system_template_path='distilabel/prompts/single_page_questions.txt',
        prompt_sampler_config=question_prompt_sampler_config,
    )

stages = [
    Stage(
        lm_configs=[ # 72b, 32b, gpt-5-nano, gemini-2.5-flash-lite
            get_lm_config('Qwen/Qwen2.5-VL-72B-Instruct', data_ratio=1.0, gpu_mesh=(1, 2)),
            get_lm_config('Qwen/Qwen2.5-VL-32B-Instruct', data_ratio=1.0, gpu_mesh=(2, 1)),
            # get_lm_config('gpt-5-nano', data_ratio=0.5, gpu_mesh=(1, None)),
            get_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None)),
            get_lm_config('gemini-2.5-flash', data_ratio=1.0, gpu_mesh=(1, None)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)

# ds of 500, 0.25 for gpt-5-nano, 0.25 for gemini flash lite, openai reports $0.09. Gemini likely the same.

# Expecting $1.44 / 7K question = 145K images = 130M tok
