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

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
DS_PATH = Path('/mnt/nfs/austin_shared/data/scraped_and_pdfa')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

TOP_K_PAGES = 4
MAX_TURNS = 6

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
followup_dist = {
    'followup': CategoricalDist(
        choices=[
            (f'ask a question to dig deeper on the previous question', 1),
            (f'ask a related question to the reference question as if the conversation sparked a new line of thought', 1),
        ],
    ),
}
sp_followup_question_prompt_sampler_config = PromptSamplerConfig(
    distributions=followup_dist | {
        'simple_or_complex': CategoricalDist(choices=[('simple', 1), ('complex', 1)]),
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
                ("ask a question regarding a table, graph, chart, diagram or other visual element. This should challenge the model to the utmost at precisely reading table rows, columns, specific values or sets of values, performing math operations, computing related mathematical/financial values, reasoning about the table data, reading elements from graphs, charts, diagrams, etc., extrapolating or interpolating graphs and charts, performing calculations based on graphs/charts/etc., answering questions conditional on values in one graph or table using values from another (e.g. what is the Q2 performance of the company with the highest Q1 performance in table/chart 12?), finding visual elements related to a specific topic, tracking/counting/finding entities from multiple pages and more. Be creative and come up with new types of questions that put the model to the test. (if no visual elements are present, you can ignore this and ask any question fitting the other requirements)", 1),
            ],
        ),
        'question_word_count': CategoricalDist(
            choices=[
                ("", 5),
                ("the question should be less than or equal to 8 words, you are a lazy user who doesn't want to type a long question", 1),
            ],
        ),
        'start_question_with': CategoricalDist(
            choices=start_question_with + [("", len(start_question_with) * 2)],
        ),
    },
)
mp_followup_question_prompt_sampler_config = PromptSamplerConfig(
    distributions=followup_dist | {
        'question_spec': CategoricalDist(choices=[
            ('', 4),
            ('Ask a question targeting tables, graphs, charts, diagrams or other visual elements, this should challenge the model to the utmost at precisely reading table rows, columns, specific values or sets of values, performing math operations, computing related mathematical/financial values, reasoning about the table data, reading elements from graphs, charts, diagrams, etc., extrapolating or interpolating graphs and charts, performing calculations based on graphs/charts/etc., answering questions conditional on values in one graph or table using values from another (e.g. what is the Q2 performance of the company with the highest Q1 performance in table/chart 12?), finding visual elements related to a specific topic, tracking/counting/finding entities from multiple pages and more. Be creative and come up with new types of questions that put the model to the test and for all of these: ESPECIALLY DOING THIS ACROSS MULTIPLE PAGES. (if no visual elements are present, you can ignore this and ask any question fitting the other requirements)', 1),
        ]),
    },
)

lc_mm_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'extra_info': CategoricalDist(
            choices=[(f'\nYou are given the top {TOP_K_PAGES} most relevant pages directly, in addition to the relevant context.\n', 1)],
        ),
    },
)

def question_lm_config(
    path: str,
    data_ratio: float = 1.0,
    gpu_mesh: tuple[int | None, int | None] = (1, 1),
    system_template_path: str = 'distilabel/prompts/lc_sft/sp_followup_question.txt',
    prompt_sampler_config: PromptSamplerConfig = sp_followup_question_prompt_sampler_config,
):
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='followup_question',
        temperature=0.7,
        max_new_tokens=16384,
        tp_size=gpu_mesh[1],
        replicas=gpu_mesh[0],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 5}'",
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.95,
            'quantization': 'fp8',
        },
        out_model='AnalysisQuestion',
        system_template_path=system_template_path,
        prompt_sampler_config=prompt_sampler_config,
    )

def lc_mm_overall_answer_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None] = (1, 1)
):
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='overall_answer_lc_mm',
        temperature=temperature,
        max_new_tokens=16384,
        tp_size=gpu_mesh[1],
        replicas=gpu_mesh[0],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": top_k}'".replace('top_k', str(TOP_K_PAGES)),
            'max-model-len': '240000',
            'gpu-memory-utilization': 0.95,
            'quantization': 'fp8',
        },
        out_model=None,
        system_template_path='distilabel/prompts/lc_sft/combine_evidence_chunks.txt',
        prompt_sampler_config=lc_mm_prompt_sampler_config,
    )

stages = [
    # Stage 0: followup questions
    Stage(
        lm_configs=[ # 72b
            # question generation has two parts, sp and mp style questions
            question_lm_config(
                path='Qwen/Qwen2.5-VL-32B-Instruct', 
                data_ratio=1.0, 
                gpu_mesh=(1, None), 
                system_template_path='distilabel/prompts/lc_sft/sp_followup_question.txt', 
                prompt_sampler_config=sp_followup_question_prompt_sampler_config
            ),
            question_lm_config(
                path='Qwen/Qwen2.5-VL-32B-Instruct', 
                data_ratio=1.0, 
                gpu_mesh=(1, None), 
                system_template_path='distilabel/prompts/lc_sft/mp_followup_question.txt', 
                prompt_sampler_config=mp_followup_question_prompt_sampler_config
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 1: collect evidence in chunks
    Stage(
        lm_configs=[ # 72b
            LMConfig(
                path='Qwen/Qwen2.5-VL-32B-Instruct',
                data_ratio=1.0,
                task_name='evidence_in_chunks',
                temperature=0.7,
                max_new_tokens=4096,
                tp_size=None,
                replicas=1,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.95,
                    'quantization': 'fp8',
                },
                out_model='EvidenceInChunks',
                system_template_path='distilabel/prompts/lc_sft/evidence_in_chunks.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 2: followup overall answer
    Stage(
        lm_configs=[
            # gemini flash, gpt 5 mini
            # LC MM models
            # lc_mm_overall_answer_lm_config('gpt-5-mini', data_ratio=0.1, gpu_mesh=(1, None)),
            # lc_mm_overall_answer_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            # lc_mm_overall_answer_lm_config('gemini-2.5-flash', data_ratio=0.1, gpu_mesh=(1, None)),
            # lc_mm_overall_answer_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None)),
            lc_mm_overall_answer_lm_config('Qwen/Qwen2.5-VL-32B-Instruct', data_ratio=1.0, gpu_mesh=(1, None)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True, path_substitution=PATH_SUBSTITUTION)

