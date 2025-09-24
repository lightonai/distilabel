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

# vllm serve Qwen/Qwen2.5-VL-32B-Instruct -tp 2 --port 41256 --quantization fp8 --limit-mm-per-prompt '{"images": 336}'


# Stage 0 question generation prompt sampler
question_prompt_sampler_config = PromptSamplerConfig(
    samples_per_prompt_kwarg='n_questions',
    distributions={
        # one of the ways to reduce the cost of this pipeline is to sample more questions
        'n_questions': CategoricalDist(choices=[(str(i), min(i, 5)) for i in range(1, 6+1)]),
        'additional_visual_question': CategoricalDist(choices=[
            ('For the last of your questions, ask a question targeting tables, graphs, charts, diagrams or other visual elements, this should challenge the model to the utmost at precisely reading table rows, columns, specific values or sets of values, performing math operations, computing related mathematical/financial values, reasoning about the table data, reading elements from graphs, charts, diagrams, etc., extrapolating or interpolating graphs and charts, performing calculations based on graphs/charts/etc., answering questions conditional on values in one graph or table using values from another (e.g. what is the Q2 performance of the company with the highest Q1 performance in table/chart 12?), finding visual elements related to a specific topic, tracking/counting/finding entities from multiple pages and more. Be creative and come up with new types of questions that put the model to the test and for all of these: ESPECIALLY DOING THIS ACROSS MULTIPLE PAGES. (if no visual elements are present, this additional question should be an empty string)', 1),
            ('', 1),
        ]),
    },
)

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
DS_PATH = Path('/mnt/nfs/austin_shared/data/scraped_and_pdfa')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [4, 5, 6, 7]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

def question_generation_lm_config(
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
            'limit-mm-per-prompt': "'{\"image\": 10}'",
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.9,
            'quantization': 'fp8',
        },
        out_model='MultiPageQuestions',
        system_template_path='distilabel/prompts/multi_page_questions.txt',
        prompt_sampler_config=question_prompt_sampler_config,
    )

def judge_answers_lm_config(
    path: str,
    data_ratio: float = 1.0,
    gpu_mesh: tuple[int | None, int | None] = (1, 1),
):
    temperature = 0.1
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path,
        data_ratio=data_ratio,
        task_name='answer_judge',
        temperature=temperature,
        max_new_tokens=16384,
        tp_size=gpu_mesh[1],
        replicas=gpu_mesh[0],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 0}'",
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.9,
            'quantization': 'fp8',
        },
        out_model='SatisfactoryAnswer',
        system_template_path='distilabel/prompts/satisfied_user.txt',
        prompt_sampler_config=PromptSamplerConfig(),
    )

# Models for stages 0–2 (no final answer generation here)
stages = [
    # Stage 0: multi-page question generation
    Stage(
        lm_configs=[
            # 72b, gpt-5-nano, gemini-2.5-flash-lite
            question_generation_lm_config('Qwen/Qwen2.5-VL-72B-Instruct', data_ratio=4.0, gpu_mesh=(2, 2)),
            question_generation_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            question_generation_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 1: single page answers + question requirements
    Stage(
        lm_configs=[
            # 32b
            # single page answer model
            LMConfig(
                path='Qwen/Qwen2.5-VL-32B-Instruct',
                data_ratio=1.0,
                task_name='single_page_answer',
                temperature=0.2,
                max_new_tokens=4096,
                tp_size=1,
                replicas=4,
                vllm_kwargs={
                    'limit-mm-per-prompt': "'{\"image\": 1}'",
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                },
                out_model=None,
                system_template_path='distilabel/prompts/single_page_answer.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            # question requirements (text-only)
            # LMConfig(
            #     path='Qwen/Qwen2.5-VL-32B-Instruct',
            #     data_ratio=1.0,
            #     task_name='question_requirements',
            #     temperature=1.0,
            #     max_new_tokens=8192,
            #     tp_size=1,
            #     replicas=2,
            #     vllm_kwargs={
            #         'limit-mm-per-prompt': "'{\"image\": 0}'",
            #         'max-model-len': '32768',
            #         'gpu-memory-utilization': 0.9,
            #         'quantization': 'fp8',
            #     },
            #     out_model='QuestionRequirements',
            #     system_template_path='distilabel/prompts/question_requirements.txt',
            #     prompt_sampler_config=PromptSamplerConfig(),
            # ),
            LMConfig(
                path='Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
                data_ratio=1.0,
                task_name='question_requirements',
                temperature=0.7,
                max_new_tokens=16384,
                tp_size=1,
                replicas=2,
                vllm_kwargs={
                    'max-model-len': '32768',
                    'gpu-memory-utilization': 0.9,
                    'quantization': 'fp8',
                },
                out_model='QuestionRequirements',
                system_template_path='distilabel/prompts/question_requirements.txt',
                prompt_sampler_config=PromptSamplerConfig(),
            ),
            # LMConfig(
            #     path='gpt-4.1-mini',
            #     data_ratio=1.0,
            #     task_name='question_requirements',
            #     temperature=0.7,
            #     max_new_tokens=8192,
            #     tp_size=None,
            #     replicas=1,
            #     vllm_kwargs={
            #         'limit-mm-per-prompt': "'{\"image\": 0}'",
            #         'max-model-len': '32768',
            #         'gpu-memory-utilization': 0.9,
            #         'quantization': 'fp8',
            #     },
            #     out_model='QuestionRequirements',
            #     system_template_path='distilabel/prompts/question_requirements.txt',
            #     prompt_sampler_config=PromptSamplerConfig(),
            # ),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),

    # Stage 2: judge answers for meeting requirements
    Stage(
        lm_configs=[
            # gpt-5-mini, gemini-2.5-flash, 72b
            judge_answers_lm_config('Qwen/Qwen2.5-VL-72B-Instruct', data_ratio=4.0, gpu_mesh=(2, 2)),
            judge_answers_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            judge_answers_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=False, path_substitution=PATH_SUBSTITUTION)

# $3.09
