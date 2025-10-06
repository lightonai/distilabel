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

question_prompt_sampler_config = PromptSamplerConfig(
    distributions={
        'question_spec': CategoricalDist(
            choices=[
                ("in your analysis, design a question requiring comprehension of a specific section of the context, your question should be", 1),
                ("in your analysis, design a question requiring comprehension of the entire context and ask for a detailed response, your question should be", 1),
                ("in your analysis, design a question requiring multi-step reasoning about the page and ask for the model's thought process, your question should be", 1),
                ("in your analysis, design a question that requires an open ended answer and ask for a detailed response, your question should be", 1),
                ("in your analysis, request a specific piece of information from the context, your question should be", 1),
                ("in your analysis, design a question requiring math, your question should be", 1),
                ("in your analysis, design a question regarding a table, graph, chart, diagram or other visual element. This should challenge the model to the utmost at precisely reading table rows, columns, specific values or sets of values, performing math operations, computing related mathematical/financial values, reasoning about the table data, reading elements from graphs, charts, diagrams, etc., extrapolating or interpolating graphs and charts, performing calculations based on graphs/charts/etc., answering questions conditional on values in one graph or table using values from another (e.g. what is the Q2 performance of the company with the highest Q1 performance in table/chart 12?), finding visual elements related to a specific topic, tracking/counting/finding entities from multiple pages and more. (if no visual elements are present, this question spec should be an empty string). Your question should be", 1),
            ],
        ),
        'question_type': CategoricalDist(
            choices=[
                ("a trick question version of this which cannot actually be answered with the context", 1),
                ("only partially answerable with the context (meaning you request information that is not present/cannot be surmised from the context)", 1),
                ("a modified version of the question that asks for information that is highly unlikely to be present inside the full document", 1),
            ],
        ),
        'question_word_count': CategoricalDist(
            choices=[
                ("", 4),
                ("the question should be less than or equal to 8 words, you are a lazy user who doesn't want to type a long question", 1),
            ],
        ),
    }
)

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines())
DS_PATH = Path('/mnt/nfs/austin_shared/data/scraped_and_pdfa')
IMAGES_DS_PATH = Path('/mnt/nfs/austin_shared/data/all_pdfs_images_ds')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/lc_sft')
AVAILABLE_GPUS = [0, 1, 2, 3]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

PIPELINE_NAME = 'unanswerable_v0'

def get_lm_config(
    path: str, 
    data_ratio: float = 1.0, 
    gpu_mesh: tuple[int | None, int | None, int | None] = (1, 1, 1),
):
    temperature = 0.7
    if 'gpt-5' in path:
        temperature = 1.0
    return LMConfig(
        path=path, 
        data_ratio=data_ratio, 
        task_name='qa_generation',
        temperature=temperature,
        max_new_tokens=16384,
        replicas=gpu_mesh[0],
        tp_size=gpu_mesh[1],
        replicas_per_vllm_server=gpu_mesh[2],
        vllm_kwargs={
            'limit-mm-per-prompt': "'{\"image\": 1}'",
            'max-model-len': '32768',
            'gpu-memory-utilization': 0.95,
        } | ({'quantization': 'fp8'} if 'FP8-Dynamic' not in path else {
            'max-num-batched-tokens': '4096',
            'max-num-seqs': '8',
            'enable-expert-parallel': None,
            'mm-processor-cache-gb': '0',
        }),
        out_model='UnanswerableQA',
        system_template_path='distilabel/prompts/unanswerable_qa.txt',
        prompt_sampler_config=question_prompt_sampler_config,
    )

stages = [
    Stage(
        lm_configs=[ # 72b, 32b, gpt-5-nano, gemini-2.5-flash-lite
            # get_lm_config('Qwen/Qwen2.5-VL-72B-Instruct', data_ratio=1.0, gpu_mesh=(1, 2)),
            # get_lm_config('Qwen/Qwen2.5-VL-32B-Instruct', data_ratio=1.0, gpu_mesh=(2, 1)),
            # get_lm_config('gpt-5-nano', data_ratio=1.0, gpu_mesh=(1, None)),
            get_lm_config('gemini-2.5-flash', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            get_lm_config('gemini-2.5-flash-lite', data_ratio=1.0, gpu_mesh=(1, None, 1)),
            get_lm_config('RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-Dynamic', data_ratio=2.0, gpu_mesh=(2, 4, 2)),
        ],
        available_gpus=AVAILABLE_GPUS,
        max_dims=(1000, 1000),
    ),
]

config = Config(stages=stages, use_running_vllm=True, path_substitution=PATH_SUBSTITUTION)

