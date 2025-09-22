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

DS_PATH = Path('/mnt/nfs/austin_shared/mp_data_gen/make_ds_from_bench/mp')
PDF_ROOT = Path('/mnt/nfs/pdfs')
CACHE_DIR = Path('/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/persona_generation')
AVAILABLE_GPUS = [0, 1, 2, 3]
PATH_SUBSTITUTION = ('/lustre/fsn1/projects/rech/eya/uzj46do/pdfs/', '/mnt/nfs/pdfs/')

stages = [
	Stage(
		lm_configs=[
			LMConfig(
				path='Qwen/Qwen2.5-VL-32B-Instruct',
				data_ratio=1.0,
				task_name='ujs_micro_personas',
				temperature=0.4,
				max_new_tokens=3072,
				vllm_kwargs={
					'limit-mm-per-prompt': '\'{"image": 1}\'',
					'max-model-len': '32000',
					'gpu-memory-utilization': 0.95,
					'quantization': 'fp8',
				},
				out_model='UJSMicroPersonas',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/ujs/micro_personas.txt',
				prompt_sampler_config=PromptSamplerConfig(),
			),
		],
		available_gpus=AVAILABLE_GPUS,
		max_dims=(1000, 1000),
	),
	Stage(
		lm_configs=[
			LMConfig(
				path='Qwen/Qwen2.5-VL-32B-Instruct',
				data_ratio=1.0,
				task_name='ujs_stitch',
				temperature=0.3,
				max_new_tokens=2048,
				out_model='UJSPersona',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/ujs/stitch_and_shape.txt',
				prompt_sampler_config=PromptSamplerConfig(),
			),
		],
		available_gpus=AVAILABLE_GPUS,
		max_dims=(1000, 1000),
	),
	Stage(
		lm_configs=[
			LMConfig(
				path='Qwen/Qwen2.5-VL-32B-Instruct',
				data_ratio=1.0,
				task_name='ujs_calibrate',
				temperature=0.2,
				max_new_tokens=1536,
				out_model='UJSConstraints',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/ujs/constraints_calibrator.txt',
				prompt_sampler_config=PromptSamplerConfig(),
			),
		],
		available_gpus=AVAILABLE_GPUS,
		max_dims=(1000, 1000),
	),
	Stage(
		lm_configs=[
			LMConfig(
				path='Qwen/Qwen2.5-VL-32B-Instruct',
				data_ratio=1.0,
				task_name='ujs_finalize',
				temperature=0.1,
				max_new_tokens=1024,
				out_model='PersonaText',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/ujs/finalize_persona.txt',
				prompt_sampler_config=PromptSamplerConfig(),
			),
		],
		available_gpus=AVAILABLE_GPUS,
		max_dims=(1000, 1000),
	),
]

config = Config(stages=stages, use_running_vllm=True, path_substitution=PATH_SUBSTITUTION) 