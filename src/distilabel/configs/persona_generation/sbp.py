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

EXCLUDE_PDFS = set(Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').read_text().splitlines()) if Path('/mnt/nfs/austin_shared/mp_data_gen/bench_pdfs.txt').exists() else set()
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
				task_name='sbp_scenarios',
				temperature=0.4,
				max_new_tokens=4096,
				tp_size=None,
				replicas=1,
				vllm_kwargs={
					'limit-mm-per-prompt': '\'{"image": 1}\'',
					'max-model-len': '32000',
					'gpu-memory-utilization': 0.95,
					'quantization': 'fp8',
				},
				out_model='SBPScenarios',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/sbp/scenario_sampler.txt',
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
				task_name='sbp_select_backcast',
				temperature=0.2,
				max_new_tokens=4096,
				replicas=1,
				out_model='SBPSelectAndBackcast',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/sbp/select_and_backcast.txt',
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
				task_name='sbp_audit',
				temperature=0.1,
				max_new_tokens=2048,
				replicas=1,
				out_model='SBPAudit',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/sbp/anti_overfit_audit.txt',
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
				task_name='sbp_rewrite',
				temperature=0.2,
				max_new_tokens=4096,
				replicas=1,
				out_model='SBPRewritePersona',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/persona_generation/sbp/rewrite_persona.txt',
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
				task_name='persona_finalize_text',
				temperature=0.1,
				max_new_tokens=2048,
				replicas=1,
				out_model='PersonaText',
				system_template_path='/mnt/nfs/austin_shared/mp_data_gen/distilabel/prompts/make_persona.txt',
				prompt_sampler_config=PromptSamplerConfig(),
			),
		],
		available_gpus=AVAILABLE_GPUS,
		max_dims=(1000, 1000),
	),
]

config = Config(stages=stages, use_running_vllm=True, path_substitution=PATH_SUBSTITUTION) 