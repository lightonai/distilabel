from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk, Dataset

from distilabel.pipeline import Pipeline
from distilabel.steps import (
	LoadDataFromDataset,
	Map,
	StepResources,
	FilterRows,
)
from distilabel.steps.tasks import LMGenerationTask

from distilabel.pydantics import Config
from distilabel.utils import pipe_utils
from distilabel import utils

from distilabel.configs.persona_generation.ujs import (
	config,
	DS_PATH,
	CACHE_DIR,
)

STAGE = 0
BATCH_SIZE = 128


def _resolve_path(path: str) -> str:
	if config.path_substitution is None:
		return path
	return path.replace(config.path_substitution[0], config.path_substitution[1])


def build_seed_ds(ds_path: Path) -> Dataset:
	ds = load_from_disk(ds_path)
	def to_source(row: dict[str, Any]) -> dict[str, Any]:
		return {'source': [_resolve_path(row['image_filename'])]}
	return ds.map(lambda row: to_source(row))


def run_pipeline(config: Config) -> Dataset:
	global STAGE
	STAGE = 0
	seed_ds = build_seed_ds(DS_PATH)

	with Pipeline(
		name='persona_generation_ujs',
		description='User Journey Stitcher: micro-personas → stitch → calibrate → finalize persona string',
		cache_dir=CACHE_DIR / 'ujs',
	) as pipeline:
		# Stage 0: Micro-personas (vision)
		stage = config.stages[STAGE]
		load_data = LoadDataFromDataset(name='load_data', dataset=seed_ds, batch_size=BATCH_SIZE)
		micro_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		generate_micro = [
			LMGenerationTask(
				use_cache=True,
				name=f'ujs_micro_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_micro = FilterRows(
			name='drop_none_micro',
			cols=['before', 'during', 'after'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'page_source'},
		)
		micro_to_text = Map(
			name='micro_to_text',
			cols=['before', 'during', 'after'],
			output_cols=['source'],
			fn=lambda output_cols=None, **row: {
				'source': json.dumps({'before': row['before'], 'during': row['during'], 'after': row['after']}, ensure_ascii=False),
			},
		)

		# Stage 1: Stitch (text-only)
		STAGE = 1
		stage = config.stages[STAGE]
		stitch_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		stitch_persona = [
			LMGenerationTask(
				use_cache=True,
				name=f'ujs_stitch_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_persona = FilterRows(
			name='drop_none_persona',
			cols=['persona'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'micro_persona'},
		)
		persona_to_text = Map(
			name='persona_to_text',
			cols=['persona'],
			output_cols=['persona_json'],
			fn=lambda output_cols=None, **row: {
				'persona_json': json.dumps(row['persona'], ensure_ascii=False),
			},
            output_mappings={'page_source': 'source'},
		)

		# Stage 2: Calibrate (text-only)
		STAGE = 2
		stage = config.stages[STAGE]
		calibrate_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		calibrate = [
			LMGenerationTask(
				use_cache=True,
				name=f'ujs_calibrate_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				lm_input_cols=['persona_json'],
				lm_input_col_prefixes=['persona: '],
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_calibration = FilterRows(
			name='drop_none_calibration',
			cols=['constraints', 'preferences', 'forbidden_page_ties'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
            output_mappings={'source': 'page_source'},
		)
		calibrated_to_text = Map(
			name='calibrated_to_text',
			cols=['constraints', 'preferences', 'forbidden_page_ties', 'persona_json'],
			output_cols=['source'],
			fn=lambda output_cols=None, **row: {
				'source': json.dumps({
					'persona': json.loads(row['persona_json']),
					'constraints': row.get('constraints', []),
					'preferences': row.get('preferences', []),
					'forbidden_page_ties': row.get('forbidden_page_ties', []),
				}, ensure_ascii=False),
			},
		)

		# Stage 3: Finalize (text-only)
		STAGE = 3
		stage = config.stages[STAGE]
		finalize_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		finalize = [
			LMGenerationTask(
				use_cache=True,
				name=f'ujs_finalize_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_persona_text = FilterRows(
			name='drop_none_persona_text',
			cols=['persona'],
			condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
			input_batch_size=BATCH_SIZE,
		)

		# Graph
		load_data >> micro_router >> generate_micro >> drop_none_micro >> micro_to_text
		micro_to_text >> stitch_router >> stitch_persona >> drop_none_persona >> persona_to_text
		persona_to_text >> calibrate_router >> calibrate >> drop_none_calibration >> calibrated_to_text
		calibrated_to_text >> finalize_router >> finalize >> drop_none_persona_text

	# run with load groups
	distiset, cost_tracker = pipeline.run(
		load_groups=(
			pipe_utils.steps_to_load_groups(
				[load_data, *generate_micro, drop_none_micro, micro_to_text],
				len(config.stages[0].available_gpus),
			) +
			pipe_utils.steps_to_load_groups(
				[*stitch_persona, drop_none_persona, persona_to_text],
				len(config.stages[1].available_gpus),
			) +
			pipe_utils.steps_to_load_groups(
				[*calibrate, drop_none_calibration, calibrated_to_text],
				len(config.stages[2].available_gpus),
			) +
			pipe_utils.steps_to_load_groups(
				[*finalize, drop_none_persona_text],
				len(config.stages[3].available_gpus),
			)
		),
		use_cache=True,
	)
	return distiset, cost_tracker


if __name__ == '__main__':
	ds, cost_tracker = run_pipeline(config)
	ds = ds['default']['train']
	ds = ds.remove_columns(['distilabel_metadata']) if 'distilabel_metadata' in ds.column_names else ds
	ds.save_to_disk(CACHE_DIR / 'ujs_personas_ds') 