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
	NoOp,
)
from distilabel.steps.tasks import LMGenerationTask

from distilabel import utils
from distilabel.pydantics import Config
from distilabel.utils import pipe_utils

from distilabel.configs.persona_generation.sbp import (
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
	# Expect column 'image_filename' with path to a specific page image
	def to_source(row: dict[str, Any]) -> dict[str, Any]:
		return {'source': [_resolve_path(row['image_filename'])]}
	return ds.map(lambda row: to_source(row))


def run_pipeline(config: Config) -> Dataset:
	global STAGE
	STAGE = 0
	seed_ds = build_seed_ds(DS_PATH)

	with Pipeline(
		name='persona_generation_sbp',
		description='Scenario Backcast Personas: scenario → backcast persona → audit → rewrite → final persona string',
		cache_dir=CACHE_DIR / 'sbp',
	) as pipeline:
		# Stage 0: Scenario generation (vision)
		stage = config.stages[STAGE]
		load_data = LoadDataFromDataset(name='load_data', dataset=seed_ds, batch_size=BATCH_SIZE)
		scenarios_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		generate_scenarios = [
			LMGenerationTask(
				use_cache=True,
				name=f'sbp_scenarios_{i}',
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
		drop_none_scenarios = FilterRows(
			name='drop_none_scenarios',
			cols=['scenarios'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
		)
		scenarios_to_text = Map(
			name='scenarios_to_text',
			cols=['source', 'scenarios'],
			output_cols=['scenarios_text', 'page_source', 'source'],
			fn=lambda output_cols=None, **row: {
				'scenarios_text': json.dumps(row['scenarios'], ensure_ascii=False),
				'page_source': row['source'],
				'source': json.dumps(row['scenarios'], ensure_ascii=False),
			},
		)
		swap_source_to_scenarios = Map(
			name='swap_source_to_scenarios',
			cols=['source', 'scenarios_text'],
			output_cols=['page_source', 'source'],
			fn=lambda output_cols=None, **row: {
				'page_source': row['source'],
				'source': row['scenarios_text'],
			},
		)

		# Stage 1: Select & Backcast (text-only)
		STAGE = 1
		stage = config.stages[STAGE]
		select_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		select_and_backcast = [
			LMGenerationTask(
				use_cache=True,
				name=f'sbp_select_backcast_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				lm_input_cols=['scenarios_text'],
				lm_input_col_prefixes=['scenarios: '],
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_persona_0 = FilterRows(
			name='drop_none_persona_0',
			cols=['persona'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
		)
		persona_to_text = Map(
			name='persona_to_text',
			cols=['source', 'persona'],
			output_cols=['persona_json', 'source'],
			fn=lambda output_cols=None, **row: {
				'persona_json': json.dumps(row['persona'], ensure_ascii=False),
				'source': json.dumps(row['persona'], ensure_ascii=False),
			},
		)

		# Stage 2: Audit (text-only)
		STAGE = 2
		stage = config.stages[STAGE]
		audit_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		audit_persona = [
			LMGenerationTask(
				use_cache=True,
				name=f'sbp_audit_{i}',
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
		drop_none_audit = FilterRows(
			name='drop_none_audit',
			cols=['abstraction_moves', 'rewrite_guidance'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
		)
		moves_to_text = Map(
			name='moves_to_text',
			cols=['source', 'abstraction_moves', 'rewrite_guidance'],
			output_cols=['abstraction_moves_text', 'rewrite_guidance'],
			fn=lambda output_cols=None, **row: {
				'abstraction_moves_text': '\n'.join(f'- {m}' for m in (row.get("abstraction_moves") or [])),
				'rewrite_guidance': row.get("rewrite_guidance", ''),
			},
		)

		# Stage 3: Rewrite (text-only)
		STAGE = 3
		stage = config.stages[STAGE]
		rewrite_router = pipe_utils.data_router(
			step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
		)
		lms = pipe_utils.make_lms(config, stage, use_cache=False)
		rewrite_persona = [
			LMGenerationTask(
				use_cache=True,
				name=f'sbp_rewrite_{i}',
				stage=stage,
				llm=lm,
				lm_config=lm.lm_config,
				input_formatter=lm.format_input,
				parallel_input_formatter=lm.parallel_format_inputs,
				lm_input_cols=['persona_json', 'abstraction_moves_text', 'rewrite_guidance'],
				lm_input_col_prefixes=['persona: ', 'abstraction_moves: ', 'rewrite_guidance: '],
				input_batch_size=BATCH_SIZE,
				resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
				output_mappings={'generation': 'persona'},
				**lm.lm_config.task_kwargs,
			)
			for i, lm in enumerate(lms)
		]
		drop_none_persona_1 = FilterRows(
			name='drop_none_persona_1',
			cols=['persona'],
			condition=utils.generation_is_structured,
			input_batch_size=BATCH_SIZE,
		)
		finalize_persona_text = Map(
			name='finalize_persona_text',
			cols=['source', 'persona'],
			output_cols=['persona'],
			fn=lambda output_cols=None, **row: {
				# Convert structured persona to a single compact string
				'persona': json.dumps(row['persona'], ensure_ascii=False)
			},
		)

		# Graph
		load_data >> scenarios_router >> generate_scenarios >> drop_none_scenarios >> scenarios_to_text >> swap_source_to_scenarios
		swap_source_to_scenarios >> select_router >> select_and_backcast >> drop_none_persona_0 >> persona_to_text >> swap_source_to_persona
		swap_source_to_persona >> audit_router >> audit_persona >> drop_none_audit >> moves_to_text
		moves_to_text >> rewrite_router >> rewrite_persona >> drop_none_persona_1 >> finalize_persona_text

	# run with load groups
	distiset, cost_tracker = pipeline.run(
		load_groups=(
			pipe_utils.steps_to_load_groups(
				[load_data, *generate_scenarios, drop_none_scenarios, scenarios_to_text],
				len(config.stages[0].available_gpus),
			) +
			[[swap_source_to_scenarios.name]] +
			pipe_utils.steps_to_load_groups(
				[*select_and_backcast, drop_none_persona_0, persona_to_text],
				len(config.stages[1].available_gpus),
			) +
			[[swap_source_to_persona.name]] +
			pipe_utils.steps_to_load_groups(
				[*audit_persona, drop_none_audit, moves_to_text],
				len(config.stages[2].available_gpus),
			) +
			pipe_utils.steps_to_load_groups(
				[*rewrite_persona, drop_none_persona_1, finalize_persona_text],
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
	ds.save_to_disk(CACHE_DIR / 'sbp_personas_ds') 