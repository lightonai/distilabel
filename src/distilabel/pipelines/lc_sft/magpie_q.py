from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset, DatasetDict
from logging import getLogger
from itertools import accumulate
import random
import re
from pathlib import Path
from typing import Any
from functools import partial

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    LoadPydanticAsColumns,
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)
from distilabel.steps import (
    NoOp,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage
from distilabel.models.llms import OpenAILM, lm_cache, structured_output, multiple_generations
from distilabel.typing import FormattedInput, GenerateOutput

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from distilabel.configs.lc_sft.magpie_q import (
    config,
    CACHE_DIR,
    PIPELINE_NAME,
    IMAGES_DS_PATH,
)


class MagpieLM(OpenAILM):
    '''
    MagpieLM is a wrapper around OpenAILM that simply removes the requirement for a non-empty input
    '''
    @lm_cache
    @multiple_generations
    @structured_output
    async def agenerate(
        self, 
        input: FormattedInput,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> 'GenerateOutput':
        no_response = GenerateOutput(
            generations=[None],
            reasoning_generations=[None],
            statistics={'input_tokens': [0], 'output_tokens': [0]},
            cache_hit=[False],
        )
        
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
        async def _generate():
            # Ensure we always send our hardcoded Magpie-style chat template so that,
            # when the input is images only, the model generates as the `user`.
            completion = await self._aclient.chat.completions.create(
                model=self.model_name,
                messages=input,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body or {} | self.api_call_extra_body,
                timeout=1800,
            )
            return completion
        
        try:
            completion = await _generate()
        except openai.APIConnectionError as e:
            self._logger.warning(f"Failed to get client response, exception: {e}")
            raise e
        except openai.BadRequestError as e:
            self._logger.warning(f"Failed to get client response, bad request error: {e}\n{input=}")
            raise e
        except Exception as e:
            completion = None
            self._logger.warning(f"Failed to get client response, exception: {e}")
        if completion is None:
            return no_response
        return self._generations_from_openai_completion(completion)

def make_lms(config: Config, stage: Stage, use_cache: bool = False, invalidate_cache: bool | list[bool] = False) -> list[MagpieLM]:
    '''initialize lms for a stage'''
    return [
        MagpieLM(
            stage=stage, 
            lm_config=lm_config, 
            model=lm_config.path, 
            generation_kwargs={
                'temperature': lm_config.temperature, 
                'max_new_tokens': lm_config.max_new_tokens,
            },
            api_call_extra_body=lm_config.api_call_extra_body,
            use_running_vllm=config.use_running_vllm,
            use_cache=use_cache,
            invalidate_cache=invalidate_cache[i] if isinstance(invalidate_cache, list) else invalidate_cache,
            replicas_per_vllm_server=lm_config.replicas_per_vllm_server,
        ) 
        for i, lm_config in enumerate(stage.lm_configs)
    ]

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 128

def run_pipeline(config: Config):
    global STAGE
    global BATCH_SIZE
    
    stages = config.stages
    dataset = load_from_disk(CACHE_DIR / 'magpie_input_ds')

    with Pipeline(
        name=PIPELINE_NAME,
        description="Sample answers using magpie/no-prompt method",
        cache_dir=CACHE_DIR / 'magpie_q',
        disable_output_queue_timeout=True,
    ) as pipeline:
        ################## STAGE 0: GENERATE QUESTIONS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)  # cols: ['source', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = make_lms(config, stage, use_cache=True, invalidate_cache=False)
        generate_questions = [
            LMGenerationTask(
                use_cache=True,
                # invalidate_cache=True,
                name=f"question_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                system_col='default_system',
                input_mappings={'source': 'question_source'},  # use the subset of pages, same as in single_page_q/true_multi_page_q
                output_mappings={'generation': 'question', 'model_name': 'question_model_name'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['questions', 'key_ideas', 'key_details', 'question_system', 'question_model_name', ...]
        drop_none_questions = FilterRows(  # drop rows where the question is None (structured output failed)
            name="drop_none_questions",
            cols=['question'],
            condition=utils.logical_and_filters(utils.not_empty_string, utils.generation_is_structured),
            input_batch_size=BATCH_SIZE,
        )  # cols: ['question', ...] -> ['question', ...]

        ################## STAGE 1: GENERATE ANSWERS ##################
        STAGE = 1
        stage = config.stages[STAGE]
        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in lms]
        )
        generate_answers = [  # note this is running on the full source (all images)
            LMGenerationTask(
                use_cache=True,
                name=f"answer_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                system_col='default_system',  # use model default system prompt
                lm_input_cols=['question'],
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]
        drop_none_answers = FilterRows(
            name='drop_none_answers',
            cols=['answer'],
            condition=utils.generation_is_structured,
            input_batch_size=BATCH_SIZE,
        )

        ## Pipeline
        (
            load_data >> data_router >> generate_questions >> drop_none_questions 
            >> answer_router >> generate_answers >> drop_none_answers
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_questions, drop_none_questions],
                len(stage.available_gpus),
            ) + pipe_utils.steps_to_load_groups(
                [*generate_answers, drop_none_answers],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    distiset, cost_tracker = run_pipeline(config)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])

    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'question_source'],#, 'split'], 
        n_workers=16,
    )

    distiset.save_to_disk(CACHE_DIR / 'magpie_vds')

