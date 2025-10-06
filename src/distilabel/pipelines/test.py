from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    LoadPydanticAsColumns,
    FilterRows,
    ListToRows,
    NoOp,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage
from distilabel.models.llms import OpenAILM

from distilabel.configs.test import config

STAGE = 0
'''tracks the current stage of the pipeline'''

def lm_task_router(lm: OpenAILM, **kwargs) -> Task:
    # Here as a demonstration of general versatility in configuring pipelines, you may want different LMs running different tasks
    match lm.lm_config.task_name:
        case ('question_generation' | 'answer_generation'):
            return LMGenerationTask(llm=lm, **kwargs)
        case _:
            raise ValueError(f"Unknown task name: {lm.lm_config.task_name}")

def run_pipeline(config: Config):
    global STAGE
    
    stages = config.stages
    dataset = load_from_disk('out/mp_synthetic_data')['single_pages']
    dataset = list(dataset.remove_columns(['hard_negs_img_img', 'hard_negs_txt_img']))[:6]
    with Pipeline(
        name="test",
        description="For debugging",
        cache_dir='out/distilabel_cache',
    ) as pipeline:
        ################## STAGE 0: GENERATE QUESTIONS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDicts(name="load_data", data=dataset, batch_size=1)  # cols: ['source', ...]

        # the data router handles routing the data to different lms according to the data_ratio
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        # initialize lms for this stage, however, they won't launch servers and stuff until they are loaded
        lms = pipe_utils.make_lms(config, stage)
        generate_questions = [
            lm_task_router(
                name=f"question_generation_{i}",
                stage=stage,
                lm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                input_batch_size=1,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.n_gpus, oversubscribe=lm.lm_config.replicas_per_vllm_server),
                output_mappings={'system': 'question_system', 'model_name': 'question_model_name'},
                **lm.lm_config.task_kwargs,
            ) 
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['questions', 'key_ideas', 'key_details', 'question_system', 'question_model_name', ...]
        questions_to_rows = ListToRows(  # expand the generated list of questions into separate rows
            name="questions_to_rows",
            input_col='questions',
            input_batch_size=1,
            output_mappings={'questions': 'question'},
            resources=StepResources(replicas=1),
        )  # cols: ['questions', ...] -> ['question', ...]
        drop_none_questions = FilterRows(  # drop rows where the question is None (structured output failed)
            name="drop_none_questions",
            cols=['question'],
            condition=utils.generation_is_structured,
            input_batch_size=1,
        )  # cols: ['question', ...] -> ['question', ...]

        ## Pipeline
        (
            load_data >> data_router >> generate_questions >> questions_to_rows >> drop_none_questions
        )
    
    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(  # handles breaking up steps so each load_group has enough gpus
                # data_router is not included because it's not quite a step, but it does actually still run
                [load_data, *generate_questions, questions_to_rows, drop_none_questions],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
    )
    return distiset, cost_tracker

if __name__ == "__main__":
    distiset, cost_tracker = run_pipeline(config)
    distiset = distiset['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])  # don't need this for this pipeline
