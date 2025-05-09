from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    FilterRows,
    ListToRows,
    Split,
    Rejoin,
    NoOp,
    JoinParallelBranches,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage

from distilabel.configs.true_multi_page import config

def structured_and_requires_multiple_pages(row: dict, cols: list[str]) -> bool:
    '''
    Check the 'question_fully_answered' column, all must be False for the question to require multiple pages

    Also check that the cols are structured (not None)
    '''
    structured = True
    for col in cols:
        if isinstance(row[col], list):
            if any([generation is None for generation in row[col]]):
                structured = False
        else:
            if row[col] is None:
                structured = False
    return not any(row['question_fully_answered']) and structured

STAGE = 0
'''tracks the current stage of the pipeline'''

def run_pipeline(config: Config):
    global STAGE
    
    stages = config.stages
    dataset = load_from_disk('out/mp_synthetic_data')['multi_page_questions']
    dataset = list(dataset)[12:13]
    
    with Pipeline(
        name='true_multi_page_qa',
        description=(
            'Try to generate questions that require multiple pages to answer using a generate -> filter strategy. '
        ),
        cache_dir='out/distilabel_cache',
    ) as pipeline:
        ################## STAGE 0: INITIAL QUESTIONS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDicts(name="load_data", data=dataset, batch_size=128)  # cols: ['source', ...]
        
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        
        lms = pipe_utils.make_lms(config, stage)
        generate_questions = [
            LMGenerationTask(
                name=f"question_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm._format_input,
                input_batch_size=64,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'question_system', 'model_name': 'question_model_name', 'analysis': 'question_analysis'},
                **lm.lm_config.task_kwargs,
            ) 
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['questions', 'question_system', 'question_model_name', ...]
        
        questions_to_rows = ListToRows(
            name="questions_to_rows",
            input_col='questions',
            input_batch_size=64,
            output_mappings={'questions': 'question'},
            resources=StepResources(replicas=1),
        )  # cols: ['questions', ...] -> ['question', ...]
        
        drop_none_questions = FilterRows(
            name="drop_none_questions",
            cols=['question'],
            condition=utils.generation_is_structured,
            input_batch_size=64,
        )  # cols: ['question', ...] -> ['question', ...]
        
        ################## STAGE 1: INDIVIDUAL PAGE ANSWERS AND BREAK DOWN QUESTION ##################
        STAGE += 1
        stage = stages[STAGE]

        lms = pipe_utils.make_lms(config, stage)
        sp_lms = [lm for lm in lms if lm.lm_config.task_name == 'single_page_answer']
        q_req_lms = [lm for lm in lms if lm.lm_config.task_name == 'question_requirements']

        split_pages = {
            branch: Split(
                name=f'split_pages_{branch}',
                input_col='source',
                keep_as_list=True,  # keep source as a list of strings (format distinguishing pages from text directly for input)
                input_batch_size=64,
                resources=StepResources(replicas=1),
            ) for branch in ['sp_branch', 'q_req_branch']
        }  # cols: ['source', ...] -> ['source', ...]

        q_req_branch = NoOp(name='q_req_branch', input_batch_size=64)
        q_req_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in q_req_lms]
        )
        generate_question_req = [
            LMGenerationTask(
                name=f'question_reqs_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm._format_input,
                input_batch_size=64,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                extra_cols=['answer'],
                input_mappings={'source': 'question'},  # don't want the pages as context, just question
                output_mappings={
                    'system': 'question_requirements_system', 
                    'model_name': 'question_requirements_model_name', 
                    'answer': 'drop',  # this (defaulted to '') can override the answer from the single page answers when joined, so dropping it
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(q_req_lms)
        ]  # cols: ['question', ...] -> ['question_requirements', 'question_requirements_system', 'question_requirements_model_name', ...]

        filter_question_requirements = FilterRows(
            name='filter_question_requirements',
            cols=['question_requirements'],
            condition=utils.generation_is_structured,
            input_batch_size=64,
        )

        sp_answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in sp_lms]
        )
        generate_single_page_answers = [
            LMGenerationTask(
                name=f'single_page_answers_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm._format_input,
                lm_input_cols=['question'],
                input_batch_size=64,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'sp_answer_system', 'model_name': 'sp_answer_model_name', 'generation': 'sp_answer'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(sp_lms)
        ]  # cols: ['source', 'question', ...] -> ['sp_answer', 'sp_answer_system', 'sp_answer_model_name', ...]
        # needed because distilabel won't let a convergence step have predecessors that are not from the set of routes
        collect_sp_answers = NoOp(name='collect_sp_answers', input_batch_size=64)

        collect_sp_answers_and_q_req = JoinParallelBranches(
            name='collect_sp_answers_and_q_req', 
            join_on_col='source',
            input_batch_size=64,
            output_mappings={'source': 'page_source'},  # renaming this so that I can replace the source with question on the judge step
        )
        
        ################## STAGE 2: JUDGE ANSWERS AND FILTER ##################
        STAGE += 1
        stage = stages[STAGE]
        
        judge_answers_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )

        lms = pipe_utils.make_lms(config, stage)
        judge_answers = [
            LMGenerationTask(
                name=f'answer_judge_{i}',
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm._format_input,
                lm_input_cols=['question_requirements', 'sp_answer'],
                lm_input_col_prefixes=['question requirements: ', 'answer: '],
                input_batch_size=64,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                extra_cols=['page_source'],
                input_mappings={'source': 'question'},  # don't want the pages as context
                output_mappings={
                    'system': 'judge_system', 
                    'model_name': 'judge_model_name', 
                    # 'source': 'question',
                    'page_source': 'source',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['question', 'question_requirements', 'sp_answer', ...] -> ['question_requirements_met', 'question_fully_answered', 'judge_system', 'judge_model_name', ...]
        
        rejoin_pages = Rejoin(name='rejoin_pages', input_col='source', keep_as_list_cols={'question_fully_answered', 'sp_answer'})

        drop_poor_questions = FilterRows(  # checks the results of individual page answers to see if any of them passed
            name='drop_poor_questions',
            cols=['question_fully_answered'],
            condition=structured_and_requires_multiple_pages,
            input_batch_size=64,
        )  # cols: ['question_fully_answered', ...] -> ['question_fully_answered', ...]
        
        ################## STAGE 3: QUALITY ANSWERS TO QUALITY QUESTIONS ##################
        # STAGE += 1
        # stage = stages[STAGE]
        
        # data_router = pipe_utils.data_router(
        #     step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        # )
        
        # lms = pipe_utils.make_lms(config, stage)
        # generate_answers = [
        #     LMGenerationTask(
        #         name=f"answer_generation_{i}",
        #         stage=stage,
        #         llm=lm,
        #         lm_config=lm.lm_config,
        #         input_formatter=lm._format_input,
        #         lm_input_cols=['question'],
        #         input_batch_size=64,
        #         resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
        #         output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
        #         **lm.lm_config.task_kwargs,
        #     )
        #     for i, lm in enumerate(lms)
        # ]  # cols: ['source', 'question', ...] -> ['answer', 'answer_system', 'answer_model_name', ...]
        
        ## Pipeline
        load_data >> data_router >> generate_questions >> questions_to_rows >> drop_none_questions
        
        # branch between single page answers and question requirements
        drop_none_questions >> [q_req_branch, split_pages['sp_branch']]

        # q_req_branch
        q_req_branch >> q_req_router >> generate_question_req >> filter_question_requirements >> split_pages['q_req_branch']
        # sp_branch
        split_pages['sp_branch'] >> sp_answer_router >> generate_single_page_answers >> collect_sp_answers

        # join branches
        [split_pages['q_req_branch'], collect_sp_answers] >> collect_sp_answers_and_q_req
        
        (
            collect_sp_answers_and_q_req >> judge_answers_router >> judge_answers >> rejoin_pages >> drop_poor_questions
            # reference_data_router >> generate_reference_answers
        )
    
    # pipeline.draw(show_edge_labels=True)
    distiset = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_questions, questions_to_rows, drop_none_questions],
                len(stage.available_gpus),
            ) + 
            pipe_utils.steps_to_load_groups(
                [
                    q_req_branch, *generate_question_req, filter_question_requirements, split_pages['q_req_branch'],
                    split_pages['sp_branch'], *generate_single_page_answers, collect_sp_answers,
                    collect_sp_answers_and_q_req
                ],
                len(stage.available_gpus),
            ) +
            pipe_utils.steps_to_load_groups(
                [*judge_answers],
                len(stage.available_gpus),
            ) +
            [[rejoin_pages.name]] +  # global step goes on its own
            pipe_utils.steps_to_load_groups(
                [drop_poor_questions],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
    )
    return distiset

if __name__ == '__main__':
    distiset: Dataset = run_pipeline(config)
    if len(distiset) > 0:
        distiset = distiset['default']['train'].remove_columns(['distilabel_metadata', 'drop'])
        pass
        # Saving will be handled separately
