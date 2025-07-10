import random

from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    LoadDataFromDataset,
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
        if (
            (isinstance(row[col], list) and any([generation is None for generation in row[col]]))
            or row[col] is None
        ):
            structured = False
    return not any(row['question_fully_answered']) and structured

STAGE = 0
'''tracks the current stage of the pipeline'''

def run_pipeline(config: Config):
    global STAGE
    random.seed(0)
    
    stages = config.stages
    dataset = load_from_disk('out/mp_synthetic_data')['multi_page_questions']
    dataset = list(dataset)

    # randomize the order of pages for hard negatives and sort for adjacent pages
    hns = [row for row in dataset if row['split'] == 'hard_negs_short']
    aps = [row for row in dataset if row['split'] == 'adjacent_pages_short']
    hns = utils.randomize_source_order(hns)
    aps = utils.sort_adjacent_pages(aps)
    dataset = Dataset.from_list(hns + aps)

    with Pipeline(
        name='true_multi_page_qa',
        description=(
            'Try to generate questions that require multiple pages to answer using a generate -> filter strategy. '
        ),
        cache_dir='out/mp_trial',
        debug=True,
    ) as pipeline:
        ################## STAGE 0: INITIAL QUESTIONS ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=64)  # cols: ['source', ...]
        
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

        q_req_branch = NoOp(
            name='q_req_branch', 
            cols=['source'],
            output_mappings={'source': 'page_source'}, 
            input_batch_size=64
        )
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
                extra_cols=['answer', 'page_source'],
                input_mappings={'source': 'question'},  # don't want the pages as context, just question
                output_mappings={
                    'system': 'question_requirements_system', 
                    'model_name': 'question_requirements_model_name', 
                    'answer': 'drop',  # this can override the answer from the single page answers when joined, so dropping it
                    'page_source': 'source',
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
        # distilabel won't let a convergence step have predecessors that are not from the set of routes
        # could use a no op here, but in case the lm returns None, we want to drop that row (even though it isn't structured, there are other ways to get a None)
        drop_none_sp_answers = FilterRows(
            name='drop_none_sp_answers',
            cols=['sp_answer'],
            condition=utils.generation_is_structured,
            input_batch_size=64,
        )

        collect_sp_answers_and_q_req = JoinParallelBranches(
            name='collect_sp_answers_and_q_req', 
            join_on_cols=['source', 'question'],  # the pair (source, question) is unique for each branch
            input_batch_size=64,
            output_mappings={'source': 'page_source'},  # renaming this so that I can replace the source with question on the judge step
        )

        # global step must take and output all current rows, so that will all get routed together
        # we use the no op to break that into batches
        break_up_collect_sp_answers_and_q_req = NoOp(name='break_up_collect_sp_answers_and_q_req', input_batch_size=64)
        
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
                    'page_source': 'source',
                },
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['question', 'question_requirements', 'sp_answer', ...] -> ['question_requirements_met', 'question_fully_answered', 'judge_system', 'judge_model_name', ...]
        
        rejoin_pages = Rejoin(
            name='rejoin_pages', 
            input_col='source',
            drop_incomplete_rows=True,
            duplicates_cols={
                # see the docstring for Rejoin for the necessity of this set
                'question', 
                'question_analysis',
                'question_system',
                'question_model_name',
                'question_requirements', 
                'question_requirements_system',
                'question_requirements_model_name',
                'split',
            }, 
            input_batch_size=64,
        )

        drop_poor_questions = FilterRows(  # checks the results of individual page answers to see if any of them passed
            name='drop_poor_questions',
            cols=['question_fully_answered'],
            condition=structured_and_requires_multiple_pages,
            input_batch_size=64,
        )  # cols: ['question_fully_answered', ...] -> ['question_fully_answered', ...]
        
        ################# STAGE 3: QUALITY ANSWERS TO QUALITY QUESTIONS ##################
        STAGE += 1
        stage = stages[STAGE]
        
        final_answer_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        
        lms = pipe_utils.make_lms(config, stage)
        generate_answers = [
            LMGenerationTask(
                name=f"answer_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm._format_input,
                lm_input_cols=['question'],
                input_batch_size=64,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'answer_system', 'model_name': 'answer_model_name', 'generation': 'answer'},
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]  # cols: ['source', 'question', ...] -> ['answer', 'answer_system', 'answer_model_name', ...]

        drop_none_answers = FilterRows(
            name='drop_none_answers',
            cols=['answer'],
            condition=utils.generation_is_structured,
            input_batch_size=64,
        )
        
        ## Pipeline
        load_data >> data_router >> generate_questions >> questions_to_rows >> drop_none_questions
        
        # branch between single page answers and question requirements
        drop_none_questions >> [q_req_branch, split_pages['sp_branch']]

        # q_req_branch (split pages at the end in this branch because the lm response is per question)
        q_req_branch >> q_req_router >> generate_question_req >> filter_question_requirements >> split_pages['q_req_branch']
        # sp_branch
        split_pages['sp_branch'] >> sp_answer_router >> generate_single_page_answers >> drop_none_sp_answers

        # join branches
        [split_pages['q_req_branch'], drop_none_sp_answers] >> collect_sp_answers_and_q_req
        
        (
            collect_sp_answers_and_q_req >> break_up_collect_sp_answers_and_q_req >> 
            judge_answers_router >> judge_answers >> rejoin_pages >> drop_poor_questions
            >> final_answer_router >> generate_answers >> drop_none_answers
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
                    split_pages['sp_branch'], *generate_single_page_answers, drop_none_sp_answers,
                ],
                len(stage.available_gpus),
            ) +
            [[collect_sp_answers_and_q_req.name]] +  # global step goes on its own
            pipe_utils.steps_to_load_groups(
                [break_up_collect_sp_answers_and_q_req, *judge_answers],
                len(stage.available_gpus),
            ) +
            [[rejoin_pages.name]] +  # global step goes on its own
            pipe_utils.steps_to_load_groups(
                [drop_poor_questions, *generate_answers, drop_none_answers],
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
        dataset = load_from_disk('out/mp_synthetic_data')['multi_page_questions']
        dataset = list(dataset)
        # distiset = utils.replace_source_col(distiset, dataset)
        
        hns = distiset.filter(lambda x: x['split'] == 'hard_negs_short')
        aps = distiset.filter(lambda x: x['split'] == 'adjacent_pages_short')
        utils.add_split_to_dataset_dict('out/mp_synthetic_data', 'true_mp_hns', hns)
        utils.add_split_to_dataset_dict('out/mp_synthetic_data', 'true_mp_aps', aps)
