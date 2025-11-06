from distilabel.pipeline import Pipeline

from datasets import load_from_disk, concatenate_datasets, Dataset
import random

import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config
from distilabel import utils

from distilabel.steps import (
    StepResources,
    LoadDataFromDataset,
    FilterRows,
)
from distilabel.steps.tasks import (
    LMGenerationTask,
)

from distilabel.configs.lc_sft.visual_reasoning_a import (
    config,
    SP_DS_PATH,
    MP_DS_PATH,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)

fn_to_idx: dict[str, int] | None = None

STAGE = 0
BATCH_SIZE = 128

def run_pipeline(config: Config, dataset: Dataset):
    global STAGE, BATCH_SIZE
    random.seed(0)

    with Pipeline(
        name=PIPELINE_NAME,
        description='Full visual context answer from strong visual lc models',
        cache_dir=CACHE_DIR / 'visual_reasoning_a',
        disable_output_queue_timeout=True,
    ) as pipeline:
        stage = config.stages[STAGE]
        load_data = LoadDataFromDataset(name='load_data', dataset=dataset, batch_size=BATCH_SIZE)

        lms = pipe_utils.make_lms(config, stage, use_cache=True)
        answer_router = pipe_utils.data_router(
            step_distribution=[lm.lm_config.data_ratio for lm in lms]
        )
        generate_answers = [
            LMGenerationTask(
                use_cache=True,
                name=f"answer_generation_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
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

        load_data >> answer_router >> generate_answers >> drop_none_answers

    distiset, cost_tracker = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [load_data, *generate_answers, drop_none_answers],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        # invalidate_distiset=True, 
    )
    return distiset, cost_tracker

def convert_to_vision(row: dict, path_substitution: tuple[str, str] | None = None, **kwargs) -> dict:
    '''
    Convert the row to vision format
    '''
    global fn_to_idx
    image_indices = [fn_to_idx[ifn.replace(path_substitution[0], path_substitution[1])] for ifn in row['source']]

    user_content = (
        ''.join([f'<IMG_{i}>' for i in range(len(image_indices))])
        + row['question']
    )
    assistant_content = f'<think>{row['reasoning']}</think>\n{row['answer']}'
    messages = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content}
    ]

    control_token = '<reasoning>'
    r = random.random()
    if r < 0.5:
        if random.random() < 0.5:
            user_content = f'{control_token} {user_content}'
        else:
            user_content = f'{user_content} {control_token}'
        messages[0]['content'] = user_content
    elif r < 0.95:
        messages.insert(0, {'role': 'system', 'content': control_token})
    else:
        # no reasoning, teaches the model that the control token implies <think></think>
        messages[-1]['content'] = row['answer']
    
    return {
        'images': image_indices,
        'messages': messages,
        'n_images': len(image_indices),
    }


if __name__ == '__main__':
    cols_to_keep = ['source', 'question', 'split', 'question_model_name']
    sp_ds_dict = load_from_disk(SP_DS_PATH)
    mp_ds_dict = load_from_disk(MP_DS_PATH)

    sp_splits = [
        'distractors_short',
        'adj_short',
        'hn_short',
        'recursive_hn',
        'recursive_doc',
        'full_context_one_shot_hn',
        'full_context_one_shot_doc',
        'reasoning_hn',
        'reasoning_doc',
    ]
    mp_splits = [
        'true_multi_page_short_hn',
        'true_multi_page_short_doc',
        'recursive_hn',
        'recursive_doc',
        'full_context_one_shot_hn',
        'full_context_one_shot_doc',
        'reasoning_hn',
        'reasoning_doc',
    ]

    datasets: list[Dataset] = []
    for split in sp_splits:
        datasets.append(utils.add_split_label_ds(sp_ds_dict[split], f'sp_{split}'))
    for split in mp_splits:
        datasets.append(utils.add_split_label_ds(mp_ds_dict[split], f'mp_{split}'))
    dataset = concatenate_datasets(datasets)

    distiset, cost_tracker = run_pipeline(config, dataset)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    images_ds = load_from_disk(IMAGES_DS_PATH)
    fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename', config.path_substitution)

    distiset = utils.format_distiset(
        distiset, 
        convert_to_vision,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=4,
    )

    # distiset = distiset.shuffle(seed=0)
    # mt = distiset.select(range(200))
    # mt.save_to_disk(CACHE_DIR / 'for_multi_turn' / 'visual_reasoning_vds')
    # distiset = distiset.select(range(200, len(distiset)))
    distiset.save_to_disk(CACHE_DIR / 'visual_reasoning_vds')
