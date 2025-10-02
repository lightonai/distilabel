from datasets import load_from_disk, concatenate_datasets, Dataset
import random

from distilabel import utils

from distilabel.configs.lc_sft.reasoning_a import (
    config,
    SP_DS_PATH,
    MP_DS_PATH,
    CACHE_DIR,
    IMAGES_DS_PATH,
    PIPELINE_NAME,
)
from distilabel.pipelines.lc_sft.full_context_one_shot_a import run_pipeline

fn_to_idx: dict[str, int] | None = None

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
    assistant_content = f'<think>{row['reasoning']}</think>{row['answer']}'
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
        'reasoning_hn',
        'reasoning_doc',
    ]
    mp_splits = [
        'reasoning_hn',
        'reasoning_doc',
    ]

    datasets: list[Dataset] = []
    for split in sp_splits:
        datasets.append(utils.add_split_label_ds(sp_ds_dict[split], f'sp_{split}'))
    for split in mp_splits:
        datasets.append(utils.add_split_label_ds(mp_ds_dict[split], f'mp_{split}'))
    dataset = concatenate_datasets(datasets)

    distiset, cost_tracker = run_pipeline(config, dataset, PIPELINE_NAME)
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    images_ds = load_from_disk(IMAGES_DS_PATH)
    fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename', config.path_substitution)

    distiset = utils.format_distiset(
        distiset, 
        convert_to_vision,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=16,
    )

    distiset.save_to_disk(CACHE_DIR / 'reasoning_vds')
