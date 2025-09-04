import re
import random
from collections import defaultdict
from datasets import Dataset, load_from_disk

from distilabel import utils
from distilabel.configs.sum_count import IMAGES_DS_PATH, SHUFFLE_FACTOR

images_fn_to_idx: dict[str, int] = None

def get_row_id_to_idxs(distiset: Dataset) -> dict[str, list[int]]:
    row_id_to_idxs = defaultdict(list)
    for distiset_idx, row in enumerate(distiset):
        row_id_to_idxs[row['row_id']].append(distiset_idx)
    return dict(row_id_to_idxs)

def convert_to_vision(rows: list[dict], **kwargs) -> dict:
    '''
    Convert the group of rows corresponding to a single pdf to vision format
    '''
    global images_fn_to_idx

    image_indices = [images_fn_to_idx[row['source'][0]] for row in rows]
    n_pages = len(image_indices)

    # we made the system prompts the same for each pdf, so we can just use the first one
    system_prompt = rows[0]['count_system']
    match = re.search(r"Count the number of ([\s\S]*?) in the page\.", system_prompt)
    object_type = match.group(1)

    user_content = (
        "".join([f"<IMG_{i}>" for i in range(n_pages)])
        + f"Count the number of {object_type} in the document."
    )

    assistant_content = str(sum([row['count'] for row in rows]))

    messages = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content}
    ]
    return {
        'images': image_indices,
        'messages': messages,
        'n_images': len(image_indices),
    }

def format_distiset(distiset: Dataset) -> Dataset:
    '''
    Format the distiset to vision format/build actual examples from extractions
    '''
    global images_fn_to_idx

    images_ds = load_from_disk(IMAGES_DS_PATH)
    images_fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename')
    row_id_to_idxs = get_row_id_to_idxs(distiset)

    distiset = distiset.select_columns(['count_system', 'count_model_name', 'count', 'source', 'row_id']).to_list()

    tqdm_desc = "Processing Tasks"
    cpe = utils.continuous_parallel_execution(
        function=convert_to_vision,
        tasks=[
            {'rows': shfl, 'idx': idx * SHUFFLE_FACTOR + shfl_idx} 
            for idx, row_id_idxs in enumerate(row_id_to_idxs.values())
            for shfl_idx, shfl in enumerate([[distiset[i] for i in row_id_idxs]] + [
                [
                    distiset[i] 
                    for i in random.sample(
                        row_id_idxs, 
                        k=random.randint(
                            min(len(row_id_idxs), 5), 
                            max(int(0.8 * len(row_id_idxs)), min(len(row_id_idxs), 5)) # avoid empty range
                        )
                    )
                ] 
                for _ in range(SHUFFLE_FACTOR - 1)
            ])
        ],
        task_count=len(row_id_to_idxs) * SHUFFLE_FACTOR,
        process_type="process",
        num_workers=16,
        max_active_tasks=1024,
        tqdm_desc=tqdm_desc,
    )
    vision_ds = [None] * len(row_id_to_idxs) * SHUFFLE_FACTOR
    for task, result in cpe:
        vision_ds[task['idx']] = result

    distiset = Dataset.from_list(vision_ds).select_columns(['images', 'messages', 'n_images'])
    return distiset



