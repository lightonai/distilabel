from datasets import load_from_disk, Dataset, DatasetDict
from distilabel import utils
import random

from pathlib import Path
from distilabel import utils

def vis(ds, idx):
    dr = Path('visualizing_generated_data')
    dr.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(ds[idx]['source']):
        utils.get_image(None, page).save(dr / f'page_{i}.png')
    print('Question:\n', ds[idx]['question'], '\n\nAnswer:\n', ds[idx]['answer'])

random.seed(0)

splits = ['distractors_short', 'hard_negs_short', 'adjacent_pages_short', 'true_mp_hns', 'true_mp_aps']
dataset = load_from_disk('out/mp_synthetic_data')

hns = utils.randomize_source_order(list(dataset['hard_negs_short']))
aps = utils.sort_adjacent_pages(list(dataset['adjacent_pages_short']))
dataset['hard_negs_short'] = Dataset.from_list(hns)
dataset['adjacent_pages_short'] = Dataset.from_list(aps)

ds = DatasetDict()
for split_name in splits:
    split = list(dataset[split_name])
    vis_split = []
    for row in split:
        vis_split.append({
            'images': row['source'],
            'messages': [
                {
                    'role': 'user',
                    'content': ''.join(
                        [
                            f'<IMG_{i}>' for i in range(len(row['source']))
                        ] 
                        + [row['question']]
                    )
                },
                {
                    'role': 'assistant',
                    'content': row['answer']
                }
            ]
        })
    ds[split_name] = Dataset.from_list(vis_split)

ds.save_to_disk('out/mp_synthetic_data_vis_format')




