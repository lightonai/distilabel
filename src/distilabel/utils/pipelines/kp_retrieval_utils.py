import re
import random
from datasets import Dataset, load_from_disk

from distilabel import utils
from distilabel.configs.kp_retrieval import PDF_ROOT, DS_PATH, CACHE_DIR

def update_hn_idxs(dataset: Dataset, distiset: Dataset) -> Dataset:
    '''
    Update the hard negative indices to point to the correct examples in the distiset
    '''
    dataset_filename_to_idx = utils.generate_field_to_idx(dataset, 'image_filename')
    distiset_filename_to_idx = utils.generate_field_to_idx(distiset, 'image_filename')
    old_idx_to_new_idx = {
        dataset_filename_to_idx[fn]: distiset_filename_to_idx[fn] 
        for fn in set(distiset['image_filename']) if fn in distiset_filename_to_idx
    }
    distiset = distiset.map(
        lambda x: {
            'hard_negs_idx_img_img': [old_idx_to_new_idx[hni] for hni in x['hard_negs_idx_img_img'] if hni in old_idx_to_new_idx],
            'hard_negs_idx_txt_img': [old_idx_to_new_idx[hni] for hni in x['hard_negs_idx_txt_img'] if hni in old_idx_to_new_idx],
        }, num_proc=2,
    )
    return distiset

def build_images_ds(fn_to_page_count: dict[str, int]) -> tuple[Dataset, dict[str, int]]:
    '''
    Build the images dataset from the fn_to_page_count
    '''
    images_ds = {'image': [], 'image_filename': []}
    fn_to_idx = {}
    for fn, page_count in fn_to_page_count.items():
        for i in range(page_count):
            ifn = utils.page_path(fn, i)
            images_ds['image'].append(None)
            images_ds['image_filename'].append(ifn)
            fn_to_idx[ifn] = len(images_ds['image']) - 1
    return Dataset.from_dict(images_ds), fn_to_idx

fn_to_page_count: dict[str, int] = None
fn_to_idx: dict[str, int] = None
dataset_idx_to_fn: dict[int, str] = None

def convert_to_vision(row: dict, use_hn: bool = False, **kwargs) -> dict:
    '''
    Convert the row to vision format

    For key retrieval, the task is to retrieve text before or after a given key.
    For pos retrieval, the task is to retrieve text described by a page number and page relative location (like 'the first sentence in the second from last paragraph from page 2')
    '''
    global fn_to_page_count, fn_to_idx, dataset_idx_to_fn
    random.seed(kwargs['idx'])
    ifn = row['source'][0]
    pdf_name = utils.pdf_name(ifn)
    if use_hn:
        negs = row['hard_negs_idx_img_img'] + row['hard_negs_idx_txt_img']
        negs = random.sample(negs, k=random.randint(5, min(len(negs), 63)))
        image_indices = [
            fn_to_idx[dataset_idx_to_fn[neg]] for neg in negs
        ]
        key_page = random.randint(0, len(image_indices))
        image_indices.insert(key_page, fn_to_idx[ifn])
        n_pages = len(image_indices)
    else:
        n_pages = fn_to_page_count[pdf_name]
        key_page = utils.pdf_page(ifn)
        image_indices = [fn_to_idx[utils.path_as_page(ifn, i)] for i in range(n_pages)]

    if row['key_extraction_system'] is not None:
        system_prompt = row['key_extraction_system']
        match = re.search(r"Select a ([\s\S]*?) from the page at a height", system_prompt)
        sentence_or_paragraph = match.group(1)

        selection = row['key_selection']
        user_content = (
            "".join([f"<IMG_{i}>" for i in range(n_pages)])
            + f"Extract the {sentence_or_paragraph} the key is part of from the document.\n<key>{selection}</key>"
        )
        assistant_content = row['key_extraction']
    else:
        system_prompt = row['pos_extraction_system']
        # from "Extract the {extraction_goal} from the page."
        match = re.search(r"Extract the ([\s\S]*?) from the page\.", system_prompt)
        extraction_goal = match.group(1)

        # going to use 1-indexed pages to teach this capability
        user_content = (
            "".join([f"Page {i + 1}:<IMG_{i}>" for i in range(n_pages)])
            + f"Extract the {extraction_goal} from page {key_page + 1}."
        )
        assistant_content = row['pos_extraction']
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
    global fn_to_page_count, fn_to_idx, dataset_idx_to_fn
    fn_to_page_count = utils.count_all_pages(
        pdf_root=PDF_ROOT,
        cache_dir=CACHE_DIR,
        n_jobs=16,
    )
    
    dataset = load_from_disk(DS_PATH)
    dataset_idx_to_fn = utils.generate_idx_to_filename(dataset)

    # build images ds out of fn_to_page_count, as doing this, build map from fn to idx in images ds
    images_ds, fn_to_idx = build_images_ds(fn_to_page_count)
    distiset = distiset.select_columns(
        [
            'key_extraction_system', 'pos_extraction_system',
            'key_selection', 'key_extraction', 'pos_extraction', 'md', 'source',
            'hard_negs_idx_img_img', 'hard_negs_idx_txt_img',
        ]
    ).to_list()

    tqdm_desc = "Processing Tasks"
    cpe = utils.continuous_parallel_execution(
        function=convert_to_vision,
        tasks=[
            {'row': row, 'idx': idx} for idx, row in enumerate(distiset)
        ] + [
            {'row': row, 'idx': idx + len(distiset), 'use_hn': True} for idx, row in enumerate(distiset)
        ],
        task_count=len(distiset) * 2,
        process_type="process",
        num_workers=16,
        max_active_tasks=1024,
        tqdm_desc=tqdm_desc,
    )
    vision_ds = [None] * (len(distiset) * 2)
    for task, result in cpe:
        vision_ds[task['idx']] = result

    distiset = Dataset.from_list(vision_ds).select_columns(['images', 'messages', 'n_images'])
    return distiset, images_ds



