from datasets import Dataset, load_from_disk

from distilabel import utils
from distilabel.configs.fill_in_middle import PDF_ROOT, IMAGES_DS_PATH, CACHE_DIR

fn_to_page_count: dict[str, int] = None
fn_to_idx: dict[str, int] = None

def convert_to_vision(row: dict, **kwargs) -> dict:
    '''
    Convert the row to vision format
    '''
    global fn_to_page_count, fn_to_idx
    ifn = row['source'][0]
    pdf_name = utils.pdf_name(ifn)
    n_pages = fn_to_page_count[pdf_name]
    drop_page = utils.pdf_page(ifn)
    image_indices = [fn_to_idx[utils.path_as_page(ifn, i)] for i in range(n_pages) if i != drop_page]

    user_content = (
        "".join([f"Page {i}:<IMG_{i - (1 if i > drop_page else 0)}>" if i != drop_page else f"Page: {drop_page}" for i in range(n_pages)])
        + f"Fill in the missing text from page {drop_page}."
    )
    assistant_content = row['md']
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
    global fn_to_page_count, fn_to_idx
    fn_to_page_count = utils.count_all_pages(
        pdf_root=PDF_ROOT,
        cache_dir=CACHE_DIR,
        n_jobs=16,
    )

    images_ds = load_from_disk(IMAGES_DS_PATH)
    fn_to_idx = utils.generate_field_to_idx(images_ds, 'image_filename')
    distiset = distiset.select_columns(['transcribe_model_name', 'md', 'source']).to_list()

    tqdm_desc = "Processing Tasks"
    cpe = utils.continuous_parallel_execution(
        function=convert_to_vision,
        tasks=[{'row': row, 'idx': idx} for idx, row in enumerate(distiset)],
        task_count=len(distiset),
        process_type="process",
        num_workers=16,
        max_active_tasks=1024,
        tqdm_desc=tqdm_desc,
    )
    vision_ds = [None] * len(distiset)
    for task, result in cpe:
        vision_ds[task['idx']] = result

    distiset = Dataset.from_list(vision_ds).select_columns(['images', 'messages', 'n_images'])
    return distiset

