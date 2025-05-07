import os
import json
from pdf2image import pdfinfo_from_path
from pathlib import Path as pth
import yaml
from contextlib import contextmanager
from io import StringIO
import sys
import re
from typing import Callable
from datasets import Dataset
from pathlib import Path

from .image import get_image, downsample_image, b64_encode_image

@contextmanager
def suppress_output(debug: bool):
    """Stfu utility."""
    if debug:
        yield
    else:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        class NullIO(StringIO):
            def write(self, txt: str) -> None:
                pass

        sys.stdout = NullIO()
        sys.stderr = NullIO()
        yield
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def normalize_distribution(dist: list[float]) -> list[float]:
    '''Normalize a distribution so that the sum of the distribution is 1'''
    total = sum(dist)
    return [d / total for d in dist]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def pdf_name(page):
    '''Return the pdf name from a page filename'''
    return page[:page.rfind('_page_')] + '.pdf'

def pdf_page(page):
    '''Return the page number from a page filename'''
    return int(page[page.rfind('_page_') + 6:page.rfind('.pdf')])

def path_as_page(path, page):
    '''Return the page filename from a path and page number'''
    return path[:path.rfind('_page_')] + f'_page_{page}.pdf'

def n_pages(path):
    '''Return the number of pages in a pdf'''
    return pdfinfo_from_path(path)['Pages']

def generate_idx_to_filename(ds):
    idx_to_filename = {
        idx: filename
        for idx, filename in enumerate(ds['image_filename'])
    }
    return idx_to_filename

def generate_field_to_idx(ds, field):
    field_to_idx = {
        field: idx
        for idx, field in enumerate(ds[field])
    }
    return field_to_idx

def clear_dir(directory):
    """Remove a directory and all its contents using subprocess and 'rm -rf'."""
    import subprocess
    subprocess.run(['rm', '-rf', str(directory)])

def add_split_to_dataset_dict(dataset_path: str | Path, split_name: str, data: Dataset):
    '''Add a split to a dataset dict by saving the dataset and updating the dataset_dict.json'''
    if isinstance(dataset_path, str): dataset_path = Path(dataset_path)

    split_path = str(dataset_path / split_name)
    if (dataset_path / split_name).exists():
        return
    json_path = str(dataset_path / 'dataset_dict.json')

    data.save_to_disk(split_path)

    dataset_dict = load_json(json_path)
    dataset_dict['splits'].append(split_name)
    save_json(json_path, dataset_dict)

def overwrite_dataset_dict_split(dataset_path: str | Path, split_name: str, data: Dataset):
    '''Overwrite a split in a dataset dict by removing the old dataset and saving the new one'''
    if isinstance(dataset_path, str): dataset_path = Path(dataset_path)

    split_path = str(dataset_path / split_name)
    clear_dir(split_path)
    data.save_to_disk(split_path)

def add_cols_to_split(distiset: Dataset, split: Dataset, cols: list[str]):
    '''
    Take the rows in distiset, take the source in split that has the same first element as the source in distiset, 
    and make a new row with the values in cols from distiset and the source from split.

    The new split will have the same order as the distiset
    '''
    source_to_row = {}
    for i, source in enumerate(split['source']):
        source_to_row[source[0]] = i
    
    updated_rows = []
    for row in distiset:
        # Find the matching row and update with the question
        split_row = split[source_to_row[row['source'][0]]]
        updated_rows.append(split_row | {col: row[col] for col in cols})
    
    return Dataset.from_list(updated_rows)

# define this as a function to make it pickleable
def generation_is_structured(generation: str) -> bool:
    '''Bool indicator of whether a generation is None or not'''
    return generation is not None

def add_split_label(dataset: list[dict], split: str) -> list[dict]:
    '''Add a split label to a dataset given as a list of dicts'''
    return [{**row, 'split': split} for row in dataset]

def load_pydantic(path, config_class):
    '''load yaml config and convert into pydantic config'''
    with open(path, 'r') as fin:
        config = yaml.safe_load(fin)
    config = {
        k: str(pth(v).expanduser().resolve()) if 'path' in k else v
        for k, v in config.items()
    }
    config = config_class.model_validate(config)
    return config

def is_openai_model_name(model_name: str) -> bool:
    """
    Check if a string is the name of an OpenAI model by matching 'gpt' or 'o' followed by a digit and anything else.
    """
    return bool(re.search(r'(gpt|o\d.*)', model_name, re.IGNORECASE))

def source_to_msg(source: str | list[str], max_dims: tuple[int, int], msg_content_img: Callable) -> dict:
    '''
    Convert a source into an openai message.
    
    A source is a string directly for input, or a list of paths to images or pdf pages.
    '''
    if isinstance(source, str):
            # Text source
        return {'role': 'user', 'content': source}
    else:
        # Image source (list of paths)
        content = []
        for path in source:
            img = get_image(None, path)
            img = downsample_image(img, max_dims)
            b64_img = b64_encode_image(img)
            content.append(msg_content_img(b64_img))
            
        return {'role': 'user', 'content': content}
