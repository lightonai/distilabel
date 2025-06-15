import os
from PIL import Image
from functools import partial
import io
import hashlib
import json
from pdf2image import pdfinfo_from_path
from pathlib import Path as pth
import yaml
from contextlib import contextmanager
from io import StringIO
import sys
import re
import random
from queue import Queue
from typing import Callable, List, Any
from datasets import Dataset
from pathlib import Path
from copy import deepcopy

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
    path = pth(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_jsonl(path, data, append=True):
    with open(path, 'a' if append else 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

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

def add_split_label(dataset: list[dict], split: str) -> list[dict]:
    '''Add a split label to a dataset given as a list of dicts'''
    return [{**row, 'split': split} for row in dataset]

def sort_adjacent_pages(dataset: list[dict]) -> list[dict]:
    '''Sort the 'source' col containing adjacent pages in a dataset given as a list of dicts'''
    return [
        row | {'source': sorted(row['source'], key=pdf_page)}
        for row in dataset
    ]

def randomize_source_order(dataset: list[dict]) -> list[dict]:
    '''Randomize the order of the 'source' col in a dataset given as a list of dicts'''
    return [
        row | {'source': random.sample(row['source'], len(row['source']))}
        for row in dataset
    ]

def replace_source_col(distiset: Dataset, dataset: list[dict]):
    '''
    Replace the source col of the distiset with the source col of the dataset
    to retain the original order
    '''
    map_to_source = {frozenset(row['source']): row['source'] for row in dataset}
    distiset = distiset.map(lambda x: {'source': map_to_source.get(frozenset(x['source']))}, num_proc=16)
    distiset = distiset.filter(lambda x: x['source'] is not None)
    return distiset

def hash_structure_with_images(obj: Any) -> str:
    """Deterministic hash of a recursive structure.

    Creates a stable hash for any structure containing dictionaries, lists, strings,
    and PIL Images. Returns a hex digest that will be consistent across different runs.

    Parameters
    ----------
    obj:
        The object to hash. Can contain nested dictionaries, lists, and PIL Images.

    Returns
    -------
        A SHA-256 hex digest that uniquely identifies the content of the object.

    Examples
    --------
    >>> from PIL import Image
    >>> import numpy as np

    >>> # Create two different red images
    >>> red_img1 = Image.new('RGB', (100, 100), color='red')
    >>> red_img2 = Image.new('RGB', (100, 100), color='red')

    >>> # Create a blue image
    >>> blue_img = Image.new('RGB', (100, 100), color='blue')

    >>> # Test 1: Same structure with same content should have same hash
    >>> test_dict1 = {"text": "hello", "image": red_img1}
    >>> test_dict2 = {"text": "hello", "image": red_img2}
    >>> hash1 = hash_structure_with_images(test_dict1)
    >>> hash2 = hash_structure_with_images(test_dict2)
    >>> hash1 == hash2
    True

    >>> # Test 2: Different content should have different hashes
    >>> test_dict3 = {"text": "hello", "image": blue_img}
    >>> hash3 = hash_structure_with_images(test_dict3)
    >>> hash1 == hash3
    False

    >>> # Test 3: Order of keys shouldn't matter
    >>> test_dict4 = {"image": red_img1, "text": "hello"}
    >>> hash4 = hash_structure_with_images(test_dict4)
    >>> hash1 == hash4
    True

    >>> # Test 4: Nested structures
    >>> nested1 = {"outer": {"inner": [1, 2, red_img1]}}
    >>> nested2 = {"outer": {"inner": [1, 2, red_img2]}}
    >>> nested3 = {"outer": {"inner": [1, 2, blue_img]}}
    >>> hash_nested1 = hash_structure_with_images(nested1)
    >>> hash_nested2 = hash_structure_with_images(nested2)
    >>> hash_nested3 = hash_structure_with_images(nested3)
    >>> hash_nested1 == hash_nested2  # Same structure, same images
    True
    >>> hash_nested1 == hash_nested3  # Same structure, different images
    False

    """

    # Helper function to process the structure recursively
    def process_obj(item: str | dict | list) -> str | dict | list:
        """Make the structure hashable recursively."""
        if isinstance(item, dict):
            # Sort the keys and recursively process each value
            processed_dict = {}
            for key in sorted(item.keys()):  # Sort keys for consistent ordering
                processed_dict[str(key)] = process_obj(item[key])
            return processed_dict

        if isinstance(item, list):
            # Recursively process each item in the list
            return [process_obj(i) for i in item]

        if isinstance(item, Image.Image):
            # Convert PIL Image to bytes for consistent hashing
            img_bytes = io.BytesIO()
            item.save(img_bytes, format="PNG")
            # Return a special identifier for images
            return f"IMAGE:{hashlib.md5(img_bytes.getvalue()).hexdigest()}"  # noqa: S324

        if isinstance(item, (str, int, float, bool)) or item is None:
            # Basic types can be returned as is
            return item

        # For any other type, convert to string
        return str(item)

    # Process the structure
    processed = process_obj(obj)

    # Convert to JSON string and hash
    serialized = json.dumps(processed, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# define this as a function to make it pickleable
def generation_is_structured(row: dict, cols: list[str]) -> bool:
    '''Bool indicator of whether any of the cols are None'''
    return all([row[col] is not None for col in cols])

def cols_true(row: dict, cols: list[str]) -> bool:
    '''Bool indicator of whether all of the cols are True'''
    return all([row[col] for col in cols])

def _not_filter(*args, filter: Callable = lambda *args, **kwargs: False, **kwargs):
    return not filter(*args, **kwargs)

def logical_not_filter(filter: Callable) -> Callable:
    '''Return a filter that is the logical negation of the filter'''
    return partial(_not_filter, filter=filter)

def _and_filter(*args, filters: list[Callable] = [], **kwargs):
    return all([f(*args, **kwargs) for f in filters])

def logical_and_filters(*filters: list[Callable]) -> Callable:
    '''Return a filter that is the logical AND of the filters'''
    return partial(_and_filter, filters=filters)

def _or_filter(*args, filters: list[Callable] = [], **kwargs):
    return any([f(*args, **kwargs) for f in filters])

def logical_or_filters(*filters: list[Callable]) -> Callable:
    '''Return a filter that is the logical OR of the filters'''
    return partial(_or_filter, filters=filters)

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

def source_to_msg(source: str | list[str] | None, max_dims: tuple[int, int], msg_content_img: Callable) -> dict:
    '''
    Convert a source into an openai message.
    
    A source is a string directly for input, or a list of paths to images or pdf pages.
    '''
    if isinstance(source, str):
        # Text source
        return {'role': 'user', 'content': source}
    elif isinstance(source, list):
        # Image source (list of paths)
        content = []
        for path in source:
            img = get_image(None, path)
            img = downsample_image(img, max_dims)
            b64_img = b64_encode_image(img)
            content.append(msg_content_img(b64_img))
            
        return {'role': 'user', 'content': content}
    else:
        return {'role': 'user', 'content': None}

def clean_structured_output(output: str | None) -> str | None:
    '''Remove some common and basic formatting errors.'''
    if output is None:
        return None
    output = output.replace('```json', '').replace('```', '')
    return output

def read_queue(queue: Queue, lock: Any) -> List:
    with lock:
        contents = [queue.get() for _ in range(queue.qsize())]
        for batch in contents:
            queue.put(batch)
    return contents
