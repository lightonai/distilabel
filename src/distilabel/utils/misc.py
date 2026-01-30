import os
from PIL import Image, ImageDraw, ImageFont
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
from pydantic import BaseModel, ValidationError
from datasets import Dataset, load_from_disk
from pathlib import Path
from copy import deepcopy
import pypdfium2 as pdfium
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .cpe import continuous_parallel_execution
from .image import get_image, downsample_image, b64_encode_image, crop_image

# Cache filename for idx→filename maps
IDX_TO_FILENAME_CACHE = 'idx_to_filename.json'

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
    '''
    Return the page filename from a path and page number. 
    Only works for page filenames, can't take the filename directly.
    If you have the pdf filename, use page_path instead.
    '''
    return path[:path.rfind('_page_')] + f'_page_{page}.pdf'

def page_path(path, page):
    '''
    Return the page filename from a pdf filename and page number.
    If you have the page filename, use path_as_page instead.
    '''
    return path[:path.rfind('.pdf')] + f'_page_{page}.pdf'

def n_pages(path):
    '''Return the number of pages in a pdf'''
    return pdfinfo_from_path(path)['Pages']

def generate_idx_to_filename(ds):
    idx_to_filename = {
        idx: filename
        for idx, filename in enumerate(ds['image_filename'])
    }
    return idx_to_filename

def get_idx_to_filename(ds_path: str | Path) -> dict[int, str]:
    """Return idx→filename mapping for an images Dataset, with on-disk caching.

    A json named IDX_TO_FILENAME_CACHE will be created in the dataset's directory.
    """
    if isinstance(ds_path, str):
        ds_path = Path(ds_path)
    cache_path = ds_path / IDX_TO_FILENAME_CACHE
    if cache_path.exists():
        mapping = load_json(cache_path)
        return {
            int(k): v
            for k, v in mapping.items()
        }
    from datasets import load_from_disk
    ds = load_from_disk(str(ds_path))
    mapping = generate_idx_to_filename(ds)
    save_json(cache_path, mapping)
    return mapping

def generate_field_to_idx(ds, field, substitution: tuple[str | re.Pattern, str] | None = None):
    field_to_idx = {
        substitution[0].sub(substitution[1], field) if substitution else field: idx
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
    '''Add a split label to a dataset given as a list of dicts. 
    For a Dataset, use add_split_label_ds instead.'''
    return [{**row, 'split': split} for row in dataset]

def add_split_label_ds(dataset: Dataset, split: str) -> Dataset:
    '''Add a split label column to a HuggingFace Dataset. 
    Overwrites existing split column if it exists.'''
    if 'split' in dataset.column_names:
        # Overwrite existing column
        dataset = dataset.remove_columns(['split'])
    return dataset.add_column('split', [split] * len(dataset))

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
    distiset = distiset.map(
        hf_batched(lambda x: {'source': map_to_source.get(frozenset(x['source']))}),
        batched=True,
        num_proc=16,
    )
    distiset = distiset.filter(
        hf_batched(lambda x: x['source'] is not None),
        batched=True,
        num_proc=16,
    )
    return distiset

def hf_batched(f: Callable) -> Callable:
    '''
    Wrap a function that works on a single row into a function that works on a batch of rows,
    given as a dict of lists for batching hf dataset processing.
    '''
    def batched_f(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        keys = list(batch.keys())
        rows = [dict(zip(keys, values)) for values in zip(*batch.values())]
        out_rows = [f(r) for r in rows]
        if isinstance(out_rows[0], bool):
            return out_rows
        out = defaultdict(list)
        for row in out_rows:
            for k, v in row.items():
                out[k].append(v)
        return dict(out)
    return batched_f

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

def not_empty_string(row: dict, cols: list[str]) -> bool:
    return all([row[col] != '' for col in cols])

def cols_true(row: dict, cols: list[str]) -> bool:
    '''Bool indicator of whether all of the cols are True'''
    return all([row[col] for col in cols])

def _not_filter(*args, filter: Callable = lambda *args, **kwargs: False, **kwargs):
    return not filter(*args, **kwargs)

def logical_not_filter(filter: Callable) -> Callable:
    '''Return a filter that is the logical negation of the filter'''
    return partial(_not_filter, filter=filter)

def _and_filter(*args, filters: list[Callable] = [], **kwargs):
    for f in filters:
        if not f(*args, **kwargs):
            return False
    return True

def logical_and_filters(*filters: list[Callable]) -> Callable:
    '''Return a filter that is the logical AND of the filters'''
    return partial(_and_filter, filters=filters)

def _or_filter(*args, filters: list[Callable] = [], **kwargs):
    for f in filters:
        if f(*args, **kwargs):
            return True
    return False

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
    if 'oss' in model_name:
        return False
    return bool(re.search(r'(gpt|o\d.*)', model_name, re.IGNORECASE))

def _is_image_path(s: str) -> bool:
    extensions = ['.pdf', '.jpg', '.png']
    return any(s.endswith(ext) for ext in extensions)

def add_index_badge_to_image(
    img: Image.Image,
    idx: int,
    source: Any,
    color: str | tuple[int, int, int] = 'red',
    alpha: int = 192,
    font_path: str | None = None,
    font_size_ratio: float = 0.04,
    padding_ratio: float = 0.02,
) -> Image.Image:
    """Overlay a prominent index number at the top-left of the image.

    The overlay is drawn on the downsampled image for consistent sizing.
    Use partials to customize color/font_path.
    """
    # mostly remove page numbers from the image
    img = crop_image(img, {'top': 0.08, 'bottom': 0.08, 'left': 0, 'right': 0})

    if img.mode != 'RGBA':
        base = img.convert('RGBA')
    else:
        base = img

    width, height = base.size
    # Scale elements relative to shorter side
    shorter_side = min(width, height)
    font_size = max(18, int(shorter_side * font_size_ratio))
    padding = max(6, int(shorter_side * padding_ratio))

    # Choose font: prefer provided TTF path if it exists; else default bitmap font
    font: ImageFont.ImageFont
    if font_path is not None and Path(font_path).exists():
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default(size=font_size)

    text = str(idx + 1)

    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Determine text position and bounding box at that position
    text_pos = (padding, padding)
    text_bbox = draw.textbbox(text_pos, text, font=font)

    # Expand bbox by padding to form the background rectangle
    bg_rect = (
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    )
    draw.rectangle(bg_rect, fill=(0, 0, 0, alpha))

    # Draw the text on top
    draw.text(text_pos, text, font=font, fill=color)

    composed = Image.alpha_composite(base, overlay)
    return composed.convert('RGB')

def source_to_msg(
    source: str | list[str | Image.Image] | None, 
    max_dims: tuple[int, int], 
    msg_content_img: Callable,
    path_substitution: tuple[str, str] | None = None,
    postprocess_image_hook: Callable[[Image.Image, int, Any], Image.Image] | None = None,
    pdf_backend: str = "pdfium",
) -> dict:
    '''
    Convert a source into an openai message.
    
    A source is a string directly for input, 
    a list of strings, a list of either paths to images or pdf pages, 
    or direct PIL Images.
    '''
    if isinstance(source, str):
        # Text source
        return {'role': 'user', 'content': source}
    elif isinstance(source, list):
        # Image source (list of paths)
        content = []
        for idx, item in enumerate(source):
            if isinstance(item, str) and not _is_image_path(item):
                content.append({'type': 'text', 'text': item})
                continue
            if isinstance(item, str):
                img = get_image(None, item, path_substitution, pdf_backend)
            elif isinstance(item, Image.Image):
                img = item
            else:
                continue
            img = downsample_image(img, max_dims)
            if postprocess_image_hook is not None:
                img = postprocess_image_hook(img, idx, source)
            b64_img = b64_encode_image(img)
            content.append(msg_content_img(b64_img))
            
        return {'role': 'user', 'content': content}
    else:
        return {'role': 'user', 'content': None}

def clean_structured_output(
    output: str | None, 
    double_escape: bool = False,
    fix_quotes: bool = False,
) -> str | None:
    '''Remove some common and basic formatting errors.'''
    if output is None:
        return None
    output = (
        output
        .replace('False', 'false')
        .replace('True', 'true')
    )
    if output.startswith('```json'):
        output = output[len('```json'):]
    if output.endswith('```'):
        output = output[:-len('```')]
    output = output.replace('<|begin_of_box|>', '')
    output = output.replace('<|end_of_box|>', '')
    # Double-escape single backslashes only inside JSON string values and normalize raw newlines
    def _double_escape_in_strings(s: str) -> str:
        s = s.replace('\\\\', '\\')
        string_pattern = r'"(?:[^"\\]|\\.)*"'
        def _repl(m: re.Match) -> str:
            quoted = m.group(0)
            inner = quoted[1:-1]
            # Normalize line endings and convert raw newlines to literal \n
            inner = inner.replace('\r\n', '\n').replace('\r', '\n')
            inner = inner.replace('\n', r'\n')
            inner = inner.replace('"', '\\"')
            # Only double lone backslashes that are NOT starting valid JSON escapes
            # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            inner = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', inner)
            return '"' + inner + '"'
        return re.sub(string_pattern, _repl, s, flags=re.DOTALL)
    if double_escape:
        output = _double_escape_in_strings(output)

    def _fix_quotes(s: str) -> str:
        '''
        Double quotes inside string values are not valid json. 
        This function attempts to fix them, but does not handle nested string values.
        '''
        field_seperator = '",\n'
        key_value_seperator = ': '
        s = s.split(field_seperator)
        corrected = []
        for field in s:
            v_start = field.find(key_value_seperator) + len(key_value_seperator)

            k, v = field[:v_start], field[v_start:]
            if v.startswith('"'): 
                v = v[1:]
            else:
                # bail on non-string values
                corrected.append(field)
                continue
            v = v.replace('"', r'\"')
            corrected.append(f'{k}"{v}')
        return field_seperator.join(corrected)
    if fix_quotes:
        output = _fix_quotes(output)

    output = output[output.find('{'):]
    return output

def try_model_validate(model: BaseModel, output: str | None, **kwargs) -> BaseModel | None:
    if output is None:
        return None

    candidates = (
        output,
        clean_structured_output(output, double_escape=False),
        clean_structured_output(output, double_escape=True),
        clean_structured_output(output, fix_quotes=True),
        clean_structured_output(output, double_escape=True, fix_quotes=True),
    ) # progressively stonger replacements

    last_exc: ValidationError | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return model.model_validate_json(candidate, **kwargs)
        except ValidationError as e:
            last_exc = e

    if last_exc is not None:
        raise last_exc

def _get_pdf_paths_from_disk(pdf_root: Path | str, limit: int | None = None):
    """
    Walks directories to find all PDF paths.
    Can stop early if a limit is provided. The limit is roughly obeyed.
    """
    paths = []
    for subdir in Path(pdf_root).iterdir():
        for root, _, files in os.walk(subdir):
            for file in files:
                if file.endswith(".pdf"):
                    paths.append(os.path.join(root, file))
            
            if limit and len(paths) >= limit:
                return paths
    return paths

def get_pdf_paths(pdf_root: Path | str, cache_dir: Path | str = 'out'):
    '''
    Gets all PDF paths, using a cache file to speed up subsequent runs.
    '''
    pdf_paths_cache = Path(cache_dir) / 'pdf_paths_cache.txt'
    if pdf_paths_cache.exists():
        with open(pdf_paths_cache, 'r') as f:
            return [line.strip() for line in f]

    paths = _get_pdf_paths_from_disk(pdf_root)

    os.makedirs(cache_dir, exist_ok=True)
    with open(pdf_paths_cache, 'w') as f:
        for p in paths:
            f.write(f"{p}\n")
    return paths

def get_page_count(pdf_path: str) -> tuple[str, int]:
    """Get the number of pages in a PDF."""
    try:
        return pdf_path, len(pdfium.PdfDocument(pdf_path))
    except Exception:
        return pdf_path, 0

def count_all_pages(
    pdf_root: Path | str,
    cache_dir: Path | str = 'out',
    n_jobs: int = 16,
    limit: int | None = None,
):
    '''
    Count the number of pages in all PDFs in a directory.

    pdf_root: root directory of the pdfs
    cache_dir: directory to cache the pdf paths
    n_jobs: number of jobs to use for counting pages
    limit: limit the number of pdfs to count pages for (for testing)

    Returns:
        dict of pdf_path to page_count
    '''
    path_to_page_count_path = pth(cache_dir) / 'path_to_page_count.json'
    if path_to_page_count_path.exists():
        return load_json(path_to_page_count_path)
    
    if limit:
        pdf_paths = _get_pdf_paths_from_disk(pdf_root, limit)
    else:
        pdf_paths = get_pdf_paths(pdf_root, cache_dir)
    with mp.Pool(n_jobs) as pool:
        page_counts_iterator = pool.imap_unordered(get_page_count, pdf_paths)
        path_to_page_count = {
            pdf_path: page_count
            for pdf_path, page_count in tqdm(page_counts_iterator, desc='Counting pages', total=len(pdf_paths))
            if page_count > 0
        }
    
    save_json(path_to_page_count_path, path_to_page_count)
    return path_to_page_count

def resolve_path(path: str, path_substitution: tuple[str, str | re.Pattern] | None = None) -> str:
    if path_substitution is None:
        return path
    if isinstance(path_substitution[0], re.Pattern):
        return path_substitution[0].sub(path_substitution[1], path)
    else:
        return path.replace(path_substitution[0], path_substitution[1])

def remove_pdfs_from_dataset(
    dataset: Dataset, 
    exclude_pdfs: set[str], 
    row_to_ifn: Callable | None = None,
    num_proc: int = 16,
):
    '''
    Remove all image filenames that are from pdfs in the exclude_pdfs set
    row_to_ifn is a function that takes a row and returns the pdf name, defaults to pdf_name
    '''
    return dataset.filter(
        hf_batched(lambda x: pdf_name(row_to_ifn(x) if row_to_ifn else x['image_filename']) not in exclude_pdfs),
        batched=True,
        num_proc=num_proc,
    )

def remove_pdfs_with_pages_(
    dataset: Dataset, 
    pdf_root: Path | str, 
    cache_dir: Path | str = 'out',  
    row_to_ifn: Callable = lambda row: row['image_filename'],
    less_than: int = 0,
    more_than: int = 10_000,
    num_proc: int = 16,
):
    '''
    Remove all pdfs that have less than less_than pages or more than more_than pages from the dataset.
    Uses count_all_pages to get the page counts.
    '''
    fn_to_page_count = count_all_pages(
        pdf_root=pdf_root,
        cache_dir=cache_dir,
        n_jobs=num_proc,
    )
    return dataset.filter(
        hf_batched(lambda x: less_than <= fn_to_page_count.get(pdf_name(row_to_ifn(x)), 0) <= more_than),
        batched=True,
        num_proc=num_proc,
    )

def take_n_first_doc_occurrences(
    dataset: Dataset, 
    row_to_ifn: Callable = lambda row: row['image_filename'], 
    _resolve_path: Callable = lambda x: x,
    n: int = 1,
) -> Dataset:
    docs_used = defaultdict(int)
    ifns = [row_to_ifn(row) for row in dataset]
    pruned_ds = []
    for idx, ifn in enumerate(ifns):
        pdf_path = pdf_name(_resolve_path(ifn))
        docs_used[pdf_path] += 1
        if docs_used[pdf_path] > n:
            continue
        pruned_ds.append(idx)
    dataset = dataset.select(pruned_ds).flatten_indices()
    return dataset

def filter_path_exists(
    dataset: Dataset, 
    row_to_ifn: Callable = lambda row: row['image_filename'],
    resolve_path: Callable = lambda x: x,
    num_proc: int = 16,
) -> Dataset:
    docs = set()
    for row in dataset:
        doc = resolve_path(pdf_name(row_to_ifn(row)))
        if Path(doc).exists():
            docs.add(doc)
    return dataset.filter(
        hf_batched(lambda x: resolve_path(pdf_name(row_to_ifn(x))) in docs),
        batched=True,
        num_proc=num_proc,
    )

fn_to_idx: dict[str, int] | None = None

def default_conversion_fn(row: dict, path_substitution: tuple[str | re.Pattern, str] | None = None, **kwargs) -> dict:
    '''
    Default conversion function for the distiset.
    '''
    global fn_to_idx
    image_indices = [
        fn_to_idx[
            (
                path_substitution[0].sub(path_substitution[1], ifn) 
                if isinstance(path_substitution[0], re.Pattern)
                else ifn.replace(path_substitution[0], path_substitution[1])
            )
            if path_substitution else ifn
        ] for ifn in row['source']
    ]

    user_content = (
        ''.join([f'<IMG_{i}>' for i in range(len(image_indices))])
        + row['question']
    )
    assistant_content = row['answer']
    messages = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content}
    ]
    return {
        'images': image_indices,
        'messages': messages,
        'n_images': len(image_indices),
    }

def format_distiset(
    distiset: Dataset,
    conversion_fn: Callable = default_conversion_fn,
    path_substitution: tuple[str, str] | None = None,
    images_ds_path: str | Path | None = None,
    cols_to_keep: list[str] = [],
    n_workers: int = 16,
) -> Dataset:
    '''
    Format the distiset to vision format/build actual examples from extractions

    Args:
        distiset: The distiset to format
        conversion_fn: The function to convert the row to vision format.
            Default simply uses the source col as images, the question as 
            user content, and the answer as assistant content. If using 
            the default, you must provide the images_ds_path.
        path_substitution: The substitution to make to the image filenames in the images_ds_path.
            Only needed if using the default conversion function.
        cols_to_keep: The columns to keep from the distiset aside from 
            `['images', 'messages', 'n_images']`
        n_workers: The number of workers to use for the parallel execution
        images_ds_path: The path to the images dataset. Only needed if using
            the default conversion function.
    '''
    distiset = distiset.to_list()

    if conversion_fn is default_conversion_fn:
        global fn_to_idx
        if fn_to_idx is None:
            images_ds = load_from_disk(images_ds_path)
            fn_to_idx = generate_field_to_idx(images_ds, 'image_filename', path_substitution)

    tqdm_desc = "Formatting to vision"
    conversion_fn = partial(conversion_fn, path_substitution=path_substitution)
    cpe = continuous_parallel_execution(
        function=conversion_fn,
        tasks=[{'row': row, 'idx': idx} for idx, row in enumerate(distiset)],
        task_count=len(distiset),
        process_type="process",
        num_workers=n_workers,
        max_active_tasks=1024,
        tqdm_desc=tqdm_desc,
    )
    vision_ds = [None] * len(distiset)
    for task, result in cpe:
        vision_ds[task['idx']] = task['row'] | result

    distiset = Dataset.from_list(vision_ds).select_columns(list(set(['images', 'messages', 'n_images'] + cols_to_keep)))
    return distiset

def sample_match_target_histogram(
    n_images: np.ndarray,
    min_total_images: int,
    bin_edges: np.ndarray,
    target_bin_probs: np.ndarray,
    *,
    seed: int = 17,
    allow_replacement_long_bins: bool = False,
    max_examples: int | None = None,
) -> np.ndarray:
    """Select example indices to match a target n_images distribution with a
    minimum total images constraint.

    Parameters
    ----------
    n_images : np.ndarray
        Per-example number of images (ints). Shape: (N,).
    min_total_images : int
        Minimum total number of images the selected subset should sum to.
    bin_edges : np.ndarray
        Monotonic array of bin edges for ``np.digitize``; length = B+1.
    target_bin_probs : np.ndarray
        Desired probability mass per bin; length = B. Non-negative. Will be
        normalized over non-empty bins.
    seed : int, optional
        RNG seed.
    allow_replacement_long_bins : bool, optional
        If True, bins may sample with replacement when requested count exceeds
        capacity; otherwise counts are capped by capacity.
    max_examples : int | None, optional
        If provided, cap the total number of selected examples (may make it
        impossible to reach ``min_total_images`` without replacement).

    Returns
    -------
    np.ndarray
        Selected example indices (dtype=int) shuffled.

    Example
    -------
    bin_edges = np.array([0, 4, 8, 16, 32, 64, 128, 192, 256, 336, 10_000], dtype=int)  # last edge large to cap
    target_probs = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 0], dtype=float)
    sel_idxs = sample_match_target_histogram(
        n_images=n_images, 
        min_total_images=100, 
        bin_edges=bin_edges, 
        target_bin_probs=target_probs,
    )
    """
    # overall process is: 
    #   Loop until we have enough images (estimated based on average images per bin and current example counts per bin)
        #   estimate the gain n_images per sampled example by probs * avg_images_per_bin
        #   estimate a number of examples to take from the bins as (min_total_images - current_images) / expected_gain_per_example
        #   sample that many examples from the bins according to target_probs
        #   add the selected examples to the current examples
        #   repeat until we have enough images
    #   We only have a number of examples to take from each bin, sample these from each
    #   Shuffle and return the selected idxs

    if n_images.ndim != 1:
        raise ValueError("n_images must be a 1D array")
    if len(bin_edges) < 2:
        raise ValueError("bin_edges must have at least two values")
    if len(target_bin_probs) != len(bin_edges) - 1:
        raise ValueError("target_bin_probs must be len(bin_edges) - 1")
    if np.any(np.asarray(target_bin_probs) < 0):
        raise ValueError("target_bin_probs must be non-negative")
    

    rng = np.random.default_rng(seed)
    num_examples = int(n_images.shape[0])
    if num_examples == 0 or min_total_images <= 0:
        return np.array([], dtype=int)

    # Bin assignments and capacities
    num_bins = len(bin_edges) - 1
    bin_ids = np.clip(np.digitize(n_images, bin_edges, right=True) - 1, 0, num_bins - 1)
    bin_to_indices: list[np.ndarray] = [np.where(bin_ids == b)[0] for b in range(num_bins)]
    capacity_per_bin = np.array([idxs.size for idxs in bin_to_indices], dtype=int)
    non_empty_mask = capacity_per_bin > 0
    if not np.any(non_empty_mask):
        raise ValueError("All bins are empty after binning; check bin_edges.")

    # Normalize target probabilities over non-empty bins
    probabilities = np.asarray(target_bin_probs, dtype=float).copy()
    probabilities[~non_empty_mask] = 0.0
    prob_sum = float(probabilities.sum())
    if prob_sum <= 0.0:
        raise ValueError("Target probabilities sum to zero over non-empty bins.")
    probabilities /= prob_sum

    # Average images per bin
    avg_images_per_bin = np.array([
        float(n_images[idxs].mean()) if idxs.size > 0 else 0.0 for idxs in bin_to_indices
    ], dtype=float)

    # Preserve previous behavior: raise if no expected images under target
    if float((probabilities * avg_images_per_bin).sum()) <= 0.0:
        raise ValueError("Expected images per sample is zero; check probs/binning.")

    counts_per_bin = np.zeros(num_bins, dtype=int)
    max_total_examples = int(max_examples) if max_examples is not None else np.iinfo(np.int64).max
    if not allow_replacement_long_bins:
        max_total_examples = min(max_total_examples, int(capacity_per_bin.sum()))

    # Allocate counts in batches until reaching the image budget or capacity
    while True:
        current_images = float((counts_per_bin * avg_images_per_bin).sum())
        if current_images >= float(min_total_images):
            break

        if allow_replacement_long_bins:
            available_bins = np.arange(num_bins)
            active_probabilities = probabilities
            remaining_slots = int(max_total_examples - counts_per_bin.sum())
            if remaining_slots <= 0:
                break
            expected_per_increment = float((active_probabilities * avg_images_per_bin).sum())
        else:
            spare_capacity = capacity_per_bin - counts_per_bin
            available_bins = np.where(spare_capacity > 0)[0]
            if available_bins.size == 0:
                break
            active_probabilities = probabilities[available_bins]
            s = float(active_probabilities.sum())
            active_probabilities = active_probabilities / s if s > 0 else np.ones_like(active_probabilities) / active_probabilities.size
            remaining_slots = int(min(spare_capacity[available_bins].sum(), max_total_examples - counts_per_bin.sum()))
            if remaining_slots <= 0:
                break
            expected_per_increment = float((active_probabilities * avg_images_per_bin[available_bins]).sum())

        if expected_per_increment <= 0.0:
            break
        images_needed = float(min_total_images - current_images)
        increment_count = int(np.ceil(images_needed / expected_per_increment))
        increment_count = max(0, min(increment_count, remaining_slots))
        if increment_count == 0:
            break

        if allow_replacement_long_bins:
            add_counts = rng.multinomial(increment_count, active_probabilities)
            counts_per_bin = counts_per_bin + add_counts
        else:
            add_local = rng.multinomial(increment_count, active_probabilities)
            add_global = np.zeros_like(counts_per_bin)
            add_global[available_bins] = np.minimum(add_local, spare_capacity[available_bins])
            counts_per_bin = counts_per_bin + add_global

    # Sample indices within each bin
    if counts_per_bin.sum() <= 0:
        return np.array([], dtype=int)
    selected_parts: list[np.ndarray] = []
    for b in range(num_bins):
        cnt = int(counts_per_bin[b])
        if cnt <= 0:
            continue
        idxs = bin_to_indices[b]
        replace_flag = bool(allow_replacement_long_bins and cnt > idxs.size)
        chosen = rng.choice(idxs, size=cnt, replace=replace_flag)
        selected_parts.append(chosen.astype(int, copy=False))

    selection = np.concatenate(selected_parts, axis=0) if selected_parts else np.array([], dtype=int)
    return rng.choice(selection, size=len(selection), replace=False)
