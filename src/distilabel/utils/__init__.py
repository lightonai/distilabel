# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Image utilities
from .image import (
    image_to_str,
    load_from_pdf,
    load_from_filename,
    get_image,
    downsample_image,
    b64_encode_image,
    b64_decode_image,
    msg_content_img_url,
    msg_content_img_anthropic,
)

# Misc utilities
from .misc import (
    add_split_to_dataset_dict,
    add_cols_to_split,
    overwrite_dataset_dict_split,
    normalize_distribution,
    load_json,
    save_json,
    pdf_name, 
    pdf_page,
    n_pages,
    path_as_page,
    page_path,
    generate_idx_to_filename,
    generate_field_to_idx,
    clear_dir,
    source_to_msg,
    generation_is_structured,
    add_split_label,
    is_openai_model_name,
    suppress_output,
    clean_structured_output,
    sort_adjacent_pages,
    replace_source_col,
    randomize_source_order,
    save_jsonl,
    load_jsonl,
    hash_structure_with_images,
    cols_true,
    logical_not_filter,
    logical_and_filters,
    logical_or_filters,
    get_pdf_paths,
    count_all_pages,
    not_empty_string,
    remove_pdfs_from_dataset,
    remove_pdfs_with_pages_,
)

from .cpe import continuous_parallel_execution

# Timer
from .timer import get_timer

__all__ = [
    # Image utilities
    "image_to_str",
    "load_from_pdf",
    "load_from_filename",
    "get_image",
    "downsample_image",
    "b64_encode_image",
    "b64_decode_image",
    "msg_content_img_url",
    "msg_content_img_anthropic",
    
    # Misc utilities
    "normalize_distribution",
    "load_json",
    "save_json",
    "pdf_name", 
    "pdf_page",
    "n_pages",
    "path_as_page",
    "page_path",
    "generate_idx_to_filename",
    "generate_field_to_idx",
    "clear_dir",
    "add_split_to_dataset_dict",
    "overwrite_dataset_dict_split",
    "source_to_msg",
    "add_cols_to_split",
    "generation_is_structured",
    "add_split_label",
    "is_openai_model_name",
    "suppress_output",
    "clean_structured_output",
    "sort_adjacent_pages",
    "replace_source_col",
    "randomize_source_order",
    "save_jsonl",
    "load_jsonl",
    "hash_structure_with_images",
    "cols_true",
    "logical_not_filter",
    "logical_and_filters",
    "logical_or_filters",
    "get_pdf_paths",
    "count_all_pages",
    "continuous_parallel_execution",
    "not_empty_string",
    "remove_pdfs_from_dataset",
    "remove_pdfs_with_pages_",
    # Timer
    "get_timer",
]
