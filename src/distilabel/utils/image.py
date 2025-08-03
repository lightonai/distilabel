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

import base64
import io
from pdf2image import convert_from_path
from PIL import Image
from datasets import Dataset
from pathlib import Path as pth
import traceback as tb
import pypdfium2 as pdfium

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def image_to_str(image: "Image.Image", image_format: str = "JPEG") -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def _render_image_with_pdfium(
    page: pdfium.PdfPage,
    max_resolution: int = 1280,
    scale: float = 2.77,
) -> Image:
    """Render a PDF page to a PIL Image using pdfium.

    Args:
    ----
        page: The PDF page object to render.
        max_resolution: The maximum resolution for the rendered image. Defaults to 1280.
        scale: The scale factor to apply to the rendering. Defaults to 2.77 (≈200 DPI).

    """
    width, height = page.get_size()

    # if scale=2.77 (≈200 DPI), this gives the size at 200 DPI
    pixel_width = width * scale
    pixel_height = height * scale

    # calculate the scaling factor
    resize_factor = min(max_resolution / pixel_width, max_resolution / pixel_height)

    # apply this factor to the original scale
    target_scale = scale * resize_factor

    # render directly at the calculated target scale
    return page.render(scale=target_scale, rev_byteorder=True).to_pil()

def load_from_pdf(filename: str | pth, pdf_backend: str = "pdfium") -> Image.Image:
    """Take a pdf "filename" w/ page and load that page into a PIL.Image.

    Args:
    ----
        filename:
            e.g. '/mnt/nfs/pdfs/pdfs_train/.../name_page_xxx.pdf'
        pdf_backend:
            "pdf2image" (default) or "pdfium"

    """
    filename = str(filename)
    page = int(filename[filename.rfind("page_") + 5 : filename.rfind(".pdf")])
    path = filename[: filename.rfind("_page_")] + ".pdf"

    if pdf_backend == "pdf2image":
        pages = convert_from_path(path, first_page=page + 1, last_page=page + 1)
        return pages[0]
    if pdf_backend == "pdfium":
        pdf = pdfium.PdfDocument(path)
        page_obj = pdf[page]  # pdfium uses 0-based indexing
        image = _render_image_with_pdfium(page_obj)
        pdf.close()
        return image
    message = "Invalid pdf_backend. Choose 'pdf2image' or 'pdfium'."
    raise ValueError(message)

def load_from_filename(
    filename: str, 
    crop: dict[str, int] | None = None,
    path_substitution: tuple[str, str] | None = None,
):
    """
    Loads an image based on a filename which includes the page as shown below

    Works for pdfs or jpgs/pngs and will load the jpg/png image if it exists for improved speed

    See load_from_pdf() for pdf filename format

    crop is an optional dict with keys from ['top', 'bottom', 'left', 'right']
    which can specify a percent of the image to crop off the top, bottom, left, or right
    """
    if path_substitution is not None:
        filename = str(filename).replace(
            path_substitution[0],
            path_substitution[1],
        )
    filename = pth(filename)
    if filename.with_suffix('.jpg').exists():
        image = Image.open(filename.with_suffix('.jpg'))
    elif filename.with_suffix('.png').exists():
        image = Image.open(filename.with_suffix('.png'))
    elif filename.suffix == '.pdf':
        image = load_from_pdf(filename)
    else:
        raise NotImplementedError(f'suffix for {filename=} is not supported')
    
    if crop is not None:
        w, h = image.size
        image = image.crop((
            crop['left'] * w,
            crop['top'] * h,
            (1 - crop['right']) * w,
            (1 - crop['bottom']) * h,
        ))
    return image

def get_image(
    ds: Dataset | None, 
    image_ptr: str | int,
    path_substitution: tuple[str, str] | None = None,
):
    '''
    This will load an image as a PIL Image from your dataset or a pdf or a jpg outside the dataset

    Standard use:
        Make a dictionary filename_to_idx, then
        ```python
        get_image(dataset, filename_to_index.get(filename, filename)))
        ```
    
    Essentially, your dataset can be configured in the following ways:
        - PIL Images stored in the 'image' column
        - At the index you want to load, the 'image' column has a None value and 'image_filename' is 
        formatted according to load_image()
        - You ignore the dataset and just pass an 'image_filename' to be loaded with load_image(). This is 
        needed when you have image_filenames that point to images that are not in the dataset.
    
    Args:
        ds (datasets.Dataset): The dataset or None if it is not needed
        image_ptr (str | int):
            - If it is a string, pass it directly to load_image()
            - If it is an int, load the image at ds[image_ptr] from either the 'image' or 'image_filename' column
    '''
    if ds is None:
        assert isinstance(image_ptr, str)
    try:
        if isinstance(image_ptr, str):
            return load_from_filename(
                image_ptr, 
                path_substitution=path_substitution,
            )
        elif isinstance(image_ptr, int) and ds[image_ptr]["image"] is None:
            return load_from_filename(
                ds[image_ptr]["image_filename"], 
                crop=ds[image_ptr].get('crop'), 
                path_substitution=path_substitution,
            )
        else:
            return ds[image_ptr]["image"]
    except NotImplementedError as exc:
        raise exc
    except Exception as exc:
        raise RuntimeError((
            'Unable to load the requested image, your dataset may be misconfigured. '
            f'received {image_ptr=}\n\n{tb.format_exc(exc)}'
        ))

def downsample_image(image: Image, max_dims: tuple[int, int] = (1000, 1100)):
    '''
    uses Lanczos (high quality) resampling to resample an image while maintaining aspect ratio
    
    image.thumbnail() resamples while maintaining aspect ratio
    '''
    size = image.size
    # horizontal or vertical layout
    dims = max_dims if size[1] > size[0] else max_dims[::-1]
    image.thumbnail(dims, Image.Resampling.LANCZOS)
    return image

def b64_encode_image(image: Image):
    '''
    Encode an image as a base64 string
    '''
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def b64_decode_image(b64_string: str) -> Image.Image:
    '''
    Decode a base64 string back to a PIL Image
    '''
    img_data = base64.b64decode(b64_string)
    buffered = io.BytesIO(img_data)
    image = Image.open(buffered)
    return image

def msg_content_img(image: Image.Image):
    """Convert image to chat message dict with type image."""
    return {
        "type": "image",
        "image": image,
    }

def msg_content_img_url(b64_image):
    """Convert image to chat message dict with type image_url for vLLM/OpenAI."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64_image}",
        },
    }

def msg_content_img_anthropic(b64_image):
    '''Convert image to chat message dict with type image for Anthropic'''
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64_image,
        },
    }
