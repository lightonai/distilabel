import logging
import json
import os
import socket
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, List, Literal, Union

import portalocker
from pydantic import BaseModel, Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel import utils

if TYPE_CHECKING:
    from logging import Logger

_VLLM_SERVER_PLACEMENT_MIXIN_FILE = None

def set_vllm_server_placement_file(pipeline_name: str) -> Path:
    global _VLLM_SERVER_PLACEMENT_MIXIN_FILE
    _VLLM_SERVER_PLACEMENT_MIXIN_FILE = (
        Path(tempfile.gettempdir())
        / "distilabel"
        / "vllm_server_placement"
        / socket.gethostname()
        / pipeline_name
        / "distilabel_vllm_server_placement_mixin.json"
    )
    return _VLLM_SERVER_PLACEMENT_MIXIN_FILE


_logger = logging.getLogger('vllm_server_placement')


class VLLMServerPlacementMixin(BaseModel):
    """Mixin class to assign a running vLLM server to the `LLM` based on the `VLLM_BASE_URLS_JSON` environment variable.

    Attributes:
        vllm_base_url: The base URL of the vLLM server to be used by the `LLM`. If set
            to "auto", the server will be assigned based on the `VLLM_BASE_URLS_JSON`
            mapping for the model.
        disable_vllm_server_placement: Whether to disable the vLLM server placement logic
            or not. Defaults to `False`.
        _llm_identifier: the identifier of the `LLM` to be used as key in `_server_llm_placement_map`.
        _server_llm_placement_map: a dictionary with the server placement information for each
            `LLM`.
    """

    vllm_base_url: RuntimeParameter[Union[str, Literal["auto"]]] = Field(
        default="auto",
        description="The base URL of the vLLM server to be used. If 'auto', it will be assigned from `VLLM_BASE_URLS_JSON` env var.",
    )
    disable_vllm_server_placement: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to disable the vLLM server placement logic or not.",
    )
    _model_path: str = PrivateAttr(default="")
    _llm_identifier: Union[str, None] = PrivateAttr(default=None)
    _available_vllm_base_urls: List[str] = PrivateAttr(default_factory=list)
    _model_to_vllm_base_url_map: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    def load(self) -> None:
        """Assigns a vLLM server URL to the LLM."""
        if self.disable_vllm_server_placement:
            return

        # Use per-model mapping via VLLM_BASE_URLS_JSON
        mapping_path = os.environ.get("VLLM_BASE_URLS_JSON")
        if mapping_path:
            mapping = utils.load_json(mapping_path)
            if not isinstance(mapping, dict):
                raise ValueError("VLLM_BASE_URLS_JSON must contain a JSON object mapping model names to base URL lists")
            # Normalize keys and ensure values are lists of non-empty strings
            normalized: Dict[str, List[str]] = {}
            for k, v in mapping.items():
                key = str(k)
                if isinstance(v, (list, tuple)):
                    urls = [str(u).strip() for u in v if str(u).strip()]
                else:
                    # Allow single string for convenience, convert to list
                    urls = [str(v).strip()] if str(v).strip() else []
                if urls:
                    normalized[key] = urls
            self._model_to_vllm_base_url_map = normalized
            if self._model_path not in self._model_to_vllm_base_url_map:
                raise ValueError(f"Model '{self._model_path}' not found in VLLM_BASE_URLS_JSON mapping")
            # Provide the available URLs for this model; selection will be balanced in _get_vllm_server
            self._available_vllm_base_urls = self._model_to_vllm_base_url_map[self._model_path]
        else:
            self._available_vllm_base_urls = []

        if self.vllm_base_url == "auto" and not self._available_vllm_base_urls:
            raise ValueError(
                "The `vllm_base_url` is set to 'auto', but the `VLLM_BASE_URLS_JSON` environment"
                " variable is not set. Please, set it to the path of a JSON file mapping"
                " model names to vLLM server base URLs."
            )

        self._assign_vllm_server()

    def unload(self) -> None:
        """Unloads the LLM and removes the vLLM server URL assigned to it from the server
        placement map."""
        if self.disable_vllm_server_placement:
            return

        with self._server_llm_placement_map() as server_map:
            if self._llm_identifier in server_map:
                _logger.debug(  # type: ignore
                    f"Removing '{self._llm_identifier}' from the vLLM server map file"
                    f" '{_VLLM_SERVER_PLACEMENT_MIXIN_FILE}'."
                )
                del server_map[self._llm_identifier]

    @contextmanager
    def _server_llm_placement_map(self) -> Generator[Dict[str, Dict[str, Union[str, None]]], None, None]:
        """Reads the content of the server placement file of the node with a lock, yields
        the content, and writes the content back to the file after the context manager is
        closed. If the file doesn't exist, an empty dictionary will be yielded.

        Yields:
            The content of the server placement file.
        """
        _VLLM_SERVER_PLACEMENT_MIXIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VLLM_SERVER_PLACEMENT_MIXIN_FILE.touch()
        with portalocker.Lock(
            _VLLM_SERVER_PLACEMENT_MIXIN_FILE,
            "r+",
            flags=portalocker.LockFlags.EXCLUSIVE,
        ) as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                content = {}
            yield content
            f.seek(0)
            f.truncate()
            f.write(json.dumps(content))
            f.flush()
            os.fsync(f.fileno())

    def _assign_vllm_server(self) -> None:
        """Assigns a vLLM server URL to the LLM based on the placement information."""
        with self._server_llm_placement_map() as server_map:
            if self.vllm_base_url == "auto":
                if (
                    vllm_base_url := self._get_vllm_server(server_map)
                ) is not None:
                    self.vllm_base_url = vllm_base_url
                else:
                    _logger.warning(  # type: ignore
                        "No available vLLM server found. Could not assign a vLLM server"
                        f" for LLM with identifier '{self._llm_identifier}'."
                    )
            else:
                self._check_vllm_server(server_map)

            if self.vllm_base_url and self.vllm_base_url != "auto":
                server_map[self._llm_identifier] = {  # type: ignore
                    "url": self.vllm_base_url,
                    "model_path": self._model_path,
                }

        if self.vllm_base_url == "auto":
            self.vllm_base_url = None  # type: ignore

        self._set_vllm_api_base_url()

    def _check_vllm_server(self, server_map: Dict[str, Dict[str, Union[str, None]]]) -> None:
        """Checks if the vLLM server URL assigned to the LLM is also assigned to other LLMs.

        Args:
            server_map: a dictionary with the server placement information for each LLM.
        """
        for llm, value in server_map.items():
            server_url = value.get("url")
            if self.vllm_base_url == server_url:
                _logger.warning(  # type: ignore
                    f"LLM with identifier '{llm}' is going to use vLLM server "
                    f"'{self.vllm_base_url}' in addition to other steps."
                )

    def _get_vllm_server(self, server_map: Dict[str, Dict[str, Union[str, None]]]) -> Union[str, None]:
        """Returns the vLLM server URL with the minimum number of assigned LLMs.

        Args:
            server_map: a dictionary with the server placement information for each LLM.

        Returns:
            The vLLM server URL to be used by the LLM.
        """
        if not self._available_vllm_base_urls:
            return None

        # Count usage per model so placement is per-model
        # Build counts for this model only when model_path information is present in the map
        server_counts: Counter = Counter()
        for _, value in server_map.items():
            url = value.get("url")
            entry_model_path = value.get("model_path")
            if entry_model_path == self._model_path:
                server_counts[url] += 1

        return min(self._available_vllm_base_urls, key=lambda url: server_counts.get(url, 0))

    def _set_vllm_api_base_url(self) -> None:
        """Sets the `VLLM_API_BASE_URL` environment variable to the vLLM server URL to be
        used by the LLM.
        """
        if not self.vllm_base_url or self.vllm_base_url == "auto":
            return

        _logger.info(  # type: ignore
            f"🎮 LLM '{self._llm_identifier}' is going to use the following vLLM server:"
            f" {self.vllm_base_url}"
        )
        os.environ["VLLM_API_BASE_URL"] = self.vllm_base_url
