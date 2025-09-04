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

if TYPE_CHECKING:
    from logging import Logger

_VLLM_SERVER_PLACEMENT_MIXIN_FILE = (
    Path(tempfile.gettempdir())
    / "distilabel"
    / "vllm_server_placement"
    / socket.gethostname()
    / "distilabel_vllm_server_placement_mixin.json"
)

if _VLLM_SERVER_PLACEMENT_MIXIN_FILE.exists():
    _VLLM_SERVER_PLACEMENT_MIXIN_FILE.unlink()


_logger = logging.getLogger('vllm_server_placement')


class VLLMServerPlacementMixin(BaseModel):
    """Mixin class to assign a running vLLM server to the `LLM` based on the `VLLM_BASE_URLS` environment variable.

    Attributes:
        vllm_base_url: The base URL of the vLLM server to be used by the `LLM`. If set
            to "auto", the server will be automatically assigned based on the environment
            variable `VLLM_BASE_URLS`.
        disable_vllm_server_placement: Whether to disable the vLLM server placement logic
            or not. Defaults to `False`.
        _llm_identifier: the identifier of the `LLM` to be used as key in `_server_llm_placement_map`.
        _server_llm_placement_map: a dictionary with the server placement information for each
            `LLM`.
    """

    vllm_base_url: RuntimeParameter[Union[str, Literal["auto"]]] = Field(
        default="auto",
        description="The base URL of the vLLM server to be used. If 'auto', it will be assigned from `VLLM_BASE_URLS` env var.",
    )
    disable_vllm_server_placement: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to disable the vLLM server placement logic or not.",
    )
    _llm_identifier: Union[str, None] = PrivateAttr(default=None)
    _available_vllm_base_urls: List[str] = PrivateAttr(default_factory=list)

    _logger: "Logger" = PrivateAttr(None)

    def load(self) -> None:
        """Assigns a vLLM server URL to the LLM."""
        if self.disable_vllm_server_placement:
            return

        vllm_base_urls = os.environ.get("VLLM_BASE_URLS")
        if vllm_base_urls:
            self._available_vllm_base_urls = [
                url.strip() for url in vllm_base_urls.split(",") if url.strip()
            ]

        if self.vllm_base_url == "auto" and not self._available_vllm_base_urls:
            raise ValueError(
                "The `vllm_base_url` is set to 'auto', but the `VLLM_BASE_URLS` environment"
                " variable is not set. Please, set it to a comma-separated list of vLLM"
                " server base URLs."
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
    def _server_llm_placement_map(self) -> Generator[Dict[str, str], None, None]:
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
                server_map[self._llm_identifier] = self.vllm_base_url  # type: ignore

        if self.vllm_base_url == "auto":
            self.vllm_base_url = None  # type: ignore

        self._set_vllm_api_base_url()

    def _check_vllm_server(self, server_map: Dict[str, str]) -> None:
        """Checks if the vLLM server URL assigned to the LLM is also assigned to other LLMs.

        Args:
            server_map: a dictionary with the server placement information for each LLM.
        """
        for llm, server_url in server_map.items():
            if self.vllm_base_url == server_url:
                _logger.warning(  # type: ignore
                    f"LLM with identifier '{llm}' is also going to use vLLM server "
                    f"'{self.vllm_base_url}'. This may lead to performance issues."
                )

    def _get_vllm_server(self, server_map: Dict[str, str]) -> Union[str, None]:
        """Returns the vLLM server URL with the minimum number of assigned LLMs.

        Args:
            server_map: a dictionary with the server placement information for each LLM.

        Returns:
            The vLLM server URL to be used by the LLM.
        """
        if not self._available_vllm_base_urls:
            return None

        server_counts = Counter(server_map.values())
        return min(
            self._available_vllm_base_urls, key=lambda url: server_counts.get(url, 0)
        )

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
