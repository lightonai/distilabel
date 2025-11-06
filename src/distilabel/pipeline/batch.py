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
import copy
import json
from uuid import uuid4
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from upath import UPath
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from distilabel.utils import get_timer
from distilabel.mixins.signature import SignatureMixin
from distilabel.utils.serialization import _Serializable
from .routed_to_cache import get_routed_to_cache_db

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from logging import Logger

_timer = get_timer()

class _Batch(_Serializable, SignatureMixin):
    """Pydantic model to represent a batch of data to be processed by a `_Step`.

    Attributes:
        seq_no: The sequence number of the batch.
        step_name: The name of the step that will process the batch.
        last_batch: A flag to indicate if the batch is the last one.
        data: The data to be processed.
        data_hash: The hash of the data. Defaults to `None`.
        data_path: The path where the data of the batch is stored. Defaults to `None`.
        accumulated: A flag to indicate if the batch is accumulated.
        created_from: A dictionary containing which batches from which steps were used
            to created this batch. The keys are the names of the steps and the values
            are lists for each step containing the `seq_no` of each batch used, the original         containing the `seq_no` of the batches of the steps that
            size of the batch used and the number of rows used from the batch to create
            this batch.
        size: The size of the batch.
    """

    seq_no: int
    step_name: str
    last_batch: bool
    route_step_last_batch: bool = False
    data: List[List[Dict[str, Any]]] = Field(default_factory=list, repr=False)
    data_hash: Optional[str] = None
    data_path: Optional[str] = None
    accumulated: bool = False
    created_from: Dict[str, List[Tuple[int, int, int]]] = Field(default_factory=dict)
    batch_routed_to: List[str] = Field(default_factory=list)
    size: int = 0
    _num_rows_fs: int | None = PrivateAttr(default=None)
    _fs: Optional[fsspec.AbstractFileSystem] = PrivateAttr(default=None)
    _signature_cache: Optional[str] = PrivateAttr(default=None)
    _signature_cache_valid: bool = PrivateAttr(default=False)
    _logger: "Logger" = PrivateAttr(None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._logger = logging.getLogger(f"distilabel.batch.{self.step_name}.{self.seq_no}")
        # want the signature to only depend on data
        self.exclude_from_signature = (set(_Batch.model_fields.keys()) | {'type_info'}) - {'data'}

    @_timer.time_it
    def invalidate_signature_cache(self) -> None:
        """Invalidates the signature cache.
        
        This should be called whenever the data of the batch is modified.
        """
        self._signature_cache_valid = False

    @property
    def signature(self) -> str:  # type: ignore[override]
        """Return cached signature if available; otherwise compute and cache it.

        The signature for a batch depends only on `data`.
        """
        with _timer.time_block("batch.signature"):
            if self._signature_cache_valid:
                return self._signature_cache
            # compute using parent mixin logic
            self._signature_cache = SignatureMixin.signature.fget(self)
            self._signature_cache_valid = True
            return self._signature_cache

    def next_batch(self) -> "_Batch":
        """Create a new `_Batch` instance with the next batch of data.

        Args:
            data: The data to be processed.

        Returns:
            A `_Batch` instance.
        """
        return _Batch(
            seq_no=self.seq_no + 1, step_name=self.step_name, last_batch=self.last_batch
        )

    @_timer.time_it
    def set_data(self, data: List[List[Dict[str, Any]]]) -> None:
        """Sets the data of the batch and updates the size of the batch.

        Args:
            data: The data of the batch.
        """
        self.data = data
        self.size = len(data[0])
        self.invalidate_signature_cache()

    @_timer.time_it
    def get_data(self, num_rows: Union[int, None] = None) -> List[Dict[str, Any]]:
        """Takes `num_rows` from the data of the batch and returns it. This method will
        also remove the data from the batch and update the hash of the data.

        If the batch is on fs, it will read the data from fs and write back if any rows are remaining.
        num_rows() will be updated.

        Args:
            num_rows: The number of rows to take from the data. If `None`, then all the
                data will be taken. Defaults to `None`.

        Returns:
            A list with the data taken from the batch.
        """

        if self.data == [] and self.data_path is not None:
            pass

        if self.num_rows() == 0:
            return []

        if self.data_path and self._fs and self._num_rows_fs is not None:
            self.read_batch_data_from_fs()

        if num_rows is None:
            data = self.data[0]
            self.data = []
        else:
            data = self.data[0][:num_rows]
            self.data[0] = self.data[0][num_rows:]

        # self.size = len(self.data[0])
        # update signature cache after in-place mutation
        self.invalidate_signature_cache()
        return data

    def num_rows(self) -> int:
        """Returns the number of rows in the batch."""
        if self._num_rows_fs is not None:
            return self._num_rows_fs
        return sum(len(d) for d in self.data)

    @classmethod
    def accumulate(cls, step_name: str, batches: List[List["_Batch"]]) -> "_Batch":
        """Creates a `_Batch` instance using the data from the list of batches that
        were received from another steps. The batches will be accumulated in a single
        list of data.

        Args:
            step_name: The name of the step that will process the batch.
            batches: a list containing the list of batches received from the predecessors.

        Returns:
            A `_Batch` instance.
        """

        data = []
        for step_batches in batches:
            accumulated_data = [row for batch in step_batches for row in batch.data[0]]
            data.append(accumulated_data)
        return cls(
            seq_no=0, step_name=step_name, last_batch=True, data=data, accumulated=True
        )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_Batch` to a dictionary, using the `dataclass` helper function.

        Args:
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            A `dict` containing the internal representation of the `_Batch`.
        """

        include_batch_data = kwargs.get("include_batch_data", True)

        dump = {
            "seq_no": self.seq_no,
            "step_name": self.step_name,
            "last_batch": self.last_batch,
            "data_hash": self.data_hash,
            "accumulated": self.accumulated,
            "created_from": self.created_from,
            "batch_routed_to": self.batch_routed_to,
            "size": self.size,
        }

        if include_batch_data:
            dump["data"] = self.data

        return dump

    def copy(self) -> "_Batch":
        """Creates a copy of the `_Batch` instance.

        Returns:
            A copy of the `_Batch` instance.
        """
        return copy.deepcopy(self)

    @_timer.time_it
    def write_batch_data_to_fs(
        self,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        base_path: Optional[UPath] = None,
    ) -> None:
        """Writes the content of the batch to the filesystem.

        Args
            fs: The `fsspec` filesystem to be used to write the data. If not provided, the
                one set in the `_fs` attribute will be used. Defaults to `None`.
            base_path: The base path where the data of the batch will be stored. If not
                provided, the one set in the `data_path` attribute will be used. Defaults
                to `None`.

        Raises:
            ValueError: If `fs` is not provided and the `_fs` attribute is not set.
        """

        if not fs and not self._fs:
            raise ValueError(
                "The `fs` parameter must be provided if the `_fs` attribute is not set."
            )

        if fs:
            self._fs = fs

        if not base_path and not self.data_path:
            raise ValueError(
                "The `base_path` parameter must be provided if the `data_path` attribute"
                " is not set."
            )

        # when there are multiple successor, base.py _manage_batch_flow makes a copy for each successor
        # these will generate the same file path and then read/write over each other
        # so we generate a unique path for each batch
        seq_no_dir = (
            base_path / f"seq_no_{self.seq_no}_{str(uuid4())}" if base_path else UPath(self.data_path)
        )
        seq_no_dir._fs_cached = self._fs  # type: ignore
        seq_no_dir.mkdir(parents=True, exist_ok=True)

        for i, data in enumerate(self.data):
            table = pa.Table.from_pylist(data)
            with self._fs.open(seq_no_dir / f"data_index_{i}.parquet", "wb") as f:  # type: ignore
                pq.write_table(table, f)

        self._num_rows_fs = self.num_rows()
        self.data = []
        self.data_path = str(seq_no_dir)

    @_timer.time_it
    def read_batch_data_from_fs(self) -> None:
        """Reads the content of the batch from the filesystem."""
        if not self.data_path:
            raise ValueError(
                "`data_path` attribute must be set to read the data from the filesystem."
                " Use `write_batch_data_to_fs` method to set the `data_path` attribute."
            )

        if not self._fs:
            raise ValueError(
                "`_fs` attribute must be set to read the data from the filesystem."
                " Use `write_batch_data_to_fs` method to set the `_fs` attribute."
            )

        for file in self._fs.ls(self.data_path):
            try:
                with self._fs.open(file, "rb") as f:
                    table = pq.read_table(f)
                    self.data.append(table.to_pylist())
            except Exception as e:
                self._logger.warning(
                    f"Error reading batch data from fs {file}: {e} "
                    "falling back to polars, but note that polars can hang in certain multi-processing scenarios. "
                    "These scenarios are rare (e.g. running a pipeline multiple times in a single program), "
                    "so this issue has not been solved yet."
                )
                # pyarrow parquet can get some errors with chunked array outputs with large batches,
                # polars seems to be more robust
                table = pl.read_parquet(file)
                self.data.append(table.to_dicts())

        self._fs.rm(self.data_path, recursive=True)
        self._num_rows_fs = None

    @_timer.time_it
    def routed_to_cached(self, cache_root: Path) -> bool:
        """Check if the batch is cached in the given cache_root."""
        cache_db = get_routed_to_cache_db(cache_root)
        return cache_db.exists(self.signature)

    @_timer.time_it
    def cache_routed_to(self, cache_root: Path) -> None:
        """Cache the field batch_routed_to in the given cache_root."""
        cache_db = get_routed_to_cache_db(cache_root)
        cache_db.set(self.signature, self.batch_routed_to)

    @_timer.time_it
    def load_routed_to(self, cache_root: Path) -> None:
        """Load the field batch_routed_to from cache."""
        cache_db = get_routed_to_cache_db(cache_root)
        if cache_db.exists(self.signature):
            self.batch_routed_to = cache_db.get(self.signature)

    @classmethod
    @_timer.time_it
    def cached(cls, path: Path) -> bool:
        """Check if the batch is cached in the given path."""
        return path.exists()

    @_timer.time_it
    def cache(self, path: Path) -> None:
        """Cache the batch in the given path."""
        self.save(path, format="json")
