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

import traceback
import time
from pathlib import Path
from queue import Queue
from multiprocessing.synchronize import Lock
from typing import Any, Dict, List, Optional, Union, cast

from distilabel.constants import LAST_BATCH_SENT_FLAG
from distilabel.errors import DISTILABEL_DOCS_URL
from distilabel.exceptions import DistilabelOfflineBatchGenerationNotFinishedException
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.pipeline.batch import _Batch
from distilabel.steps.base import GeneratorStep, Step, _Step
from distilabel.typing import StepLoadStatus


class _StepWrapper:
    """Wrapper to run the `Step`.

    Attributes:
        step: The step to run.
        replica: The replica ID assigned.
        input_queue: The queue to receive the input data.
        output_queue: The queue to send the output data.
        load_queue: The queue used to notify the main process that the step has been loaded,
            has been unloaded or has failed to load.
        is_route_step: Whether the step is a route step.
    """

    def __init__(
        self,
        step: Union["Step", "GeneratorStep"],
        replica: int,
        input_queue: "Queue[_Batch]",
        output_queue: "Queue[_Batch]",
        load_queue: "Queue[Union[StepLoadStatus, None]]",
        dry_run: bool = False,
        ray_pipeline: bool = False,
        is_route_step: bool = False,
        cache_location: Path = Path(),
    ) -> None:
        """Initializes the `_ProcessWrapper`.

        Args:
            step: The step to run.
            input_queue: The queue to receive the input data.
            output_queue: The queue to send the output data.
            load_queue: The queue used to notify the main process that the step has been
                loaded, has been unloaded or has failed to load.
            dry_run: Flag to ensure we are forcing to run the last batch.
            ray_pipeline: Whether the step is running a `RayPipeline` or not.
            is_route_step: Whether the step is a route step.
            cache_location: The root directory of the batches cache.
        """
        self.step = step
        self.replica = replica
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.load_queue = load_queue
        self.dry_run = dry_run
        self.ray_pipeline = ray_pipeline
        self.is_route_step = is_route_step
        self._init_cuda_device_placement()
        self._loaded = False
        self._cache_location = cache_location

    def _init_cuda_device_placement(self) -> None:
        """Sets the LLM identifier and the number of desired GPUs of the `CudaDevicePlacementMixin`"""

        def _init_cuda_device_placement_mixin(attr: CudaDevicePlacementMixin) -> None:
            if self.ray_pipeline:
                attr.disable_cuda_device_placement = True
            else:
                desired_num_gpus = self.step.resources.gpus or 1
                attr._llm_identifier = f"{self.step.name}-replica-{self.replica}"
                attr._desired_num_gpus = desired_num_gpus

        for field_name in self.step.model_fields_set:
            attr = getattr(self.step, field_name)
            if isinstance(attr, CudaDevicePlacementMixin):
                _init_cuda_device_placement_mixin(attr)

        if isinstance(self.step, CudaDevicePlacementMixin):
            _init_cuda_device_placement_mixin(self.step)

    def run(self) -> str:
        """The target function executed by the process. This function will also handle
        the step lifecycle. For `GeneratorStep`s it will execute the `load` method first.
        For normal `Step`s, the `load` method will be executed lazily when the first
        batch is received.

        Returns:
            The name of the step that was executed.
        """
        if self.step.is_generator:
            self.load_step()

        self._notify_load()

        try:
            if self.step.is_generator:
                self._generator_step_process_loop()
            else:
                self._non_generator_process_loop()
        except Exception as e:
            if self._loaded:
                self.step.unload()

            # if it's not a load error, we need to notify unload.
            if not (isinstance(e, _StepWrapperException) and e.is_load_error):
                self._notify_unload()

            if not isinstance(e, _StepWrapperException):
                raise _StepWrapperException(str(e), self.step, 2, e) from e
            raise e

        if self._loaded:
            self.step.unload()

        self._notify_unload()

        self.step._logger.info(
            f"🏁 Finished running step '{self.step.name}' (replica ID: {self.replica})"
        )

        return self.step.name  # type: ignore

    def load_step(self) -> None:
        """Loads the step."""
        try:
            self.step.load()
            self._loaded = True
        except Exception as e:
            self.step.unload()
            self._notify_load_failed()
            raise _StepWrapperException.create_load_error(
                message=f"Step load failed: {e}",
                step=self.step,
                subprocess_exception=e,
            ) from e

    def _notify_load(self) -> None:
        """Notifies that the step has finished executing its `load` function successfully."""
        self.step._logger.debug(
            f"Notifying load of step '{self.step.name}' (replica ID {self.replica})..."
        )
        self.load_queue.put({"name": self.step.name, "status": "loaded"})  # type: ignore

    def _notify_unload(self) -> None:
        """Notifies that the step has been unloaded."""
        self.step._logger.debug(
            f"Notifying unload of step '{self.step.name}' (replica ID {self.replica})..."
        )
        self.load_queue.put({"name": self.step.name, "status": "unloaded"})  # type: ignore

    def _notify_load_failed(self) -> None:
        """Notifies that the step failed to load."""
        self.step._logger.debug(
            f"Notifying load failed of step '{self.step.name}' (replica ID {self.replica})..."
        )
        self.load_queue.put({"name": self.step.name, "status": "load_failed"})  # type: ignore

    def cache_key(self, batch: "_Batch") -> Path:
        # the cache maps (batch, step its going to) -> (response)
        # if the same batch goes to the same type of step, 
        # then we expect the same response and load it from the cache
        step_class_name = self.step.__class__.__name__
        return (
            self._cache_location / f'{batch.signature}_{step_class_name}.json'
        )

    def _generator_step_process_loop(self) -> None:
        """Runs the process loop for a generator step. It will call the `process` method
        of the step and send the output data to the `output_queue` and block until the next
        batch request is received (i.e. receiving an empty batch from the `input_queue`).

        If the `last_batch` attribute of the batch is `True`, the loop will stop and the
        process will finish.

        Raises:
            _StepWrapperException: If an error occurs during the execution of the
                `process` method.
        """
        step = cast("GeneratorStep", self.step)

        try:
            batch = self.input_queue.get()
            if batch is None:
                self.step._logger.info(
                    f"🛑 Stopping yielding batches from step '{self.step.name}'"
                )
                return
            offset = batch.seq_no * step.batch_size  # type: ignore

            self.step._logger.info(
                f"🚰 Starting yielding batches from generator step '{self.step.name}'."
                f" Offset: {offset}"
            )
            for data, last_batch in step.process_applying_mappings(offset=offset):
                batch.set_data([data])
                batch.last_batch = self.dry_run or last_batch
                self._send_batch(batch)

                if batch.last_batch:
                    return

                self.step._logger.debug(
                    f"Step '{self.step.name}' waiting for next batch request..."
                )
                batch = self.input_queue.get()
                if batch is None:
                    self.step._logger.info(
                        f"🛑 Stopping yielding batches from step '{self.step.name}'"
                    )
                    return
        except Exception as e:
            raise _StepWrapperException(str(e), self.step, 2, e) from e

    def _non_generator_process_loop(self) -> None:
        """Runs the process loop for a non-generator step. It will call the `process`
        method of the step for each batch received from the input queue.
        """
        step = cast("Step", self.step)
        step._logger.info(f"✨ Starting process loop for step '{step.name}'...")
        while True:
            if self.is_route_step:
                # For route steps, we want to ensure that we have at least two items
                # or the LAST_BATCH_SENT_FLAG before processing a batch. This is to
                # ensure that we can determine if the current batch is the actual
                # last batch for this specific route, as the batch with last_batch = True
                # from the predecessor might go to a different route.
                while True:
                    batch, counts = self.input_queue.pop_if(
                        lambda n_batches_, n_flags_, n_none_: n_batches_ >= 2 or n_flags_ > 0,
                    )
                    if batch is not None:
                        break
                    time.sleep(5)
            else:
                batch, counts = self.input_queue.get_with_counts()

            if batch is None:
                self.step._logger.info(
                    f"🛑 Stopping processing batches from step '{self.step.name}' (replica"
                    f" ID: {self.replica})"
                )
                break
        
            n_batches, n_flags, n_none = counts

            if isinstance(batch, _Batch) and batch.data_path is not None:
                self.step._logger.debug(f"Reading batch data from '{batch.data_path}'")
                batch.read_batch_data_from_fs()

            # get the cache key before any modifications to the received batch
            # so that we can correctly map the sent batch to the response
            cache_key = self.cache_key(batch) if isinstance(batch, _Batch) else Path('none')
            # Since only one of the route steps will receive a batch with last_batch = True, if this step is a route step, 
            # it likely won't receive a batch with last_batch = True and needs to create this
            # itself once it knows which batch is actually the last one for it.
            # Re-inspect the queue after getting the current batch.
            if (
                self.is_route_step
                and batch != LAST_BATCH_SENT_FLAG
                # the next condition checks if all remaining batches are either none or last batch flags
                and (n_flags + n_none) > 0 and n_batches == 0
            ):
                batch.route_step_last_batch = True  # type: ignore

            # handle cache hit
            # but we want route_step_last_batch logic to be handled normally
            # so it comes after that
            if (
                _Batch.cached(cache_key) 
                and not self.step.invalidate_cache
                and self.step.use_cache
            ):
                response = _Batch.from_json(cache_key)
                response.route_step_last_batch = batch.route_step_last_batch
                self._send_batch(response)
                self.step._logger.info(f"🔍 Cache hit for batch {batch.seq_no}")
                if response.last_batch or response.route_step_last_batch:
                    break
                continue

            if batch == LAST_BATCH_SENT_FLAG:
                self.step._logger.debug("Received `LAST_BATCH_SENT_FLAG`. Stopping...")
                break

            self.step._logger.info(
                f"📦 Processing batch {batch.seq_no} in '{batch.step_name}' (replica ID: {self.replica})"
            )
            # lazy loading of the step, if we end up with all cache hits,
            # we don't need to load the step
            if not self._loaded:
                self.load_step()

            result = []
            try:
                if self.step.has_multiple_inputs:
                    result = next(step.process_applying_mappings(*batch.data))
                else:
                    result = next(step.process_applying_mappings(batch.data[0]))
            except Exception as e:
                if self.step.is_global:
                    self.step.unload()
                    self._notify_unload()
                    data = (
                        batch.data
                        if isinstance(
                            e, DistilabelOfflineBatchGenerationNotFinishedException
                        )
                        else None
                    )
                    raise _StepWrapperException(str(e), self.step, 2, e, data) from e

                # Impute step outputs columns with `None`
                result = self._impute_step_outputs(batch)

                # if the step is not global then we can skip the batch which means sending
                # an empty batch to the output queue
                self.step._logger.warning(
                    f"⚠️ Processing batch {batch.seq_no} with step '{self.step.name}' failed."
                    " Sending empty batch filled with `None`s..."
                )
                self.step._logger.warning(
                    f"Subprocess traceback:\n\n{traceback.format_exc()}"
                )
            finally:
                batch.set_data([result])
                if self.step.use_cache:
                    batch.cache(cache_key)
                self._send_batch(batch)

            if batch.last_batch or batch.route_step_last_batch:
                break

    def _impute_step_outputs(self, batch: "_Batch") -> List[Dict[str, Any]]:
        """Imputes the step outputs columns with `None` in the batch data.

        Args:
            batch: The batch to impute.
        """
        return self.step.impute_step_outputs(batch.data[0])

    def _send_batch(self, batch: _Batch) -> None:
        """Sends a batch to the `output_queue`."""
        if batch.data_path is not None:
            self.step._logger.debug(f"Writing batch data to '{batch.data_path}'")
            batch.write_batch_data_to_fs()

        self.step._logger.info(
            f"📨 Step '{batch.step_name}' sending batch {batch.seq_no} to output queue"
        )
        self.output_queue.put(batch)


class _StepWrapperException(Exception):
    """Exception to be raised when an error occurs in the `_StepWrapper` class.

    Attributes:
        message: The error message.
        step: The `Step` that raised the error.
        code: The error code.
        subprocess_exception: The exception raised by the subprocess.
        data: The data that caused the error. Defaults to `None`.
    """

    def __init__(
        self,
        message: str,
        step: "_Step",
        code: int,
        subprocess_exception: Exception,
        data: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        self.message = f"{message}\n\nFor further information visit '{DISTILABEL_DOCS_URL}api/pipeline/step_wrapper'"
        self.step = step
        self.code = code
        self.subprocess_exception = subprocess_exception
        self.formatted_traceback = "".join(
            traceback.format_exception(
                type(subprocess_exception),
                subprocess_exception,
                subprocess_exception.__traceback__,
            )
        )
        self.data = data

    @classmethod
    def create_load_error(
        cls,
        message: str,
        step: "_Step",
        subprocess_exception: Optional[Exception] = None,
    ) -> "_StepWrapperException":
        """Creates a `_StepWrapperException` for a load error.

        Args:
            message: The error message.
            step: The `Step` that raised the error.
            subprocess_exception: The exception raised by the subprocess. Defaults to `None`.

        Returns:
            The `_StepWrapperException` instance.
        """
        return cls(message, step, 1, subprocess_exception, None)

    @property
    def is_load_error(self) -> bool:
        """Whether the error is a load error.

        Returns:
            `True` if the error is a load error, `False` otherwise.
        """
        return self.code == 1
