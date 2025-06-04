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

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, Set

from distilabel.constants import (
    RECEIVES_ROUTED_BATCHES_ATTR_NAME,
    STEP_ATTR_NAME,
    LAST_BATCH_ROUTED_FLAG,
)
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.batch import _Batch
from distilabel.steps.base import _Step
from distilabel.utils.files import list_files_in_dir
from distilabel.utils.serialization import (
    StrOrPath,
    _check_is_dir,
    _Serializable,
    read_json,
)

if TYPE_CHECKING:
    from distilabel.utils.serialization import StrOrPath


@dataclass
class _BatchManagerStep(_Serializable):
    """A class that will accumulate data for a step from the predecessors and create
    batches for the step to process when there is enough data.

    Attributes:
        step_name: The name of the step that will process the data.
        accumulate: A flag to indicate if the data should be accumulated and create a
            batch with all the data received from the predecessors instead of creating
            batches with the `input_batch_size`.
        route_step: A flag to indicate if the step is a route step (receives routed batches).
        input_batch_size: The size of the batch to be created for the step to process.
            If `None`, then `accumulate` must be `True`. Defaults to `None`.
        data: A dictionary with the predecessor step name as the key and a list of
            dictionaries (rows) as the value.
        built_batches: A list with the batches that were built and sent to the step queue,
            but the step was stopped before processing the batch, so the batch doesn't get
            lost. Defaults to an empty list.
        seq_no: The sequence number of the next batch to be created. It will be
            incremented for each batch created.
        last_batch_received: A list with the names of the steps that sent the last
            batch of data.
        convergence_step: A flag to indicate if the step is a convergence step. An
            `Step` is a convergence step if all its predecessors are receiving routed
            batches. Defaults to `False`.
        convergence_step_batches_consumed: A dictionary in which the key is the `seq_no`
            of the batch created by step A, that was used by step B and C and obtained from
            the `created_from` of the batches created by them. It's used to know if all
            the batches from B and C steps created from batches of A have been consumed
            by D, in order to not mess up the order of the batches. Only used if `convergence_step=True`.
            Defaults to an empty dictionary.
        next_expected_created_from_batch_seq_no: The next expected sequence number of the
            batch from step A used by steps B and C and obtained from the `created_from`
            of the batches created by them. It's used to avoid messing up the order of the
            batches. Only used if `convergence_step=True`. Defaults to `0`.
        step_signature: The signature that defines a given `Step`. It will be used for the
            caching mechanism.
        use_cache: Flag from the original `Step` to indicate whether this step should make use of
            the cached data.
        step_offset: Dictionary with each key the predecessor/s step/s and as value a dict
            with keys `batch` and `offset`, containing the name of the file for the corresponding
            batch, and the number of rows that were read from that step, respectively. Used
            for caching mechanism.
    """

    step_name: str
    accumulate: bool
    input_batch_size: Union[int, None] = None
    data: Dict[str, List[_Batch]] = field(default_factory=dict)
    built_batches: List[_Batch] = field(default_factory=list)
    seq_no: int = 0
    last_batch_received: List[str] = field(default_factory=list)
    last_batch_routed: bool = False
    convergence_step: bool = False
    convergence_step_batches_consumed: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    convergence_step_n_batches_received: Dict[str, int] = field(default_factory=dict)
    convergence_step_n_batches_to_receive: Dict[str, int] = field(default_factory=dict)
    convergence_step_receives_from: Dict[str, bool] = field(default_factory=dict)
    next_expected_created_from_batch_seq_no: int = 0
    next_expected_seq_no: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    global_next_seq_no: int = 0
    global_received_seq_nos: List[int] = field(default_factory=list)
    step_signature: Optional[str] = None
    use_cache: bool = False
    step_offset: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    _all_added_batches: List[_Batch] = field(default_factory=list)

    def add_batch(self, batch: _Batch, prepend: bool = False) -> None:
        """Add a batch of data from `batch.step_name` to the step. It will accumulate the
        data and keep track of the last batch received from the predecessors.

        Args:
            batch: The output batch of an step to be processed by the step.
            prepend: If `True`, the content of the batch will be added to the `built_batches`
                list. This is done so if a `_Batch` was already built and send to the step
                queue, and the step is stopped before processing the batch, the batch doesn't
                get lost. Defaults to `False`.
        """
        from copy import deepcopy
        self._all_added_batches.append(deepcopy(batch))
        from_step = batch.step_name

        if prepend:
            self.built_batches.append(batch)
        else:
            self.data[from_step].append(batch)
            self.data[from_step].sort(key=lambda batch: batch.seq_no)

        if batch.last_batch:
            self.last_batch_received.append(from_step)

        if self.convergence_step:
            if from_step not in self.convergence_step_n_batches_received:
                self.convergence_step_n_batches_received[from_step] = 0
            self.convergence_step_n_batches_received[from_step] += 1

    def _all_batches_received_convergence_step(self) -> bool:
        """Checks if all the batches have been received for the convergence step.

        Returns:
            `True` if all the batches have been received, `False` otherwise.    
        """
        if LAST_BATCH_ROUTED_FLAG not in self.convergence_step_receives_from:
            return False
        receives_from = [
            step for step in self.convergence_step_receives_from.keys() 
            if step != LAST_BATCH_ROUTED_FLAG
        ]
        return (
            all(
                route_step in self.convergence_step_n_batches_to_receive 
                for route_step in receives_from
            )
            and sum(
                self.convergence_step_n_batches_to_receive.values()
            ) == sum(
                self.convergence_step_n_batches_received.values()
            )
        )

    def get_batch(self) -> Union[_Batch, None]:
        """Create a new batch of data for the step to process. It will return `None` if
        there is not enough data to create a batch.

        Returns:
            A `_Batch` instance if there is enough data to create a batch. Otherwise,
            `None`.
        """
        # If there are batches in the `built_batches` list, then return the first one
        # and remove it from the list.
        if self.built_batches:
            return self.built_batches.pop(0)

        if not self._ready_to_create_batch():
            return None

        seq_no = self._get_seq_no()

        # `_last_batch` must be called before `_get_data`, as `_get_data` will update the
        # list of data which is used to determine if the batch to be created is the last one.
        last_batch = self._last_batch()

        # Get the batch data and the information from which batches of the upstream steps
        # the data was taken.
        data, created_from, batch_routed_to = self._get_data()

        # Update the step offset i.e. which is the last batch and last row index from that
        # batch that the step has consumed
        self._update_offset(created_from)

        return _Batch(
            seq_no=seq_no,
            step_name=self.step_name,
            last_batch=last_batch,
            data=data,
            accumulated=self.accumulate,
            created_from=created_from,
            batch_routed_to=batch_routed_to,
        )

    def empty_buffers(self) -> List[str]:
        """Checks if the input buffer for the step is empty.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        if self.accumulate:
            return [
                previous_step
                for previous_step in self.data.keys()
                if previous_step not in self.last_batch_received
            ]

        return [
            previous_step
            for previous_step, batches in self.data.items()
            if previous_step not in self.last_batch_received
            and sum(len(batch.data[0]) for batch in batches) < self.input_batch_size  # type: ignore
        ]

    def set_next_expected_seq_no(
        self, from_step: str, next_expected_seq_no: int
    ) -> None:
        """Sets the next expected sequence number of a `_Batch` received by the step coming
        from `from_step`.

        Args:
            from_step: The name of the step from which its next expected sequence number
                in step has to be updated.
            next_expected_seq_no: the next expected sequence number of a `_Batch` coming
                from `from_step`.
        """

        if not self.data[from_step] or (
            self.data[from_step]
            and self.data[from_step][0].seq_no >= next_expected_seq_no
        ):
            self.next_expected_seq_no[from_step] = (
                next_expected_seq_no,
                next_expected_seq_no,
            )
        else:
            self.next_expected_seq_no[from_step] = (
                self.next_expected_seq_no[from_step][0],
                next_expected_seq_no,
            )

    def set_global_next_seq_no(self, received_seq_no: int) -> None:
        """Sets a shared next expected sequence number for the successors of a step.
        This tracks the seq_no for which all batches with lower seq_no have been received.
        """
        self.global_received_seq_nos.append(received_seq_no)
        # maximum of 1 past what has been received
        for i in range(max(self.global_received_seq_nos) + 2):
            if i not in self.global_received_seq_nos:
                self.global_next_seq_no = i
                break

    def notify_route_step_of_last_batch(self) -> None:
        """Sets last_batch_routed flag to True (assumes this is a route step)."""
        self.last_batch_routed = True

    @classmethod
    def from_step(
        cls, step: "_Step", predecessors: Iterable[str], convergence_step: bool = False
    ) -> "_BatchManagerStep":
        """Creates a `_BatchManagerStep` instance from a `_Step` instance and its
        predecessors.

        Args:
            step: The `_Step` instance.
            predecessors: The names of the predecessors of the step.
            convergence_step: A flag to indicate if the step is a convergence step. An
                `Step` is a convergence step if all its predecessors are receiving routed
                batches. Defaults to `False`.

        Returns:
            A `_BatchManagerStep` instance.
        """
        return cls(
            step_name=step.name,  # type: ignore
            accumulate=step.is_global,
            input_batch_size=getattr(step, "input_batch_size", None),
            data={predecessor: [] for predecessor in predecessors},
            convergence_step=convergence_step,
            next_expected_seq_no={predecessor: (0, 0) for predecessor in predecessors},
            step_signature=step.signature,
            use_cache=step.use_cache,
            step_offset={predecessor: (0, 0) for predecessor in predecessors},
        )

    def _get_seq_no(self) -> int:
        """Gets the sequence number for the next batch to be created and increments it.

        Returns:
            The sequence number for the next batch to be created.
        """
        seq_no = self.seq_no
        self.seq_no += 1
        return seq_no

    def _get_data(
        self,
    ) -> Tuple[
        List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int, int]]], List[str]
    ]:
        """Gets the data needed to create a batch for the step to process. If the step is
        accumulating data, then it will return a list with all the data received from the
        predecessors. Otherwise, it will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the
        batch from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a normal step.
        """
        if self.accumulate:
            # Steps accumulating cannot receive routed batches
            return self._get_data_for_accumulate() + ([],)

        if self.convergence_step:
            # Convergence steps will receive routed batches, but we need to clean the
            # `batch_routed_to` list
            return self._get_data_for_convergence_step() + ([],)

        return self._get_data_normal()

    def _get_data_for_accumulate(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step
        is accumulating data (a global step). It will return a list with all the data received from the
        predecessors. In addition, it will remove the data used to create the batch from
        the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        data = []
        batches_used = defaultdict(list)
        for step_name, batches in self.data.items():
            for batch in batches:
                batches_used[step_name].append((batch.seq_no, batch.size, batch.size))
            data.append([row for batch in batches for row in batch.get_data()])
        # Reset the data buffer
        self.data = {step_name: [] for step_name in self.data}
        return data, dict(batches_used)

    def _get_data_for_convergence_step(
        self,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int, int]]]]:
        """Gets the data needed to create a batch for the step to process when the step is
        a convergence step.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process and a dictionary with the sequence numbers of the batches that were
            used to create the batch.
        """
        grouped_batches = self._group_batches_by_created_from()
        
        n_selected_rows = 0
        selected_data = defaultdict(list)
        batches_used = defaultdict(list)
        for seq_no, batches in grouped_batches:
            remaining_rows = 0
            for batch, _ in batches:
                if n_selected_rows != self.input_batch_size:
                    batch_data = batch.get_data(self.input_batch_size - n_selected_rows)
                    selected_data[batch.step_name].extend(batch_data)
                    n_selected_rows += len(batch_data)

                    # Keep track of the batches used to create the batch
                    batches_used[batch.step_name].append((batch.seq_no, batch.size, len(batch_data)))

                remaining_rows += batch.num_rows()

                if batch.num_rows() == 0:
                    self.data[batch.step_name].remove(batch)

            if n_selected_rows == self.input_batch_size:
                break

        return list(selected_data.values()), dict(batches_used)

    def _get_data_normal(
        self,
    ) -> Tuple[
        List[List[Dict[str, Any]]], Dict[str, List[Tuple[int, int, int]]], List[str]
    ]:
        """Gets the data needed to create a batch for the step to process when the step is
        not accumulating data. It will return a list of data with the `input_batch_size`
        for each predecessor. In addition, it will remove the data used to create the batch
        from the step's data.

        Returns:
            A tuple containing the list of data needed to create a batch for the step to
            process, a dictionary with the sequence numbers of the batches that were used
            to create the batch and the list of steps to which the batch was routed to if
            the step is a convergence step.
        """
        n_selected_rows = 0
        data = []  # list[list[rows from a step]]
        batches_used = defaultdict(list)  # dict[step_name, tuple[batch_seq_no, batch_size, num_rows_taken_from_batch]]
        batch_routed_to = []  # list[step_name] 
        # (from the last batch used to create the batch)
        # should only be different from [] when the step is a route step (or convergence step, but conv step has a different function)
        # and it should be the same for each batch used since a route step should only have one predecessor
        for step_name in self.data:
            if len(self.data[step_name]) == 0 or n_selected_rows == self.input_batch_size:
                continue
            
            # Pull batches from this step, accumulating into step_data, and data.append(step_data) at the end
            # continues to the next step until we have enough rows or we run out of batches
            step_data = []
            idx_drop_batches = []
            next_expected_seq_no = None
            for idx, batch in enumerate(self.data[step_name]):
                if n_selected_rows != self.input_batch_size:
                    # pull data from batch 
                    batch_data = batch.get_data(self.input_batch_size - n_selected_rows)
                    step_data.extend(batch_data)
                    n_selected_rows += len(batch_data)

                    # keep track of which batches were used to create the batch
                    batch_routed_to = batch.batch_routed_to
                    batches_used[step_name].append((batch.seq_no, batch.size, len(batch_data)))

                    # tracks the minimum seq_no we can take from, gets += 1 if we took everything from this batch
                    next_expected_seq_no = batch.seq_no
                    if batch.num_rows() == 0:
                        idx_drop_batches.append(idx)
                        next_expected_seq_no += 1
                else: 
                    break

            # Remove the batches that were entirely consumed
            idx_drop_batches.reverse()
            for idx in idx_drop_batches:
                self.data[step_name].pop(idx)

            # Update the `next_expected_seq_no` from `step_name`. It can happen that:
            # 1. This step didn't receive one batch because it was routed to other batches
            # and `set_next_expected_seq_no` method was called. If the first element
            # is not equal to the second, that means there is a potential `next_expected_seq_no`
            # from `step_name`. If there is no data left, then we set that as the `next_expected_seq_no`.
            # 2. `set_next_expected_seq_no` has not been called, so we set the `next_expected_seq_no`
            # taking into account the data left in the step.
            step_next_expected_seq_no = self.next_expected_seq_no[step_name]
            if step_next_expected_seq_no[0] != step_next_expected_seq_no[1] and (
                not self.data[step_name]
                or self.data[step_name][0].seq_no >= step_next_expected_seq_no[1]
            ):
                self.next_expected_seq_no[step_name] = (
                    step_next_expected_seq_no[1],
                    step_next_expected_seq_no[1],
                )
            elif next_expected_seq_no:
                self.next_expected_seq_no[step_name] = (
                    next_expected_seq_no,
                    max(next_expected_seq_no, step_next_expected_seq_no[1]),
                )

            data.append(step_data)

        return data, dict(batches_used), batch_routed_to

    def _ready_to_create_batch(self) -> bool:
        """Checks if there is enough data to create a batch for the step.

        Returns:
            `True` if there is enough data to create a batch for the step. Otherwise,
            `False`.
        """

        if self.accumulate:
            return self._ready_to_create_batch_accumulate()

        if self.convergence_step:
            return self._ready_to_create_batch_convergence_step()

        return self._ready_to_create_batch_normal()

    def _ready_to_create_batch_accumulate(self) -> bool:
        """Checks if there is enough data for an step accumulating data. It will return
        `True` if the last batch was received from all the predecessors.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        total_rows = sum(
            sum(batch.num_rows() for batch in batches) 
            for batches in self.data.values()
        )
        if total_rows == 0: 
            return False
        
        if self.convergence_step:
            return self._all_batches_received_convergence_step()
    
        return all(
            step in self.last_batch_received
            for step in self.data.keys()
        )
        
    def _ready_to_create_batch_convergence_step(self) -> bool:
        """Checks if there is enough data for creating a batch for an step in which output
        batches that were generated by steps that received routed batches are received.
        It will return `True`, if all the output batches that were generated from a routed
        batch have been received.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        # no longer considering order because we check that all batches are received, so things are 
        # in the right order and all contiguous
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches or not self._all_batches_received_convergence_step():
            # we don't want to draw from sequences out of order, but checking if we have received 
            # all batches from a certain seq number (and can therefore draw the last batch from it)
            # is quite complex, so we just wait for all batches to be received
            return False
        available_rows = 0
        includes_last_batch = False
        for seq_no, batches in grouped_batches:
            if any(batch.last_batch for batch, _ in batches):
                includes_last_batch = True
            available_rows += sum(batch.num_rows() for batch, _ in batches)
            if available_rows >= self.input_batch_size:
                return True

        if available_rows > 0 and includes_last_batch:
            return True
        return False

    def _ready_to_create_batch_normal(self) -> bool:
        """Checks if there is enough data for creating a batch for a normal step. It will
        be `True` it there are at least `input_batch_size` rows from a predecessor or 
        the last batch from that predecessor has been received. It will pull from predecessors
        in the order they appear in self.data and wait until a predecessor is done to 
        move on to the next one to maintain an order.

        Returns:
            `True` if ready to create a batch, `False` otherwise.
        """
        if sum(len(batches) for batches in self.data.values()) == 0:
            return False
        elif self._last_batch() or self.last_batch_routed:
            return True
        
        available_rows = 0
        for step_name, batches in self.data.items():
            # Depending on the number of replicas of the `Step` it can happen that some
            # replica is faster and send batch with `seq_no==1` faster than the other that
            # sends the batch with `seq_no==0`. We need to check which `seq_no` was expected
            # next to not mess up the ordering of the rows.
            next_expected_seq_nos = self.next_expected_seq_no[step_name]

            # `batches` are sorted by `seq_no`
            for batch in batches:
                # Need to create batches using the data from batches with sequential `seq_no`
                # The reason for the change to <= <= is that when a batch is created out of part of a received batch (say 2)
                # and there is some leftover, then the next expected seq no[0] will remain pointing at that batch (2)
                # then this step, when it is a route step, will receive some non-sequential batch next (say 7),
                # and the previous logic would check rows from 2 then look for a batch with seq_no 3 (cause it just +1'd)
                # and never make it past this. The logic works in a simpler scenario where there aren't leftover rows
                # in batches of route steps, because then the next expected seq no[0] gets incremented in _manage_batch_flow
                # for all steps not routed to and it would be at 7 when it received batch 7.
                if not (next_expected_seq_nos[0] <= batch.seq_no <= self.global_next_seq_no):
                    return False
                available_rows += batch.num_rows()
                if self.input_batch_size and available_rows >= self.input_batch_size:
                    return True

        return False

    def _last_batch(self) -> bool:
        """Checks if the batch to be created is the last one i.e. if the last batch was
        received from all the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        if self.accumulate:
            return self._last_batch_accumulate()

        if self.convergence_step:
            return self._last_batch_convergence_step()

        return self._last_batch_normal()

    def _update_offset(
        self, created_from: Dict[str, List[Tuple[int, int, int]]]
    ) -> None:
        """Update the offset for the batch buffers of the upstream steps.

        Args:
            created_from: A dictionary containing which batches from which steps were used
                to created this batch. The keys are the names of the steps and the values
                are lists for each step containing the `seq_no` of each batch used, the original         containing the `seq_no` of the batches of the steps that
                size of the batch used and the number of rows used from the batch to create
                this batch.
        """
        for predecessor, seq_no_and_batch in created_from.items():
            prev_last_batch_seq_no, prev_last_batch_offset = self.step_offset[
                predecessor
            ]
            last_batch_seq_no, _, last_batch_size = seq_no_and_batch[-1]
            batch_offset = (
                prev_last_batch_offset + last_batch_size
                if prev_last_batch_seq_no == last_batch_seq_no
                else last_batch_size
            )
            last_batch_seq_no = (
                last_batch_seq_no
                if last_batch_seq_no > prev_last_batch_seq_no
                else prev_last_batch_seq_no
            )
            self.step_offset[predecessor] = (last_batch_seq_no, batch_offset)

    def _last_batch_accumulate(self) -> bool:
        """Checks if the batch to be created is the last one for an step accumulating data.
        Since a global step/accumulate step only produces one batch, we just need to check
        that we are ready to make that batch.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        return self._ready_to_create_batch()

    def _last_batch_convergence_step(self) -> bool:
        """Checks if the batch to be created is the last one for a convergence step. `True`
        if the last batch of all the steps (`batch_routed_to`) in the last routed batch
        have been received.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        # since these are sorted by created_from seq no, last batch should only be true for the last one
        grouped_batches = self._group_batches_by_created_from()
        if not grouped_batches:
            return False

        if not self._all_batches_received_convergence_step():
            return False

        available_rows = 0
        for _, batches in grouped_batches:
            available_rows += sum(batch.num_rows() for batch, _ in batches)

        return available_rows <= self.input_batch_size

    def _last_batch_normal(self) -> bool:
        """Checks if the batch to be created is the last one for a normal step. `True` if
        there is no more data to be received from the predecessors.

        Returns:
            `True` if the batch to be created is the last one. Otherwise, `False`.
        """
        includes_last_batch = False
        available_rows = 0
        for step_name, batches in self.data.items():
            if step_name not in self.last_batch_received:
                return False

            available_rows += sum(batch.num_rows() for batch in batches)
            if any(batch.last_batch for batch in batches):
                includes_last_batch = True

            if self.input_batch_size and available_rows > self.input_batch_size:
                return False

        return includes_last_batch

    def _group_batches_by_created_from_dict(
        self,
    ) -> Dict[int, List[Tuple["_Batch", int]]]:
        """Group the batches by the first key of `created_from` field. This method is
        meant to be used only with a `convergence_step`.

        Returns:
            A list of the batches grouped by the `seq_no` of the first step name in `created_from`.
            The list is sorted by the `seq_no`.
        """
        grouped_batches: Dict[int, List[Tuple["_Batch", int]]] = defaultdict(list)
        for batches in self.data.values():
            for batch in batches:
                first_key = next(iter(batch.created_from))
                # must be the max (at least for when checking if this is the last batch)
                # otherwise the the received last batch could be made of batches from different seq_no
                # where one is low and another is high, and if you took the first idx in created_from, 
                # the received last batch would be used early on (since it is sorted by seq_no), 
                # then it would mess up things that check if a batch is last_batch
                batch_seq_no = max(created_from[0] for created_from in batch.created_from[first_key])
                _, batch_size, _ = batch.created_from[first_key][-1]
                grouped_batches[batch_seq_no].append((batch, batch_size))
        return grouped_batches

    def _group_batches_by_created_from(
        self,
    ) -> List[Tuple[int, List[Tuple["_Batch", int]]]]:
        """Group the batches by the first key of `created_from` field. This method is
        meant to be used only with a `convergence_step`.

        Returns:
            A list of the batches grouped by the `seq_no` of the first step name in `created_from`.
            The list is sorted by the `seq_no`.
        """
        '''
        any entry[1] in grouped_batches is a list of output batches from steps a single batch was routed to
            i.e. entry[1] is a list of outputs for duplicate inputs, of which, 
            there are len(entry[1][0].batch_routed_to) batches in total in the pipeline
        '''
        grouped_batches = self._group_batches_by_created_from_dict()
        return sorted((seq_no, batches) for seq_no, batches in grouped_batches.items())

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManagerStep` to a dictionary.

        Args:
            obj: Unused, just kept to match the signature of the parent method.
            kwargs: Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Internal representation of the `_BatchManagerStep`.
        """
        return {
            "step_name": self.step_name,
            "accumulate": self.accumulate,
            "input_batch_size": self.input_batch_size,
            "data": {
                step_name: [batch.dump(**kwargs) for batch in batches]
                for step_name, batches in self.data.items()
            },
            "built_batches": [batch.dump(**kwargs) for batch in self.built_batches],
            "seq_no": self.seq_no,
            "last_batch_received": self.last_batch_received,
            "last_batch_routed": self.last_batch_routed,
            "convergence_step": self.convergence_step,
            "convergence_step_batches_consumed": self.convergence_step_batches_consumed,
            "convergence_step_n_batches_to_receive": self.convergence_step_n_batches_to_receive,
            "convergence_step_n_batches_received": self.convergence_step_n_batches_received,
            "convergence_step_receives_from": self.convergence_step_receives_from,
            "next_expected_created_from_batch_seq_no": self.next_expected_created_from_batch_seq_no,
            "next_expected_seq_no": self.next_expected_seq_no,
            "global_next_seq_no": self.global_next_seq_no,
            "global_received_seq_nos": self.global_received_seq_nos,
            "step_signature": self.step_signature,
            "use_cache": self.use_cache,
            "step_offset": self.step_offset,
        }

    @property
    def signature(self) -> str:
        return f"{self.step_name}_{self.step_signature}"


class _BatchManager(_Serializable):
    """Class to manage the batches received from the steps. It keeps track of the
    received batches and returns new batches for the steps to process based on their
    input batch size and the batches received from the predecessors.

    Attributes:
        steps: A dictionary with the step name as the key and a `_BatchManagerStep`
            instance as the value.
        last_batch_received: A dictionary with the step name as the key and a flag to
            indicate whether we received the last batch from the step.
    """

    def __init__(
        self,
        steps: Dict[str, _BatchManagerStep],
        last_batch_received: Dict[str, Union[_Batch, None]],
        last_batch_sent: Dict[str, Union[_Batch, None]],
        last_batch_flag_sent_to: List[str],
        received_batch_seq_nos: Dict[str, List[int]],
    ) -> None:
        """Initialize the `_BatchManager` instance.

        Args:
            steps: A dictionary with the step name as the key and a dictionary with the
                predecessor step name as the key and a list of batches as the value.
            last_batch_received: A dictionary with the step name as the key and the last
                `_Batch` received from the step.
            last_batch_sent: A dictionary with the step name as the key and the last
                `_Batch` sent to the step.
            last_batch_flag_sent_to: A list with the names of the steps to which `LAST_BATCH_SENT_FLAG`
                was sent.
            received_batch_seq_nos: a dictionary containing the list of batches sequence
                numbers received per step.
        """

        self._steps = steps
        self._last_batch_received = last_batch_received
        self._last_batch_sent = last_batch_sent
        self._last_batch_flag_sent_to = last_batch_flag_sent_to
        self._received_batch_seq_nos = received_batch_seq_nos

    def _missing_seq_no(self, last_batch: _Batch) -> bool:
        """Checks if there's any missing sequence number in the batches received from the
        step.

        Args:
            last_batch: the batch with `last_batch==True` received from the step.

        Returns:
            `True` if there's any missing sequence number, `False` otherwise.
        """
        received_batch_seq_nos = self._received_batch_seq_nos[last_batch.step_name]
        for i in range(last_batch.seq_no + 1):
            if i not in received_batch_seq_nos:
                return True
        return False

    def can_generate(self) -> bool:
        """Checks if there are still batches to be processed by the steps.

        Returns:
            `True` if there are still batches to be processed by the steps. Otherwise,
            `False`.
        """
        for step_name, batch in self._last_batch_received.items():
            if (  # exemptions from the following list of conditions
                # a route step that received a last batch flag
                # (not all route steps will send a last batch to be added to _last_batch_received)
                step_name in self._last_batch_flag_sent_to
            ):
                continue

            if not batch:
                return True

            if batch.last_batch and self._missing_seq_no(batch):
                return True

            if not batch.last_batch:
                return True

            if not self.get_last_batch_sent(step_name):
                return True

        return False

    def register_batch(
        self, batch: _Batch, steps_data_path: Optional["StrOrPath"] = None
    ) -> None:
        """Method to register a batch received from a step. It will keep track of the
        sequence number and the last batch received from the step in the internal maps.

        Args:
            batch: _Batch from which we will register the sequence number and the last batch received.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.
        """
        step_name = batch.step_name
        seq_no = batch.seq_no
        self._received_batch_seq_nos[step_name].append(seq_no)

        last_batch = self._last_batch_received[step_name]
        if not last_batch or (last_batch and last_batch.seq_no < seq_no):
            self._last_batch_received[step_name] = batch

        if steps_data_path:
            self.write_batch_data(batch, steps_data_path)

    def write_batch_data(self, batch: _Batch, steps_data_path: Path) -> None:
        """Writes the batch to the steps data directory.

        Argument:
            batch: the batch to be written.
            steps_data_path: the steps data base directory.
        """
        step = self._steps[batch.step_name]
        batch_manager_data_dir = Path(steps_data_path) / step.signature
        batch_manager_data_dir.mkdir(parents=True, exist_ok=True)
        filename = batch_manager_data_dir / f"batch_{batch.seq_no}.json"
        if not filename.exists():
            self.save(path=filename, format="json", dump=batch.dump())

    def get_last_batch(self, step_name: str) -> Union[_Batch, None]:
        """Gets the last batch received from a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch received from the step or `None` if no batch was received.
        """
        return self._last_batch_received.get(step_name)

    def add_batch(
        self,
        to_step: str,
        batch: _Batch,
        prepend: bool = False,
    ) -> None:
        """Add an output batch from `batch.step_name` to `to_step`.

        Args:
            to_step: The name of the step that will process the batch.
            batch: The output batch of an step to be processed by `to_step`.
            prepend: If `True`, the content of the batch will be added at the start of
                the buffer.

        Raises:
            ValueError: If `to_step` is not found in the batch manager.
        """
        if to_step not in self._steps:
            raise ValueError(f"Step '{to_step}' not found in the batch manager.")
        step = self._steps[to_step]
        step.add_batch(batch, prepend)

    def add_batch_to_recover_offline_batch_generation(
        self, to_step: str, data: List[List[Dict[str, Any]]]
    ) -> None:
        """Add a batch to recover pipeline execution from an `_Step` that used an `LLM`
        with offline batch generation. It will add the batch to the start of the buffer
        of the step  and set the last batch received of the step to `None`.

        Args:
            to_step: The name of the step that will process the batch.
            data: The data that was used with the offline batch generation.
        """
        self.add_batch(
            to_step=to_step,
            batch=_Batch(seq_no=0, step_name=to_step, last_batch=True, data=data),
            prepend=True,
        )
        self._last_batch_received[to_step] = None

    def get_batch(self, step_name: str) -> Union[_Batch, None]:
        """Get the next batch to be processed by the step.

        Args:
            step_name: The name of the step that will process the batch.

        Returns:
            A `_Batch` instance if there is a batch to be processed by the step. Otherwise,
            `None`.
        """
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found in the batch manager.")

        return self._steps[step_name].get_batch()

    def step_empty_buffers(self, step_name: str) -> List[str]:
        """Checks if the input buffer for a step is empty.

        Args:
            step_name: The name of the step.

        Returns:
            The name of the previous steps for which the input buffer for this step is
            empty.
        """
        return self._steps[step_name].empty_buffers()

    def set_last_batch_sent(self, batch: "_Batch") -> None:
        """Set the last batch sent to a step.

        Args:
            batch: The last batch sent to a step.
        """
        self._last_batch_sent[batch.step_name] = batch

    def get_last_batch_sent(self, step_name: str) -> Union["_Batch", None]:
        """Get the last batch sent to a step.

        Args:
            step_name: The name of the step.

        Returns:
            The last batch sent to a step or `None` if no batch was sent.
        """
        return self._last_batch_sent.get(step_name, None)

    def set_last_batch_flag_sent_to(self, step_name: str) -> None:
        """Set the flag to indicate that the last batch was sent to a step.

        Args:
            step_name: The name of the step.
        """
        self._last_batch_flag_sent_to.append(step_name)

    def set_next_expected_seq_no(
        self, step_name: str, from_step: str, next_expected_seq_no: int
    ) -> None:
        """Sets the next expected sequence number of a `_Batch` received by `step` coming
        from `from_step`.

        Args:
            step_name: The step name whose next expected sequence number for `from_step`
                has to be updated.
            from_step: The name of the step from which its next expected sequence number
                in step has to be updated.
            next_expected_seq_no: the next expected sequence number of a `_Batch` coming
                from `from_step`.
        """
        self._steps[step_name].set_next_expected_seq_no(from_step, next_expected_seq_no)

    def set_global_next_seq_no(self, step_name: str, received_seq_no: int) -> None:
        """Sets the global next sequence number for a step.

        Args:
            step_name: The name of the successor step.
            received_seq_no: The sequence number of the last batch received.
        """
        self._steps[step_name].set_global_next_seq_no(received_seq_no)

    def set_n_batches_to_receive(self, conv_step: str, route_step: str, n_batches: int) -> None:
        """Set the number of batches to receive for a (presumed convergence) step."""
        self._steps[conv_step].convergence_step_n_batches_to_receive[route_step] = n_batches

    def set_convergence_step_receives_from(self, conv_step: str, route_step: str) -> None:
        """When a route step gets a batch routed to it, add the name of the route step 
        to the convergence step's `convergence_step_receives_from` dict so it knows 
        which route steps to expect batches from."""
        self._steps[conv_step].convergence_step_receives_from[route_step] = True

    def convergence_step_receives_from(self, conv_step: str) -> List[str]:
        """Get the route steps that the convergence step receives batches from."""
        return [
            s for s in 
            list(self._steps[conv_step].convergence_step_receives_from.keys())
            if s != LAST_BATCH_ROUTED_FLAG
        ]

    def notify_route_step_of_last_batch(self, step_name: str) -> None:
        """Notifies the step that the last batch has been received."""
        self._steps[step_name].notify_route_step_of_last_batch()

    def step_has_finished(self, step_name: str) -> bool:
        """Indicates if the step has finished by checking if it sent a batch with `last_batch==True`
        or it was sent the `LAST_BATCH_SENT_FLAG`.

        Args:
            step_name: the name of the step to be checked.

        Returns:
            `True` if step has finished generating batches, `False` otherwise.
        """
        return step_name in self._last_batch_flag_sent_to or (
            self._last_batch_received[step_name] is not None
            and self._last_batch_received[step_name].last_batch  # type: ignore
        )

    @classmethod
    def from_dag(  # noqa: C901
        cls, dag: "DAG", use_cache: bool = False, steps_data_path: Optional[Path] = None
    ) -> "_BatchManager":
        """Create a `_BatchManager` instance from a `DAG` instance.

        Args:
            dag: The `DAG` instance.
            use_cache: whether or not to try loading outputs from steps of previous pipelines
                executions. Defaults to `False`.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.

        Returns:
            A `_BatchManager` instance.
        """
        steps = {}
        last_batch_received = {}
        last_batch_sent = {}
        last_batch_flag_sent_to = []
        received_batch_seq_nos = {}

        load_batches = {}
        steps_to_load_data_from_previous_executions: Dict[str, Union[Path, None]] = {}
        for step_name in dag:
            step: "_Step" = dag.get_step(step_name)[STEP_ATTR_NAME]
            last_batch_received[step.name] = None
            last_batch_sent[step.name] = None
            received_batch_seq_nos[step.name] = []
            predecessors = list(dag.get_step_predecessors(step_name))
            convergence_step = all(
                dag.get_step(predecessor).get(RECEIVES_ROUTED_BATCHES_ATTR_NAME, False)
                for predecessor in predecessors
            ) and len(predecessors) > 0
            batch_manager_step = _BatchManagerStep.from_step(
                step=step,
                predecessors=predecessors,
                convergence_step=convergence_step,
            )

            all_step_precessors_use_cache = all(
                dag.get_step(step_name)[STEP_ATTR_NAME].use_cache
                for step_name in predecessors
            )
            if use_cache and step.use_cache and all_step_precessors_use_cache:
                step_data_path = steps_data_path / batch_manager_step.signature
                if step_data_path.exists():
                    steps_to_load_data_from_previous_executions[step_name] = (
                        step_data_path
                    )
                    # We only want to load the outputs that are directly needed by the added
                    # steps, so if we need to load the outputs of one step and one of its
                    # predecessors it's in the list, then we remove it.
                    for predecessor in predecessors:
                        if predecessor in steps_to_load_data_from_previous_executions:
                            steps_to_load_data_from_previous_executions[predecessor] = (
                                None
                            )

            steps[step_name] = batch_manager_step

        for (
            step_name,
            step_outputs_path,
        ) in steps_to_load_data_from_previous_executions.items():
            last_batch_flag_sent_to.append(step_name)
            if step_outputs_path is None:
                continue
            load_batches[step_name] = sorted(
                [
                    _Batch.from_json(batch_file)
                    for batch_file in step_outputs_path.glob("*.json")
                    if batch_file.is_file() and batch_file.suffix == ".json"
                ],
                key=lambda x: x.seq_no,
            )
            last_batch_received[step_name] = load_batches[step_name][-1]

        # Load batches from previous steps in batch manager steps
        for step_name, batch_manager_step in steps.items():
            for predecessor in dag.get_step_predecessors(step_name):
                if predecessor in load_batches:
                    batch_manager_step.data[predecessor] = deepcopy(
                        load_batches[predecessor]
                    )
                    batch_manager_step.last_batch_received.append(predecessor)

        return cls(
            steps,
            last_batch_received,
            last_batch_sent,
            last_batch_flag_sent_to,
            received_batch_seq_nos,
        )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the `_BatchManager` to a dictionary.

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that are kept to match the signature of the parent method.

        Returns:
            Dict[str, Any]: Internal representation of the `_BatchManager`.
        """
        return {
            "steps": {name: step.dump(**kwargs) for name, step in self._steps.items()},
            "last_batch_received": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_received.items()
            },
            "last_batch_sent": {
                step_name: batch.dump(**kwargs) if batch is not None else None
                for step_name, batch in self._last_batch_sent.items()
            },
            "last_batch_flag_sent_to": self._last_batch_flag_sent_to,
            "received_batch_seq_nos": self._received_batch_seq_nos,
        }

    def cache(self, path: Path, steps_data_path: Path) -> None:  # noqa: C901
        """Cache the `_BatchManager` to a file.

        Args:
            path: The path to the file where the `_BatchManager` will be cached. If `None`,
                then the `_BatchManager` will be cached in the default cache folder.
            steps_data_path: The path where the outputs of each `Step` (considering its
                signature) will be saved for later reuse in another pipelines executions.
        """

        def save_batch(
            batches_dir: Path, batch_dump: Dict[str, Any], batch_list: List[_Batch]
        ) -> Path:
            seq_no = batch_dump["seq_no"]
            data_hash = batch_dump["data_hash"]
            batch_file = batches_dir / f"batch_{seq_no}_{data_hash}.json"

            # Save the batch if it doesn't exist
            if not batch_file.exists():
                # Get the data of the batch before saving it
                batch = next(batch for batch in batch_list if batch.seq_no == seq_no)
                batch_dump["data"] = batch.data
                self.save(path=batch_file, format="json", dump=batch_dump)

            return batch_file

        def remove_files(keep_files: List[str], dir: Path) -> None:
            files = list_files_in_dir(dir, key=None)
            remove = set(files) - {Path(file) for file in keep_files}
            for file in remove:
                file.unlink()

        path = Path(path)

        # Do not include `_Batch` data so `dump` is fast
        dump = self.dump(include_batch_data=False)
        batch_manager_step_files = {}

        # Do this to avoid modifying the dictionary while iterating over it
        batch_manager_steps = set(dump["steps"].keys())
        for step_name in batch_manager_steps:
            step_dump = dump["steps"].pop(step_name)

            # Create a directory for each batch manager step to store their batches
            batch_manager_step_dir = path.parent / "batch_manager_steps" / step_name
            batch_manager_step_dir.mkdir(parents=True, exist_ok=True)

            # Store each built `_Batch` in a separate file
            built_batches_dir = batch_manager_step_dir / "built_batches"
            built_batches_dir.mkdir(parents=True, exist_ok=True)
            step_dump["built_batches"] = [
                str(
                    save_batch(
                        batches_dir=built_batches_dir,
                        batch_dump=batch_dump,
                        batch_list=self._steps[step_name].built_batches,
                    )
                )
                for batch_dump in step_dump["built_batches"]
            ]
            # Remove built `_Batch`es that were consumed from cache
            remove_files(step_dump["built_batches"], built_batches_dir)

            # Store the `_BatchManagerStep` info
            batch_manager_step_file = str(
                path.parent / f"batch_manager_steps/{step_name}/batch_manager_step.json"
            )
            self.save(path=batch_manager_step_file, format="json", dump=step_dump)

            # Store the path to the `_BatchManagerStep` file
            batch_manager_step_files[step_name] = batch_manager_step_file

        dump["steps"] = batch_manager_step_files
        self.save(path=path, format="json", dump=dump)

    @classmethod
    def load_from_cache(
        cls, dag: "DAG", batch_manager_path: "StrOrPath", steps_data_path: "StrOrPath"
    ) -> "_BatchManager":
        """Loads the `_BatchManager` from a cache file.

        Args:
            path: The path to the cache file.
        """
        _check_is_dir(batch_manager_path)
        content = read_json(batch_manager_path)

        # Read each `_BatchManagerStep` from file
        steps = {}
        for step_name, step_file in content["steps"].items():
            steps[step_name] = read_json(step_file)

            # When reading back from JSON, `next_expected_seq_no` and `step_offset` is a
            # list (because JSON files do not have tuples).
            steps[step_name]["next_expected_seq_no"] = {
                k: tuple(v) for k, v in steps[step_name]["next_expected_seq_no"].items()
            }
            steps[step_name]["step_offset"] = {
                k: tuple(v) for k, v in steps[step_name]["step_offset"].items()
            }

            # TODO: where are we writing built batches now? xD
            # Read each `_Batch` from file
            steps[step_name]["built_batches"] = [
                read_json(batch) for batch in steps[step_name]["built_batches"]
            ]

            # Read the batches from the `steps_data` directory to populate back the `_BatchManagerStep`
            step_offset = steps[step_name]["step_offset"]
            for successor_step_name, offset in step_offset.items():
                batch_offset, batch_row_offset = offset
                step: "_Step" = dag.get_step(successor_step_name)[STEP_ATTR_NAME]
                successor_step_data_path = (
                    steps_data_path / f"{step.name}_{step.signature}"
                )

                # read batches from successor step from the step data directory taking into
                # account offset
                batches = []
                for batch_file in successor_step_data_path.glob("*.json"):
                    if not batch_file.is_file() or batch_file.suffix != ".json":
                        continue

                    # If the batch number is lower than the batch offset then we should
                    # skip it as it has already been processed by the step
                    batch_no = int(batch_file.stem.split("batch_")[1])
                    if batch_no < batch_offset:
                        continue

                    # read the batch and skip the first N rows of the first batch
                    batch = read_json(batch_file)
                    if batch_no == batch_offset:
                        batch["data"][0] = batch["data"][0][batch_row_offset:]

                    batches.append(batch)

                # sort batches by `seq_no` as it's a requirement for checking if ready to
                # create next batch
                batches.sort(key=lambda batch: batch["seq_no"])
                steps[step_name]["data"][successor_step_name] = batches

        content["steps"] = steps
        return cls.from_dict(content)

    def invalidate_cache_for(
        self, step_name: str, dag: "DAG", steps_data_path: Path
    ) -> None:
        """Invalidates the cache for the given step and its predecessors.

        Args:
            step_name: the name of the step for which the cache will be invalidated.
            dag: the `DAG` of the pipeline containing the steps.
            steps_data_path: the path where the output batches of each `Step` were saved
                for reuse in another pipeline execution.
        """
        invalidate_if_predecessor = []
        for sorted_step in dag:
            if (sorted_step == step_name) or any(
                predecessor in invalidate_if_predecessor
                for predecessor in dag.get_step_predecessors(sorted_step)
            ):
                self._reset_batch_manager_for_step(sorted_step, dag)
                invalidate_if_predecessor.append(sorted_step)

        self._load_predecessor_batches(step_name, dag, steps_data_path)

    def _reset_batch_manager_for_step(self, step_name: str, dag: "DAG") -> None:
        """Resets the batch manager state for a given step i.e. creates a new clean `_BatchManagerStep`
        for the step and removes the step name from the lists of states of the `BatchManager`.

        Args:
            step_name: the name of step for which its batch manager state needs to be cleaned.
            dag: the `DAG` of the pipeline containing the steps.
        """
        predecessors = list(dag.get_step_predecessors(step_name))
        convergence_step = dag.is_convergence_step(step_name)
        step = dag.get_step(step_name)[STEP_ATTR_NAME]
        self._steps[step_name] = _BatchManagerStep.from_step(
            step, predecessors=predecessors, convergence_step=convergence_step
        )

        self._last_batch_received[step_name] = None
        self._last_batch_sent[step_name] = None
        if step_name in self._last_batch_flag_sent_to:
            self._last_batch_flag_sent_to.remove(step_name)

    def _load_predecessor_batches(
        self, step_name: str, dag: "DAG", steps_data_path: Path
    ) -> None:
        """Loads the cached batches of the predecessors of the step in its `_BatchManagerStep`.

        Args:
            step_name: the name of the step whose predecessors' batches will be loaded.
            dag: the `DAG` of the pipeline containing the steps.
            steps_data_path: the path where the output batches of each `Step` were saved
                for reuse in another pipeline execution.
        """
        for predecessor in dag.get_step_predecessors(step_name):
            step_predecessor = dag.get_step(predecessor)[STEP_ATTR_NAME]
            predecessor_step_data_path = (
                steps_data_path
                / f"{step_predecessor.name}_{step_predecessor.signature}"
            )
            batch_files = list_files_in_dir(
                predecessor_step_data_path, key=lambda x: int(x.stem.split("_")[-1])
            )
            for file in batch_files:
                batch = _Batch.from_file(file)
                if batch.last_batch:
                    self._steps[step_name].last_batch_received.append(batch.step_name)
                self._steps[step_name].data[predecessor].append(batch)
