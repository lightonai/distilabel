from __future__ import annotations

"""High-performance multiprocessing queue with lightweight introspection.

This class is a drop-in **write-path** replacement for the former
`ManagedListQueue`.  It keeps the hot operations (`put`, `get`) backed by a
native `multiprocessing.Queue` (C-level fifo), while maintaining a few integer
counters in shared memory so other processes can *query* the queue state
without incurring the expensive Manager/proxy round-trip.

Only the information actually needed by the pipeline is tracked:
    • number of *real* batches (everything except `None` and the
      `LAST_BATCH_SENT_FLAG` sentinels)
    • presence/count of `LAST_BATCH_SENT_FLAG`
    • total size (for convenience)

The public API purposefully covers just what the pipeline uses – we do **not**
aim for full feature parity with `queue.Queue`.
"""

import multiprocessing as mp
from multiprocessing import Lock, Queue, Value
from typing import Any, Callable, Optional

from distilabel import constants

__all__ = ["TrackingQueue"]


class TrackingQueue:  # pylint: disable=too-few-public-methods
    """A fast, process-shared queue with O(1) `put`/`get`.

    Parameters
    ----------
    maxsize:
        Maximum size of the queue. 0 (default) means unlimited (same semantics
        as ``multiprocessing.Queue``).
    """

    def __init__(self, manager: mp.Manager, maxsize: int = 0):
        self._queue: Queue[Any] = manager.Queue(maxsize)
        self._lock: Lock = manager.Lock()
        # Using signed int ('i') – plenty for queue length counting.
        self._batches_count: Value = manager.Value("i", 0)
        self._flag_count: Value = manager.Value("i", 0)
        self._none_count: Value = manager.Value("i", 0)
        self._maxsize = maxsize

    # ---------------------------------------------------------------------
    # Standard queue interface (subset)
    # ---------------------------------------------------------------------
    def qsize(self) -> int:  # noqa: D401 — keep same name as stdlib
        return self._queue.qsize()

    def empty(self) -> bool:  # noqa: D401
        return self.qsize() == 0

    def full(self) -> bool:  # noqa: D401
        return 0 < self._maxsize <= self.qsize()

    # ------------------------------------------------------------------
    # Lightweight state-inspection helpers
    # ------------------------------------------------------------------
    def batches_count(self) -> int:
        """Number of *real* (data) batches currently in the queue."""
        return self._batches_count.value

    def flag_count(self) -> int:
        return self._flag_count.value

    def flag_present(self) -> bool:
        return self._flag_count.value > 0

    def total_size(self) -> int:
        return self.qsize()

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:  # noqa: D401,E501
        """Enqueue *item* and update counters."""
        self._queue.put(item, block, timeout)
        with self._lock:
            self._inc(item)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:  # noqa: D401
        """Dequeue an item and update counters."""
        item = self._queue.get(block, timeout)
        with self._lock:
            self._dec(item)
        return item

    def get_with_counts(self, *, block: bool = True, timeout: Optional[float] = None):
        """Equivalent to ``get`` but returns a tuple *(item, counts)* where counts
        is ``(batches_count, flag_count, none_count)`` *after* removal."""
        item = self._queue.get(block, timeout)
        with self._lock:
            self._dec(item)
            counts = (
                self._batches_count.value,
                self._flag_count.value,
                self._none_count.value,
            )
        return item, counts 

    # ------------------------------------------------------------------
    # Conditional pop used by route-steps
    # ------------------------------------------------------------------
    def pop_if(
        self,
        predicate: Callable[[int, int, int], bool],
    ) -> Optional[Any | tuple[Any, tuple[int, int, int]]]:
        """Atomically pop and return first item **iff** *predicate* holds.

        The predicate receives three integers: ``n_batches``, ``n_flags``
        and ``n_none`` *before* removal.  Returns the popped element or
        *None* when the predicate is *False* (queue left untouched).
        """
        with self._lock:
            if not predicate(
                self._batches_count.value,
                self._flag_count.value,
                self._none_count.value,
            ):
                return (None, None)
            try:
                item = self._queue.get_nowait()
            except Exception:
                # Another process might have dequeued at the exact moment –
                # treat as predicate failure.
                return (None, None)
            self._dec(item)
            counts = (
                self._batches_count.value,
                self._flag_count.value,
                self._none_count.value,
            )
            return (item, counts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _inc(self, item: Any) -> None:
        if item is None:
            self._none_count.value += 1
        elif item == constants.LAST_BATCH_SENT_FLAG:
            self._flag_count.value += 1
        else:
            self._batches_count.value += 1

    def _dec(self, item: Any) -> None:
        if item is None:
            self._none_count.value -= 1
        elif item == constants.LAST_BATCH_SENT_FLAG:
            self._flag_count.value -= 1
        else:
            self._batches_count.value -= 1

