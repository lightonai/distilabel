
import time
from collections import defaultdict
from functools import wraps
import logging
from typing import Any, Callable, Dict, List, Generator
from contextlib import contextmanager

logger = logging.getLogger("distilabel.timer")

class PipelineTimer:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "PipelineTimer":
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, enabled: bool = False) -> None:
        if not hasattr(self, "_initialized"):
            self.timings: Dict[str, List[float]] = defaultdict(list)
            self.total_times: Dict[str, float] = defaultdict(float)
            self.call_counts: Dict[str, int] = defaultdict(int)
            self._enabled = enabled
            self._initialized = True

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def reset(self) -> None:
        self.timings.clear()
        self.total_times.clear()
        self.call_counts.clear()

    def time_it(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self._enabled:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            func_name = func.__name__
            self.timings[func_name].append(elapsed_time)
            self.total_times[func_name] += elapsed_time
            self.call_counts[func_name] += 1

            return result
        return wrapper
    
    @contextmanager
    def time_block(self, name: str) -> Generator[None, None, None]:
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.timings[name].append(elapsed_time)
            self.total_times[name] += elapsed_time
            self.call_counts[name] += 1

    def get_summary(self) -> str:
        if not self._enabled or not self.total_times:
            return "No timing data collected."

        summary = ["--- Timing Summary ---"]
        sorted_times = sorted(
            self.total_times.items(), key=lambda item: item[1], reverse=True
        )

        for name, total_time in sorted_times:
            count = self.call_counts[name]
            avg_time = total_time / count if count > 0 else 0
            summary.append(
                f"{name:<40} | total: {total_time:>8.4f}s | count: {count:>6} | avg: {avg_time:>8.4f}s"
            )
        return "\n".join(summary)


_timer_instance = PipelineTimer()

def get_timer() -> "PipelineTimer":
    return _timer_instance 
