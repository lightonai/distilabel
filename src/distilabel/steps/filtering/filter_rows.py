from typing import TYPE_CHECKING, Callable
from distilabel.steps import Step, StepInput
from pydantic import Field
from distilabel.utils.timer import get_timer

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

_timer = get_timer()

class FilterRows(Step):
    '''
    For each row, check if the condition is met (`cols` is passed to the condition)

    If the condition is met, the row is kept, otherwise it is dropped.

    If `cols` is not passed, the condition is only passed the row. This allows you to e.g.
    condition=utils.logical_and_filters(
        partial(utils.generation_is_structured, cols=['generation']),
        partial(utils.logical_not_filter(utils.cols_true), cols=['is_references_page']),
    )

    Example:
    ---
    ```python
    drop_none_questions = FilterRows(
        cols=['question'],
        condition=lambda row, cols: row[cols[0]] is not None
    )
    ```
    '''
    cols: list[str] | None = None
    condition: Callable = Field(default=lambda **kwargs: True, exclude=True)

    @property
    def inputs(self) -> 'StepColumns':
        return self.cols

    @property
    def outputs(self) -> 'StepColumns':
        return self.cols

    @_timer.time_it
    def process(self, *inputs: StepInput) -> 'StepOutput':  
        for step_input in inputs:
            yield [
                row for row in step_input
                if (
                    self.condition(row, self.cols) if self.cols is not None 
                    else self.condition(row)  # allow specifying cols with partial or taking the whole row
                )
            ]
