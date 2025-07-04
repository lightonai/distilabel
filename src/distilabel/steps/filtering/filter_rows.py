from typing import TYPE_CHECKING, Callable
from distilabel.steps import Step, StepInput
from pydantic import Field

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class FilterRows(Step):
    '''
    For each row, check if the condition is met (`cols` is passed to the condition)

    If the condition is met, the row is kept, otherwise it is dropped.

    Example:
    ---
    ```python
    drop_none_questions = FilterRows(
        cols=['question'],
        condition=lambda row, cols: row[cols[0]] is not None
    )
    ```
    '''
    cols: list[str]
    condition: Callable = Field(default=lambda **kwargs: True, exclude=True)

    @property
    def inputs(self) -> 'StepColumns':
        return self.cols

    @property
    def outputs(self) -> 'StepColumns':
        return self.cols

    def process(self, *inputs: StepInput) -> 'StepOutput':  
        for step_input in inputs:
            yield [
                row for row in step_input
                if self.condition(row, self.cols)
            ]
