from collections import defaultdict
import re
from typing import TYPE_CHECKING, Any, List, Dict, Callable

from distilabel.steps.base import GlobalStep, Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput


class Map(Step):
    """
    A `Step` that applies a function to each row.

    Attributes:
        fn: A function that takes the input row as input and returns a dictionary of output columns.
            Called like fn(**row_dict, output_cols=output_cols)
        cols: A list of columns to ensure are present in the input and can be used for input/output mappings.
        output_cols: A list of columns that are output by the function.

    Output:
        list[`StepOutput`] (a list of dictionaries)
    """

    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    cols: List[str] = []
    output_cols: List[str] = []

    @property
    def inputs(self) -> "StepColumns":
        return self.cols

    @property
    def outputs(self) -> "StepColumns":
        return self.cols + self.output_cols

    def process(self, *inputs: StepInput) -> "StepOutput":
        for step_input in inputs:
            yield [{**row, **self.fn(**row, output_cols=self.output_cols)} for row in step_input]
