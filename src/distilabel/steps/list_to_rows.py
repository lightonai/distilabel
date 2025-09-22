from typing import TYPE_CHECKING, Optional
from distilabel.steps import Step, StepInput
import random

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class ListToRows(Step):
    '''
    Takes a list from column `input_col` (expected to be a list of any Python base type) and splits it 
    into separate rows, replacing the `input_col` and passing through `other_fields` 
    (any field not referenced is dropped, so account for that).

    If `sample_n` is provided, up to `sample_n` random items will be selected from each list
    (without replacement) before splitting. Use `sample_random_seed` for deterministic sampling.
    '''
    input_col: str
    sample_n: Optional[int] = None
    sample_random_seed: Optional[int] = None

    @property
    def inputs(self) -> 'StepColumns':
        return [self.input_col]

    @property
    def outputs(self) -> 'StepColumns':
        return [self.input_col]

    def process(self, *inputs: StepInput) -> 'StepOutput':
        rng = random.Random(self.sample_random_seed) if self.sample_random_seed is not None else random
        expanded_fields = []
        for step_input in inputs:
            for row in step_input:
                values = row[self.input_col]
                if values:
                    if self.sample_n is not None:
                        k = min(self.sample_n, len(values))
                        sampled = rng.sample(values, k=k)
                    else:
                        sampled = values
                else:
                    sampled = [None]
                for field in sampled:
                    expanded_fields.append(row | {self.input_col: field})
        yield expanded_fields
