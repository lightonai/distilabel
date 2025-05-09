import uuid
from itertools import chain
from typing import TYPE_CHECKING
from distilabel.steps import Step, StepInput, GlobalStep

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class Split(Step):
    '''
    Like `ListToRows`, but also adds a uuid to each row so that they can be joined back together later.

    input_col is expected to have list of None values. 
    keep_as_list will keep the value as a list of len 1 after splitting.
    '''
    input_col: str
    keep_as_list: bool = True

    @property
    def inputs(self) -> 'StepColumns':
        return [self.input_col]

    @property
    def outputs(self) -> 'StepColumns':
        return [self.input_col, f'{self.input_col}_uuid']

    def process(self, *inputs: StepInput) -> 'StepOutput':
        # track a uuid for each row so that they can be joined back together later
        for step_input in inputs:
            step_input = [
                row | {f'{self.input_col}_uuid': str(uuid.uuid4())}
                for row in step_input
            ]
            expanded_fields = [
                row | {self.input_col: [field] if self.keep_as_list and field is not None else field}
                for row in step_input
                    for field in (row[self.input_col] if row[self.input_col] else [None])
            ]
            yield expanded_fields

class Rejoin(GlobalStep):
    '''
    Joins rows that have the same uuid in the `{input_col}_uuid` column.

    All values in other columns that are lists will be concatenated together

    Otherwise the values will be joined into a list, excepting the following:

    If all the values in a column are the same, the column will be joined into a single value
    unless it is in `keep_as_list_cols`

    This is a global step because we can't have rows with same uuid in different batches
    (they wouldn't get joined)
    '''
    input_col: str
    keep_as_list_cols: set[str] = {}

    @property
    def inputs(self) -> 'StepColumns':
        return [f'{self.input_col}_uuid']

    @property
    def outputs(self) -> 'StepColumns':
        return [self.input_col]

    def process(self, *inputs: StepInput) -> 'StepOutput':
        from collections import defaultdict

        for step_input in inputs:
            uuid_col = f"{self.input_col}_uuid"
            groups = defaultdict(list)
            for row in step_input:
                groups[row[uuid_col]].append(row)
            rejoined_rows = []
            for rows in groups.values():
                merged = {}
                for key in set().union(*rows):
                    if key == uuid_col:
                        continue
                    vals = [r[key] for r in rows if key in r]
                    if all(isinstance(v, (list, type(None))) for v in vals):  # concatenate lists case
                        merged[key] = list(chain(*(v if v is not None else [None] for v in vals)))
                    elif all([v == vals[0] for v in vals]) and key not in self.keep_as_list_cols:  # all the same value case
                        merged[key] = vals[0]
                    else:
                        merged[key] = vals
                rejoined_rows.append(merged)
            yield rejoined_rows
