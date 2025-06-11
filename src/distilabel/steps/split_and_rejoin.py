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
                row 
                | {f'{self.input_col}_uuid': str(uuid.uuid4())}
                | {
                    f'{self.input_col}_len_before_split': 
                    len(row[self.input_col]) if row[self.input_col] else 0
                }
                for row in step_input
            ]
            expanded_fields = [
                row 
                | {self.input_col: [field] if self.keep_as_list and field is not None else field}
                for row in step_input
                    for field in (row[self.input_col] if row[self.input_col] else [None])
            ]
            yield expanded_fields

class Rejoin(GlobalStep):
    '''
    Joins rows that have the same uuid in the `{input_col}_uuid` column.

    All values in other columns that are lists will be concatenated together

    Otherwise the values will be joined into a list, excepting the following:

    We need a policy per col of not turning into a list (which we don't want to do for duplicate values)
    because if is based on each instance, you can end up with a list or a single value in a col, which isn't allowed 
    by pytable. So, you must be sure these cols are cols with duplicated and don't need to be made into lists. 
    Specify these cols in `duplicates_cols`.

    This is a global step because we can't have rows with same uuid in different batches
    (they wouldn't get joined)

    If you use split, drop/lose one of the rows, then use rejoin, then when rejoined, it won't have the full list of original elements. You can drop these rows with drop_incomplete_rows=True (default).

    This step won't work as you might want if rows are created after being split.
    '''
    input_col: str
    duplicates_cols: set[str] = {}
    drop_incomplete_rows: bool = True

    @property
    def inputs(self) -> 'StepColumns':
        return [f'{self.input_col}_uuid']

    @property
    def outputs(self) -> 'StepColumns':
        return [self.input_col]

    def drop_incomplete(self, rows: list[dict]) -> list[dict]:
        return [
            row
            for row in rows
            if row.get(f'{self.input_col}_len_before_split') == len(row[self.input_col])
        ]        

    def process(self, *inputs: StepInput) -> 'StepOutput':
        from collections import defaultdict

        for step_input in inputs:
            uuid_col = f"{self.input_col}_uuid"

            # maps uuid to list of rows with that uuid
            groups = defaultdict(list)
            for row in step_input:
                groups[row[uuid_col]].append(row)
            
            rejoined_rows = []
            for rows in groups.values():
                # rows is a list of dict
                merged = {} # new merged row (dict with cols)
                for key in set().union(*rows):  # find all unique keys
                    if key == uuid_col:
                        continue
                    vals = [r.get(key) for r in rows]
                    # values of None will be concatenated into a list
                    if all(isinstance(v, (list, type(None))) for v in vals):  # concatenate lists case
                        merged[key] = list(chain(*(v if v is not None else [None] for v in vals)))
                    elif key in self.duplicates_cols:  # all the same value case
                        merged[key] = vals[0]
                    else:
                        merged[key] = vals
                rejoined_rows.append(merged)
            if self.drop_incomplete_rows:
                rejoined_rows = self.drop_incomplete(rejoined_rows)
            # Remove the len_before_split column as it's no longer needed
            for row in rejoined_rows:
                row.pop(f'{self.input_col}_len_before_split', None)
            yield rejoined_rows
