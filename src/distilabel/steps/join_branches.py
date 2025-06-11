from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Dict

from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput

def make_hashable(item: Any):
    '''
    Recursively convert an item to a hashable representation.
    '''
    if isinstance(item, list):
        # Recursively convert list elements and form a tuple
        return tuple(make_hashable(sub_item) for sub_item in item)
    elif isinstance(item, dict):
        converted_items = []
        for k, v in item.items():
            converted_items.append(
                (make_hashable(k), make_hashable(v))
            )
        # Sort by the first element of the pair (the converted key)
        return tuple(sorted(converted_items, key=lambda x: x[0]))
    elif isinstance(item, set):
        converted_set_items = [make_hashable(sub_item) for sub_item in item]
        return frozenset(converted_set_items)
    return item

class JoinParallelBranches(GlobalStep):
    """
    A `Step` that joins rows from multiple input branches based on a common column.

    This step is designed to merge data from different processing paths (branches)
    that have a common identifier. It requires that the specified join column
    exists in all rows to be successfully joined.

    Attributes:
        join_on_cols: The name of the columns to use for joining rows across branches.
        extra_cols: A list of columns that you can reference in input or output mappings.

    Output:
        A single `StepOutput` (a list of dictionaries) where each dictionary is a
        merged row. A row is included in the output only if its `join_on_col` value
        was present in every input branch. Otherwise, the row (and its counterparts
        from other branches) is dropped.

    Merging Behavior:
        - Columns from all contributing branches are merged into a single dictionary.
        - If column names (other than `join_on_col`) overlap between branches, the
          value from the branch that appears later in the `*inputs` sequence will
          overwrite values from earlier branches.
    """

    join_on_cols: List[str]
    extra_cols: List[str] = []

    @property
    def inputs(self) -> "StepColumns":
        return self.join_on_cols + self.extra_cols

    @property
    def outputs(self) -> "StepColumns":
        return self.join_on_cols + self.extra_cols

    def process(self, *inputs: StepInput) -> "StepOutput":
        num_branches = len(inputs)

        # grouped_by_join_cols maps: join_key -> [row_from_branch_0, row_from_branch_1, ...]
        # Initialize with Nones to easily check for completeness.
        grouped_by_join_cols = defaultdict(lambda: [None] * num_branches)

        for branch_idx, step_input_branch in enumerate(inputs):
            for row in step_input_branch:
                join_val = make_hashable([row[col] for col in self.join_on_cols])

                # If a branch has multiple rows with the same join_val, raise a warning
                if grouped_by_join_cols[join_val][branch_idx] is not None:
                    self._logger.warning((
                        f"Duplicate values in column {self.join_on_cols}: {join_val=} in JoinParallelBranches step {self.name}. "
                        "This will cause some rows to be dropped, normally you want the join_val to be unique to a branch. "
                        "This can be a false alarm if a model previously generated duplicate outputs."
                    )
                )
                grouped_by_join_cols[join_val][branch_idx] = row

        merged_rows: List[Dict[str, Any]] = []
        for join_val, rows_from_branches in grouped_by_join_cols.items():
            # Check if all branches contributed a row for this join_val
            if all(row is not None for row in rows_from_branches):
                combined_row: Dict[str, Any] = {}
                for row_from_branch in rows_from_branches:
                    combined_row.update(row_from_branch)
                merged_rows.append(combined_row)
            # Else, at least one branch did not have this join_val, so we drop these rows

        yield merged_rows
        
