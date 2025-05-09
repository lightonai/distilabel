from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Dict

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput


class JoinParallelBranches(Step):
    """
    A `Step` that joins rows from multiple input branches based on a common column.

    This step is designed to merge data from different processing paths (branches)
    that have a common identifier. It requires that the specified join column
    exists in all rows that are to be successfully joined.

    Attributes:
        join_on_col: The name of the column to use for joining rows across branches.
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

    join_on_col: str
    extra_cols: List[str] = []

    @property
    def inputs(self) -> "StepColumns":
        return [self.join_on_col] + self.extra_cols

    @property
    def outputs(self) -> "StepColumns":
        return [self.join_on_col] + self.extra_cols

    def process(self, *inputs: StepInput) -> "StepOutput":
        num_branches = len(inputs)

        # grouped_by_join_col maps: join_key -> [row_from_branch_0, row_from_branch_1, ...]
        # Initialize with Nones to easily check for completeness.
        grouped_by_join_col = defaultdict(lambda: [None] * num_branches)

        for branch_idx, step_input_branch in enumerate(inputs):
            for row in step_input_branch:
                join_val = row[self.join_on_col]
                if isinstance(join_val, list): join_val = tuple(join_val)  # can't hash lists cause they're mutable
                # If a branch has multiple rows with the same join_val, raise a warning
                if grouped_by_join_col[join_val][branch_idx] is not None:
                    self._logger.warning((
                        f"Duplicate values in {self.join_on_col} column: {join_val} in JoinParallelBranches step {self.name}. "
                        "This will cause some rows to be dropped, normally you want the join_val to be unique to a branch."
                    )
                )
                grouped_by_join_col[join_val][branch_idx] = row

        merged_rows: List[Dict[str, Any]] = []
        for join_val, rows_from_branches in grouped_by_join_col.items():
            # Check if all branches contributed a row for this join_val
            if all(row is not None for row in rows_from_branches):
                combined_row: Dict[str, Any] = {}
                for row_from_branch in rows_from_branches:
                    combined_row.update(row_from_branch)
                merged_rows.append(combined_row)
            # Else, at least one branch did not have this join_val, so we drop these rows
            # as per the requirement.

        yield merged_rows
