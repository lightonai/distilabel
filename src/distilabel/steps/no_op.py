from typing import TYPE_CHECKING
from distilabel.steps import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class NoOp(Step):
    '''
    A step that does nothing.

    Useful in a few cases I can think of:
    1. When you want to rename columns
    2. When you need to connect the output of a list of Tasks (e.g. LMGenerationTasks) to a RoutingBatchFunction
        (this will throw an error, so we use this step to avoid that)  
    3. When you need to join a convergence step and another step's results, you have to send the convergence step's results
        through this step before distilabel will let you join with the other step's results.
    4. When you need to break up a global step's results into batches (say so they don't all get routed together)

    You'll need to specify the `cols` argument if you want to rename columns cause they have to be within
    the inputs/outputs of the step.
    '''
    cols: list[str] = []

    @property
    def inputs(self) -> 'StepColumns':
        return self.cols

    @property
    def outputs(self) -> 'StepColumns':
        return self.cols

    def process(self, *inputs: StepInput) -> 'StepOutput':
        for step_input in inputs:
            yield step_input
