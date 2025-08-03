from typing import TYPE_CHECKING, Callable, Annotated
from functools import partial
from pydantic import Field, model_validator

from distilabel.steps.tasks import Task
from distilabel.pydantics import Stage, LMConfig

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        ChatType,
    )

class LMGenerationTask(Task):
    '''
    Task for running LM/VLM generation with a sampled system prompt and structured output.

    The pydantic model will be unpacked into separate columns for output or will be none if 
    structured output fails within `STRUCTURED_OUTPUT_RETRIES`. If the pydantic model is None,
    the output will be returned as a string in the 'generation' column.

    Args:
    ---
        system_col: column to use for the system prompt, if specified, replaces sampling from the lm_config.system_template_path
        lm_input_cols: extra columns to include in the messages to the LM, postfixed in order 
        lm_input_col_prefixes: prefixes to prepend to the lm_input_cols (e.g. 'reference answer: ')
        extra_cols: extra columns for the step to know about for input or output mappings
    '''
    stage: Stage = Field(default_factory=Stage, exclude=True)
    lm_config: LMConfig = Field(default_factory=LMConfig, exclude=True)
    system_col: str | None = None
    lm_input_cols: list[str] = []
    lm_input_col_prefixes: list[str] = []
    input_formatter: Callable = Field(default=lambda **kwargs: kwargs, exclude=True)
    parallel_input_formatter: Callable | None = Field(default=None, exclude=True)
    extra_cols: list[str] = []
    use_cache: bool = True  # this affects the batch level caching, not the lm level caching

    @model_validator(mode='after')
    def valid_prefix_length(self) -> 'LMGenerationTask':
        if len(self.lm_input_col_prefixes) != 0 and len(self.lm_input_col_prefixes) != len(self.lm_input_cols):
            raise ValueError((
                f'lm_input_col_prefixes must be the same length as lm_input_cols, '
                f'got {len(self.lm_input_col_prefixes)} and {len(self.lm_input_cols)}'
            ))
        return self

    # note that Task.unload() will unload the llm, so we don't need to do that ourselves
    def load(self):
        # add_raw_input is set to False because if it has image type messages, they can't be formatted in a pytable
        super().load()
        self.add_raw_input = False
        self.lm_config.lm_response_cache_root = self.cache_location['lm_cache']
    
    @property
    def pydantic_fields(self) -> list[str]:
        if self.lm_config.out_model is None:
            return ['generation']
        return list(self.lm_config.out_model.model_fields.keys())

    @property
    def inputs(self) -> 'StepColumns':
        return ['source'] + self.lm_input_cols + self.extra_cols

    @property
    def outputs(self) -> 'StepColumns':
        return ['source', 'model_name', *self.pydantic_fields, 'system'] + self.extra_cols

    def format_input(self, input: dict) -> 'ChatType':
        return self.input_formatter(input, self.system_col, self.lm_input_cols, self.lm_input_col_prefixes)

    def can_parallel_format_inputs(self) -> bool:
        return self.parallel_input_formatter is not None

    def parallel_format_inputs(self, inputs: list[dict]) -> list['ChatType']:
        return self.parallel_input_formatter(inputs, self.system_col, self.lm_input_cols, self.lm_input_col_prefixes)

    def format_output(self, output: str | None, input: dict) -> dict:
        pydantic_output = {'generation': output}
        if self.lm_config.out_model is not None:
            # if using structured output, split the generation into columns with names from the pydantic model
            none_dict = dict.fromkeys(self.pydantic_fields)
            load_pydantic = partial(self.lm_config.out_model.model_validate_json, strict=True)

            pydantic_output = load_pydantic(output).model_dump() if output is not None else none_dict
        return {**pydantic_output, 'source': input['source'], 'system': input['system']}
