from typing import TYPE_CHECKING, Callable, Annotated, get_origin, get_args
from functools import partial
from pydantic import Field, model_validator

from distilabel.steps.tasks import Task
from distilabel.pydantics import Stage, LMConfig

from distilabel.steps.base import (
    StepInput,
)

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        ChatType,
        StepOutput,
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
        if 'default_system', no system prompt will be added to the messages (uses model default)
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
            return ['generation', 'reasoning']
        return list(self.lm_config.out_model.model_fields.keys())

    @property
    def none_allowed_fields(self) -> list[str]:
        if self.lm_config.out_model is None:
            return []
        fields_allowing_none: list[str] = []
        for name, field in self.lm_config.out_model.model_fields.items():
            # unwrap Annotated types
            underlying = field.annotation
            while get_origin(underlying) is Annotated:
                args = get_args(underlying)
                if not args:
                    break
                underlying = args[0]
            # if NoneType is part of the union args, then the field accepts None
            if type(None) in get_args(underlying):
                fields_allowing_none.append(name)
        return fields_allowing_none

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

    def format_output(self, output: str | None, reasoning: str | None, input: dict) -> dict:
        pydantic_output = {'generation': output, 'reasoning': reasoning}
        if self.lm_config.out_model is not None:
            # if using structured output, split the generation into columns with names from the pydantic model
            none_dict = dict.fromkeys(self.pydantic_fields)
            load_pydantic = partial(self.lm_config.out_model.model_validate_json, strict=True)

            pydantic_output = load_pydantic(output).model_dump() if output is not None else none_dict
        return {**pydantic_output, 'source': input['source'], 'system': input['system']}

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        results = next(super().process(inputs))
        n_failed = sum(
            1 for result in results if 
            any(result[k] is None for k in self.pydantic_fields 
            if (k != 'reasoning' and k not in self.none_allowed_fields))
        )
        if n_failed > int(0.1 * len(results)):
            valued_fields = [f for f in self.pydantic_fields if f not in self.none_allowed_fields]
            self._logger.warning(
                f"{n_failed}/{len(results)} outputs have None values in the output fields "
                f"{valued_fields} which is more than 10% of the batch")
        yield results
