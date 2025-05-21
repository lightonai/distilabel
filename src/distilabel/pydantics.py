from pydantic import BaseModel, model_validator, Field
from pathlib import Path as pth
import sys
from typing import Any

from distilabel import utils

class CategoricalDist(BaseModel):
    choices: list[tuple[str, float]]
    '''A list of (value, probability) pairs'''
    samples_per_prompt: int | None = 1
    '''if an integer, number of samples to take per prompt; if None, refers to another sampled kwarg 
    specified in the main config by samples_per_prompt_kwarg which is sampled once per prompt'''
    side_by_side: bool = False
    '''if True, will be broadcasted to a section in the prompt gathering list like kwargs rather than appearing separately'''

    @model_validator(mode='after')
    def normalize(self) -> 'CategoricalDist':
        """
        Normalize the probabilities of choices so they sum to 1,
        and update the choices list in-place.
        """
        weights = [abs(prob) for _, prob in self.choices]
        total = sum(weights)
        normalized_choices = [
            (val, prob / total) for (val, prob) in self.choices
        ]
        self.choices = normalized_choices
        return self

class PromptSamplerConfig(BaseModel):
    distributions: dict[str, CategoricalDist] = {}
    '''map to distributions for formatting the prompt template'''
    samples_per_prompt_kwarg: str | None = None
    '''if a string, will be sampled once per prompt and used as the samples_per_prompt 
    kwarg for distributions with samples_per_prompt=None'''

class LMConfig(BaseModel):
    '''Config for a model used to generate data'''
    path: str = ''
    '''path/hf id for the model'''

    ## task section
    data_ratio: float = 1.0
    '''ratio of the data for this model to generate, doesn't have to be normalized, as it goes to random.choices(weights=...)'''
    task_name: str | None = None
    '''name of the task the model is used for, use this in your pipeline to map the lm_config to the task'''
    task_kwargs: dict[str, Any] = {}
    '''kwargs for the task, use this to pass in task specific kwargs'''
    out_model: type[BaseModel] | str | None = None
    '''
    pass a string that is the name of a pydantic model in configs.py, a pydantic model, or None.
    
    if None, the model will not attempt to format the output as a pydantic model
    '''
    prompt_sampler_config: PromptSamplerConfig = Field(default_factory=PromptSamplerConfig)
    '''config for the prompt sampler, which formats the prompt kwargs probabilistically'''

    ## generation section
    system_template_path: str | None = None
    system_template: str = ''
    temperature: float = 0.4
    max_new_tokens: int = 2048

    ## gpu section
    tp_size: int | None = None
    '''number of gpus to use for the model, applies if using vllm'''
    replicas: int = 1
    '''number of replicas to create'''
    vllm_kwargs: dict[str, Any] = {}
    '''kwargs passed directly to vllm. Use None as a value if the kwarg is just a flag'''

    def model_post_init(self, context) -> None:
        if isinstance(self.out_model, str):
            self.out_model = getattr(sys.modules[__name__], self.out_model)
        if self.system_template_path:
            self.system_template = pth(self.system_template_path).read_text()
    
class Stage(BaseModel):
    '''
    Config for a stage of the pipeline 

    (essentially letting you logically separate a pipeline into stages with different configs for each)

    you should run each stage in a separate load stage so that all gpus are available to each stage
    '''
    ## lm section
    lm_configs: list[LMConfig] = []
    '''list of LMConfigs that are active in this stage'''

    ## gpu section
    available_gpus: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]

    ## global/defaults section
    default_system_template_path: str | None = None
    '''default system template path, used for LMConfigs that don't specify one'''
    max_dims: tuple[int, int] = (1000, 1100)
    '''max dimensions for the images, [shorter side, longer side]'''

    @model_validator(mode='after')
    def apply_default_system_template(self) -> 'Stage':
        # Apply default system template path to LMConfigs if needed
        if self.default_system_template_path is not None:
            for lm_config in self.lm_configs:
                # if lm_config has no system_template_path, load the default
                if lm_config.system_template_path is None:
                    lm_config.system_template_path = self.default_system_template_path
                    lm_config.system_template = pth(lm_config.system_template_path).read_text()
        return self

class Config(BaseModel):
    '''Config for a pipeline'''
    stages: list[Stage]
    debug_with_running_vllm: bool = False
    '''
    if True, all vllm models will expect to be able to call a pre-running model on port 8000.
    
    This is useful for debugging, when you don't want to pay for proprietary models or start a vllm server on launch
    '''

class SinglePageQuestions(BaseModel):
    '''Config for the single page questions output format'''
    questions: list[str]

class MultiPageQuestions(BaseModel):
    '''Model for structured output from multi-page question generation'''
    analysis: str
    questions: list[str]

class QuestionRequirements(BaseModel):
    '''Ask the LM to break down the question into requirements'''
    question_requirements: str
    # making this a string so that it can serve as input to a LM

class SatisfactoryAnswer(BaseModel):
    '''
    Determine if the answer satisfies requirements
    '''
    question_requirements_met: list[bool]
    question_fully_answered: bool = False

    @model_validator(mode='after')
    def apply_default_system_template(self) -> 'SatisfactoryAnswer':
        # might want custom logic like 75% of requirements met
        if all(self.question_requirements_met):
            self.question_fully_answered = True
        return self
