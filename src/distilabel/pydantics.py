from pydantic import BaseModel, model_validator, Field
from pathlib import Path as pth
import sys
import re
from typing import Any, Literal, Callable
from PIL import Image as PILImage

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
    pp_size: int | None = None
    '''pipeline_parallel size, applies if using vllm'''
    n_gpus: int | None = None
    '''product of vllm parallelisms, applies if using vllm'''
    replicas: int = 1
    '''number of replicas to create'''
    replicas_per_vllm_server: int = 1
    '''number of model replicas that share a single vLLM server instance (oversubscription factor)'''
    vllm_kwargs: dict[str, Any] = Field(default_factory=dict)
    '''kwargs passed directly to vllm. Use None as a value if the kwarg is just a flag'''
    api_call_extra_body: dict[str, Any] = Field(default_factory=dict)
    '''extra body passed to the api call. Can be used for e.g. chat templates'''

    lm_response_cache_root: pth = Field(default_factory=pth)
    '''root directory for the lm response cache, set by the step when created'''

    _path_substitution: tuple[str, str] | None = None
    '''set by the config'''

    postprocess_image_hook: Callable[[PILImage, int, Any], PILImage] | None = None
    '''Optional hook applied to each image loaded in utils.source_to_msg.
    Signature: (img: PIL.Image.Image, idx: int, source: Any) -> PIL.Image.Image'''

    def model_post_init(self, context) -> None:
        if isinstance(self.out_model, str):
            self.out_model = getattr(sys.modules[__name__], self.out_model)
        if self.system_template_path:
            self.system_template = pth(self.system_template_path).read_text()
        self.n_gpus = None if not (self.tp_size or self.pp_size) else (self.tp_size or 1) * (self.pp_size or 1)
    
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
    use_running_vllm: bool = False
    '''
    if True, all vllm models will expect to be able to call a pre-running model on port 8000.
    
    This is useful for debugging, when you don't want to pay for proprietary models or start a vllm server on launch
    You can also set the VLLM_API_BASE_URL environment variable to the base url of the vllm server and set up e.g. multi-node vllm servers
    behind an nginx proxy to scale to multiple nodes
    '''
    path_substitution: tuple[str | re.Pattern, str] | None = None
    '''
    if a tuple, will call str.replace(substitution[0], substitution[1]) on any paths in the source column
    '''
    @model_validator(mode='after')
    def apply_default_path_substitution(self) -> 'Config':
        for stage in self.stages:
            for lm_config in stage.lm_configs:
                if lm_config._path_substitution is None:
                    lm_config._path_substitution = self.path_substitution
        return self

class CoT(BaseModel):
    chain_of_thought: str

class SinglePageQuestions(BaseModel):
    '''Config for the single page questions output format'''
    questions: list[str]
    page_word_count: int
    is_table_of_contents: bool
    is_bibliography: bool
    not_suitable_for_questions: bool

class MultiPageQuestions(BaseModel):
    '''Model for structured output from multi-page question generation'''
    analysis: str
    questions: list[str]
    not_suitable_for_questions: bool

class AnalysisQuestion(BaseModel):
    analysis: str
    question: str

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

class Metalabel(BaseModel):
    is_references_page: bool
    word_count: int

class CountNumberedPages(BaseModel):
    scratchpad: str
    is_page_number_visible: bool

class KeyExtraction(CoT):
    extraction: str

class PosExtraction(CoT):
    extraction: str

class ThinkingCount(CoT):
    count: int

class EvidenceInChunks(BaseModel):
    evidence: str
    relevance_score: float
    relevant: bool = False

    @model_validator(mode='after')
    def apply_default_system_template(self) -> 'EvidenceInChunks':
        if self.relevance_score > 0.0:
            self.relevant = True
        return self

class UnanswerableQA(BaseModel):
    analysis: str
    answerable_question: str
    question: str
    answer: str

class CheckAnswerLanguage(BaseModel):
    analysis: str
    answer_language_matches_question_language: bool

class AnswerToClaims(BaseModel):
    answer_claims: list[str]

class PageFactCheck(BaseModel):
    analysis: str
    claims_supported: list[bool]
    low_quality: bool

class CheckClaims(BaseModel):
    analysis: str
    claims_supported: list[bool]
    corrected_answer: str | None

class CheckClaimsMMLongDoc(BaseModel):
    analysis: str
    claims_supported: list[bool]
    corrected_question: str | None
    corrected_answer: str | int | float | list | None
    @model_validator(mode='after')
    def cast_answer_to_str(self) -> 'CheckClaimsMMLongDoc':
        if not isinstance(self.corrected_answer, (str, type(None))):
            self.corrected_answer = str(self.corrected_answer)
        return self


class Persona(CoT):
    persona: str

class SBPScenario(BaseModel):
    title: str
    actor_role: str
    org_context: str
    trigger_event: str
    objective: str
    constraints: list[str]
    time_pressure: Literal['low', 'medium', 'high']
    evidence_signals_from_page: list[str]

class SBPScenarios(BaseModel):
    scenarios: list[SBPScenario]

class SBPPersona(BaseModel):
    name: str
    role: str
    industry: str
    org_size: Literal['startup', 'smb', 'mid', 'enterprise', 'public-sector', 'ngo']
    seniority: Literal['junior', 'mid', 'senior', 'exec']
    domain_knowledge_level: Literal['novice', 'competent', 'expert']
    current_goal: str
    longer_term_goal: str
    pain_points: list[str]
    decision_factors: list[str]
    constraints: list[str]
    info_seeking_style: list[str]
    communication_tone: list[str]
    question_motives: list[str]
    distance_from_page: float

class SBPSelectAndBackcast(BaseModel):
    chosen_scenario_title: str
    persona: SBPPersona

class SBPAudit(BaseModel):
    page_specificity_risks: list[str]
    abstraction_moves: list[str]
    generality_score: int
    rewrite_guidance: str

class SBPRewritePersona(BaseModel):
    persona: SBPPersona

class PersonaText(BaseModel):
    persona: str

class UJSMicroPersona(BaseModel):
    role: str
    goal: str
    concerns: list[str]
    knowledge: Literal['novice', 'competent', 'expert']

class UJSMicroPersonas(BaseModel):
    before: UJSMicroPersona
    during: UJSMicroPersona
    after: UJSMicroPersona

class UJSPersona(BaseModel):
    name: str
    role: str
    industry: str
    journey_summary: str
    goals_now: list[str]
    success_criteria: list[str]
    risks: list[str]
    collaborators: list[str]
    communication_style: list[str]
    question_motives: list[str]

class UJSConstraints(BaseModel):
    constraints: list[str]
    preferences: list[str]
    forbidden_page_ties: list[str]
