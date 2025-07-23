import os
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, cast
from pydantic import ValidationError, Field, PrivateAttr

import openai
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from distilabel.models.llms import OpenAILLM, vLLMAPI
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.typing import ChatType, FormattedInput, GenerateOutput

from distilabel import utils
from distilabel.pydantics import LMConfig, Stage
from distilabel.prompt_sampler import PromptSampler
from distilabel.constants import STRUCTURED_OUTPUT_RETRIES
from .lm_cache import get_lm_cache
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def _format_one_input(args) -> 'ChatType':
    (
        input,
        system_prompt,
        lm_input_cols,
        lm_input_col_prefixes,
        max_dims,
        msg_content_img_func,
        logger,
    ) = args
    messages = [{'role': 'system', 'content': system_prompt}]

    if not all(input.get(col) is not None for col in lm_input_cols + ['source']):
        # some assurance that the values we want to use exist in input
        # generation will be skipped in LMGenerationTask for this kind of input
        logger.warning(
            f"Skipping generation because some required columns are missing from {lm_input_cols + ['source']}\n"
            f"Input: {input}"
        )
        return [{'role': 'system', 'content': ''}]

    messages.append(utils.source_to_msg(input['source'], max_dims, msg_content_img_func))
    
    if len(lm_input_col_prefixes) == 0:
        lm_input_col_prefixes = [''] * len(lm_input_cols)
    for col, prefix in zip(lm_input_cols, lm_input_col_prefixes):
        message = utils.source_to_msg(input[col], max_dims, msg_content_img_func)
        if isinstance(message['content'], str):
            message['content'] = prefix + message['content']
        else:
            messages.append({'role': 'user', 'content': prefix})
        messages.append(message)
    
    return messages

class VLM:
    stage: Stage = Field(default_factory=Stage, exclude=True)
    lm_config: LMConfig = Field(default_factory=LMConfig, exclude=True)
    prompt_sampler: PromptSampler | None = None
    debug_with_running_vllm: bool = Field(default=False, exclude=True)

    _vlm_logger = logging.getLogger(f"distilabel.vlm")
    _executor: "ProcessPoolExecutor | None" = PrivateAttr(default=None)

    def format_input(self, input: dict, lm_input_cols: list[str], lm_input_col_prefixes: list[str]) -> 'ChatType':
        system = self.prompt_sampler.generate_prompt()
        input |= {'system': system}  # inplace update the input to sneak it into the format_output of LMGenerationTask
        return _format_one_input((
            input,
            system,
            lm_input_cols,
            lm_input_col_prefixes,
            self.stage.max_dims,
            VLM.msg_content_img,
            self._vlm_logger,
        ))

    def parallel_format_inputs(self, inputs: list[dict], lm_input_cols: list[str], lm_input_col_prefixes: list[str]) -> list['ChatType']:
        '''
        Format input serially is a big bottleneck due probably to image loading. Parallelizing this is great for throughput.
        '''
        prompts = [self.prompt_sampler.generate_prompt() for _ in inputs]
        for inp, system in zip(inputs, prompts):
            inp |= {'system': system}  # inplace update the input to sneak it into the format_output of LMGenerationTask
        
        tasks = [
            (
                input_data,
                system_prompt,
                lm_input_cols,
                lm_input_col_prefixes,
                self.stage.max_dims,
                VLM.msg_content_img,
                self._vlm_logger
            )
            for input_data, system_prompt in zip(inputs, prompts)
        ]
        
        if self._executor is None:
            raise DistilabelError("ProcessPoolExecutor not initialized. Make sure to call `load()`.")

        return list(self._executor.map(_format_one_input, tasks))

    def load(self):
        self.prompt_sampler = PromptSampler(self.lm_config.prompt_sampler_config, self.lm_config.system_template)
        self._executor = ProcessPoolExecutor(max_workers=min(max(4, cpu_count()), 32))
    
    def unload(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    @staticmethod
    def msg_content_img(b64_img):
        """Convert base64 image to appropriate format. To be implemented by subclasses."""
        return utils.msg_content_img_url(b64_img)

def lm_cache(agenerate: Callable) -> Callable:
    '''
    Decorator for agenerate methods to handle caching of input -> output.
    
    Uses a SQLite database for efficient storage instead of individual JSON files.
    Maps all input parameters to a GenerateOutput object using a hash-based cache.
    The cache is stored in self.pipeline._cache_location['lm_cache'] as a SQLite database.
    
    Behavior:
    - Does not read/write cache if self.use_cache is False
    - Overwrites cache (without reading) if self.invalidate_cache is True
    - Creates cache directory if it doesn't exist
    - Uses SQLite for efficient storage and fast updates
    
    This decorator should be applied before other decorators that modify the function signature.
    
    Returns:
        GenerateOutput: The cached or newly generated output containing generations,
                       statistics, and optional logprobs.
    '''
    async def agenerate_cached(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerateOutput:
        self = cast(OpenAILM, self)
        # Skip caching if use_cache is False
        if not getattr(self, 'use_cache', True):
            return await agenerate(
                self,
                input=input,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
        
        # Create cache key from all parameters
        cache_params = {
            'input': input,
            'num_generations': num_generations,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'extra_body': extra_body,
            'model_name': getattr(self, 'model_name', 'unknown'),
        }
        
        # Get cache instance
        cache_dir = self.lm_config.lm_response_cache_root
        lm_cache_db = get_lm_cache(cache_dir)
        
        # Check if we should read from cache
        should_read_cache = not getattr(self, 'invalidate_cache', False)
        
        if should_read_cache:
            cached_response = lm_cache_db.get(cache_params)
            if cached_response is not None:
                self._logger.debug(f"🔍 Cache hit for LM {self.lm_config.path}")
                return cached_response
        
        # Generate new result
        result = await agenerate(
            self,
            input=input,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        
        # Save to cache
        lm_cache_db.set(cache_params, result)
        
        return result
    
    return agenerate_cached

def multiple_generations(agenerate: Callable) -> Callable:
    '''
    Decorator for agenerate methods to handle multiple generations
    '''
    async def agenerate_multiple(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerateOutput:
        results = []
        for _ in range(num_generations):
            result = await agenerate(
                self,
                input=input,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            results.append(result)
        
        return GenerateOutput(
            generations=[result['generations'][0] for result in results],
            statistics={
                'input_tokens': [result['statistics']['input_tokens'] for result in results],
                'output_tokens': [result['statistics']['output_tokens'] for result in results],
            },
        )
    
    return agenerate_multiple

def structured_output(agenerate: Callable) -> Callable:
    '''
    Decorator for agenerate methods to handle retries and structured output 
    according to the pydantic schema in self.lm_config.out_model

    Must come before multiple_generations because it expects only a single generation
    '''
    async def agenerate_structured(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerateOutput:
        self = cast(OpenAILM, self)
        for _ in range(STRUCTURED_OUTPUT_RETRIES):
            ## call the wrapped agenerate
            generate_output = await agenerate(
                self,
                input=input,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            if self.lm_config.out_model is None:  # allow for no pydantic model
                return generate_output
            try:
                # assume only one generation
                generate_output['generations'][0] = self.lm_config.out_model.model_validate_json(
                    utils.clean_structured_output(generate_output['generations'][0]), 
                    strict=True
                ).model_dump_json()
                # if your pydantic model allows extra fields, no worries, they will be dropped

                return generate_output
            except ValidationError:
                continue
        
        return GenerateOutput(
            generations=[None],
            statistics={'input_tokens': [0], 'output_tokens': [0]},
        )

    return agenerate_structured

class OpenAILM(OpenAILLM, CudaDevicePlacementMixin, VLM):
    '''OpenAILLM wrapper for handling images 
    
    OpenAI is the default client

    vLLM is supported by setting use_vllm=True in the model config

    Grok is supported with 'grok' in the model path

    Gemini is supported with 'gemini' in the model path
    
    Anthropic is supported with 'claude' in the model path
    '''
    use_vllm: bool = False
    use_cache: bool = True
    invalidate_cache: bool = False
    _vllm_api: vLLMAPI = PrivateAttr(None)

    def load(self):
        # self.base_url = kwargs.get('base_url', None)

        if utils.is_openai_model_name(self.lm_config.path):
            self.base_url = None
            self.api_key = os.getenv('OPENAI_API_KEY')
        elif 'claude' in self.lm_config.path:
            self.base_url = 'https://api.anthropic.com/v1/'
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
        elif 'gemini' in self.lm_config.path:
            self.base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            self.api_key = os.getenv('GEMINI_API_KEY')
        elif 'grok' in self.lm_config.path:
            self.base_url = 'https://api.x.ai/v1'
            self.api_key = os.getenv('XAI_API_KEY')
        else:
            self.use_vllm = True

            self._vllm_api = vLLMAPI(self.lm_config)
            self.base_url = f'http://localhost:{self._vllm_api.port}/v1'
            if self.debug_with_running_vllm:
                self.base_url = 'http://localhost:8000/v1'

        # must come after the base_url is set properly
        super().load()
        VLM.load(self)
        self.disable_cuda_device_placement = not self.lm_config.tp_size
        CudaDevicePlacementMixin.load(self)
        
        # must come after CUDA_VISIBLE_DEVICES is set properly
        if self.use_vllm and not self.debug_with_running_vllm:
            # I want this to throw an error because the visible devices should be set by the placement mixin
            gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
            self._vllm_api.gpu = gpu
            self._vllm_api.start_vllm()

    def _assign_cuda_devices(self):
        '''Override the default cuda device assignment to only assign to the available gpus'''
        self._available_cuda_devices = self.stage.available_gpus
        super()._assign_cuda_devices()

    @staticmethod
    def msg_content_img(b64_img):
        return utils.msg_content_img_url(b64_img)

    @lm_cache
    @multiple_generations
    @structured_output
    async def agenerate(
        self, 
        input: FormattedInput,
        num_generations: int = 1,  # handled by the decorator
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> 'GenerateOutput':
        '''
        Wrapping to ignore unhandled parameters by other client's openai compatible APIs
        
        These include frequency_penalty, presence_penalty, stop, response_format, maybe others
        '''
        no_response = GenerateOutput(
            generations=[None],
            statistics={'input_tokens': [0], 'output_tokens': [0]},
        )
        # in case previous steps somehow gave empty inputs
        if len(input) == 0 or len(input) == 1 and input[0]['content'] in [None, '']:  # nothing for lm to respond to
            return no_response
        
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True, retry_error_callback=openai.RateLimitError)
        async def _generate():
            completion = await self._aclient.chat.completions.create(
                model=self.model_name,
                messages=input,
                max_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            return completion
        
        try:
            completion = await _generate()
        except Exception as e:
            completion = None
        if completion is None:
            return no_response
        return self._generations_from_openai_completion(completion)

    # also cleanup vLLM
    def unload(self) -> None:
        if self.use_vllm and self._vllm_api:
            self._vllm_api.cleanup()
        super().unload()
        CudaDevicePlacementMixin.unload(self)
        VLM.unload(self)
