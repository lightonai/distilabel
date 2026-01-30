import os
from httpx import URL
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
from distilabel.models.mixins.vllm_server_placement import VLLMServerPlacementMixin
from distilabel.models.mixins.vllm_server_sharing import VLLMServerSharingMixin
from distilabel.typing import ChatType, FormattedInput, GenerateOutput

from distilabel import utils
from distilabel.pydantics import LMConfig, Stage
from distilabel.prompt_sampler import PromptSampler
from distilabel.constants import STRUCTURED_OUTPUT_RETRIES
from .lm_cache import get_lm_cache
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import cpu_count

def _format_one_input(args) -> 'ChatType':
    (
        input,
        system_prompt,
        lm_input_cols,
        lm_input_col_prefixes,
        path_substitution,
        max_dims,
        msg_content_img_func,
        postprocess_image_hook,
        logger,
    ) = args
    messages = [] if system_prompt == 'default_system' else [{'role': 'system', 'content': system_prompt}]

    try:
        # converts the source col into a message in openai format (does downsampling and base64 encoding as well)
        messages.append(
            utils.source_to_msg(
                input['source'], 
                max_dims, 
                msg_content_img_func, 
                path_substitution,
                postprocess_image_hook,
                'pdfium',
            )
        )
        
        # 1. I allow prefixes for the lm_input_cols, so that you can say: 'description: {description_col}' or whatever
        # 2. This code simply converts lm_input_cols into messages and adds the prefix
        # also make this one user section because some models complain when roles don't alternate
        prefixes_content = []
        if len(lm_input_col_prefixes) == 0:
            lm_input_col_prefixes = [''] * len(lm_input_cols)
        for col, prefix in zip(lm_input_cols, lm_input_col_prefixes):
            if input[col] is None:
                continue
            message = utils.source_to_msg(
                input[col], 
                max_dims, 
                msg_content_img_func, 
                path_substitution,
                postprocess_image_hook,
                'pdfium',
            )
            if isinstance(message['content'], str):
                prefixes_content.append({'type': 'text', 'text': prefix + message['content'] + '\n'})
            else:
                prefixes_content.append({'type': 'text', 'text': prefix})
                prefixes_content.extend(message['content'])

        if len(prefixes_content) == 0:
            return messages

        if isinstance(messages[-1]['content'], list):
            messages[-1]['content'].extend(prefixes_content)
        elif isinstance(messages[-1]['content'], str):
            messages[-1]['content'] = [{'type': 'text', 'text': messages[-1]['content']}, *prefixes_content]
        else:
            raise ValueError(f"Invalid content type: {type(messages[-1]['content'])}")

        return messages
    except Exception as e:
        logger.warning(f"Error converting source to message, skipping generation for this sample:\n{e}\n{input}")
        return [{'role': 'system', 'content': ''}]

class VLM:
    stage: Stage = Field(default_factory=Stage, exclude=True)
    lm_config: LMConfig = Field(default_factory=LMConfig, exclude=True)
    prompt_sampler: PromptSampler | None = None
    use_running_vllm: bool = Field(default=False, exclude=True)

    _vlm_logger = logging.getLogger(f"distilabel.vlm")
    _executor: "ProcessPoolExecutor | None" = PrivateAttr(default=None)

    def format_input(self, input: dict, system_col: str | None, lm_input_cols: list[str], lm_input_col_prefixes: list[str]) -> 'ChatType':
        if system_col == 'default_system':
            system = 'default_system'
        elif system_col is None:
            system = self.prompt_sampler.generate_prompt(
                seed=utils.hash_structure_with_images(input)
            )
        else:
            system = input[system_col]
        input |= {'system': system}  # inplace update the input to sneak it into the format_output of LMGenerationTask
        return _format_one_input((
            input,
            system,
            lm_input_cols,
            lm_input_col_prefixes,
            self.lm_config._path_substitution,
            self.stage.max_dims,
            VLM.msg_content_img,
            self.lm_config.postprocess_image_hook,
            self._vlm_logger,
        ))

    def parallel_format_inputs(
        self,
        inputs: list[dict],
        system_col: str | None,
        lm_input_cols: list[str],
        lm_input_col_prefixes: list[str],
    ) -> list['ChatType']:
        '''
        Format input serially is a big bottleneck due probably to image loading. Parallelizing this is great for throughput.
        '''
        if system_col == 'default_system':
            prompts = ['default_system'] * len(inputs)
        else:
            prompts = [
                self.prompt_sampler.generate_prompt(seed=utils.hash_structure_with_images(inputs[i])) 
                if system_col is None else inputs[i][system_col] 
                for i in range(len(inputs))
            ]
        for inp, system in zip(inputs, prompts):
            inp |= {'system': system}  # inplace update the input to sneak it into the format_output of LMGenerationTask
        
        tasks = [
            (
                input_data,
                system_prompt,
                lm_input_cols,
                lm_input_col_prefixes,
                self.lm_config._path_substitution,
                self.stage.max_dims,
                VLM.msg_content_img,
                self.lm_config.postprocess_image_hook,
                self._vlm_logger
            )
            for input_data, system_prompt in zip(inputs, prompts)
        ]

        try:
            return list(self._executor.map(_format_one_input, tasks))
        except Exception as e:  # try to restart the pool and continue
            try:
                self._executor.shutdown(wait=False)
                del self._executor
                self._executor = ProcessPoolExecutor(max_workers=min(max(4, cpu_count()), 32))
                return list(self._executor.map(_format_one_input, tasks))
            except Exception as e:
                # common cause of it happening reproducibly is pdfium segfaulting so just move on
                self._vlm_logger.error(f"Error despite restarting executor: {e}")
                self._executor.shutdown(wait=False)
                del self._executor
                self._executor = ProcessPoolExecutor(max_workers=min(max(4, cpu_count()), 32))
                return [[{'role': 'system', 'content': ''}] for _ in range(len(tasks))]

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
            if cached_response is not None and any(g is not None for g in cached_response['generations']):
                self._logger.debug(f"🔍 Cache hit for LM {self.lm_config.path}")
                cached_response['cache_hit'] = [True] * len(cached_response['generations'])
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
            reasoning_generations=[result['reasoning_generations'][0] for result in results],
            statistics={
                'input_tokens': [result['statistics']['input_tokens'] for result in results],
                'output_tokens': [result['statistics']['output_tokens'] for result in results],
            },
            cache_hit=[False] * len(results),
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
        format_exc = None
        input_toks, output_toks = 0, 0
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
            input_toks += generate_output['statistics']['input_tokens'][0]
            output_toks += generate_output['statistics']['output_tokens'][0]
            if self.lm_config.out_model is None:  # allow for no pydantic model
                return generate_output
            try:
                # assume only one generation
                generate_output['generations'][0] = utils.try_model_validate(
                    self.lm_config.out_model,
                    generate_output['generations'][0],
                    strict=True,
                ).model_dump_json()
                # if your pydantic model allows extra fields, no worries, they will be dropped

                generate_output['statistics']['input_tokens'] = [input_toks]
                generate_output['statistics']['output_tokens'] = [output_toks]
                return generate_output
            except openai.APIConnectionError as e:
                raise e
            except Exception as e:
                format_exc = e
                continue
        
        self._logger.warning(f"Failed to format structured output, last example: {generate_output['generations'][0]}\n{format_exc}")
        return GenerateOutput(
            generations=[None],
            reasoning_generations=[None],
            statistics={'input_tokens': [0], 'output_tokens': [0]},
            cache_hit=[False],
        )

    return agenerate_structured

class OpenAILM(OpenAILLM, CudaDevicePlacementMixin, VLLMServerPlacementMixin, VLLMServerSharingMixin, VLM):
    '''OpenAILLM wrapper for handling images 
    
    OpenAI is the default client

    vLLM is supported by using a model path that is not recognized as proprietary

    Grok is supported with 'grok' in the model path

    Gemini is supported with 'gemini' in the model path
    
    Anthropic is supported with 'claude' in the model path
    '''
    use_vllm: bool = False
    use_cache: bool = True
    invalidate_cache: bool = False
    api_call_extra_body: dict[str, Any] = Field(default_factory=dict)
    _vllm_api: vLLMAPI = PrivateAttr(None)

    def load(self):
        self._model_path = self.lm_config.path

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
            # round robin distribute LLMs to vLLM servers listed in VLLM_BASE_URLS_JSON if using running vLLM
            if self.use_running_vllm:
                # Enable placement only if VLLM_BASE_URLS_JSON is present
                self.disable_vllm_server_placement = os.environ.get('VLLM_BASE_URLS_JSON') is None
                VLLMServerPlacementMixin.load(self)
                base_url = os.environ.get('VLLM_API_BASE_URL', 'http://localhost:8000')
                self.base_url = f'{base_url}/v1'

        # Initialize clients and prompt sampler
        super().load()
        VLM.load(self)

        # CUDA placement depends on server ownership for vLLM
        self.disable_cuda_device_placement = not (self.lm_config.tp_size or self.lm_config.pp_size)

        if self.use_vllm and not self.use_running_vllm:
            # Perform ownership election BEFORE assigning GPUs
            self.disable_vllm_server_sharing = self.lm_config.replicas_per_vllm_server <= 1
            self.replicas_per_vllm_server = self.lm_config.replicas_per_vllm_server
            self.model_path = self.lm_config.path
            VLLMServerSharingMixin.load(self)

            # Only the elected owner should request GPUs
            if not self._server_info['is_owner']:
                # non-owners should not request GPUs
                try:
                    self._desired_num_gpus = 0  # type: ignore[attr-defined]
                except Exception:
                    self.disable_cuda_device_placement = True

            # Now assign CUDA devices accordingly
            CudaDevicePlacementMixin.load(self)

            if self._server_info['is_owner']:
                # Owner sets its GPU index and starts vLLM
                gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
                self._vllm_api.gpu = gpu
                self._vllm_api.start_vllm()
                VLLMServerSharingMixin.update_server_info(
                    self, self._vllm_api.port, self._vllm_api.vllm_server_pid
                )
                self.base_url = f'http://localhost:{self._vllm_api.port}/v1'
            else:
                # Join existing server; no GPUs needed
                ready_port = self._server_info['port']
                ready_pid = self._server_info['pid']
                self._vllm_api.gpu = self._replica_id
                self._vllm_api.wait_for_existing_server(ready_port, ready_pid)
                self.base_url = f'http://localhost:{self._vllm_api.port}/v1'
        else:
            # Non-vLLM or running vLLM, do CUDA placement as before
            CudaDevicePlacementMixin.load(self)

        self._aclient.base_url = self._aclient._enforce_trailing_slash(URL(self.base_url))

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
            reasoning_generations=[None],
            statistics={'input_tokens': [0], 'output_tokens': [0]},
            cache_hit=[False],
        )
        # in case previous steps somehow gave empty inputs
        if len(input) == 0 or len(input) == 1 and input[0]['content'] in [None, '']:  # nothing for lm to respond to
            return no_response
        
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True)
        async def _generate():
            completion = await self._aclient.chat.completions.create(
                model=self.model_name,
                messages=input,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                extra_body=extra_body or {} | self.api_call_extra_body,
                timeout=1800,
            )
            return completion
        
        try:
            completion = await _generate()
        except openai.APIConnectionError as e:
            self._logger.warning(f"Failed to get client response, exception: {e}")
            raise e
        except openai.BadRequestError as e:
            self._logger.warning(f"Failed to get client response, bad request error: {e}\n{input=}")
            raise e
        except Exception as e:
            completion = None
            self._logger.warning(f"Failed to get client response, exception: {e}")
        if completion is None:
            return no_response
        return self._generations_from_openai_completion(completion)

    # also cleanup vLLM
    def unload(self) -> None:
        # Check if we should cleanup the vLLM server
        should_cleanup_vllm = True
        if self.use_vllm and not self.use_running_vllm:
            # Only cleanup if this is the last replica using the shared server
            VLLMServerSharingMixin.unload(self)
            should_cleanup_vllm = self._cleanup
        
        if should_cleanup_vllm and self.use_vllm and self._vllm_api:
            self._vllm_api.cleanup()
        
        super().unload()
        CudaDevicePlacementMixin.unload(self)
        VLM.unload(self)
