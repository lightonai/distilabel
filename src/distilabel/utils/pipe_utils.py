import random
from typing import Callable

from distilabel.pipeline import routing_batch_function
from distilabel.steps import Step

from distilabel.models.llms import OpenAILM
from distilabel.pydantics import Stage, Config

def steps_to_load_groups(steps: list[Step], n_gpus: int) -> list[list[str]]:
    '''
    Given a list of steps in an steps, break the steps into load_groups so that there are enough gpus for each load_group
    
    Return a list of lists of step names, where each inner list is a load_group in the pipeline
    '''
    load_groups = []
    load_group = []
    load_group_gpus = 0
    for step in steps:
        if not hasattr(step, 'resources') or step.resources.gpus is None:
            load_group.append(step.name)
            continue
        requested_gpus = step.resources.gpus * step.resources.replicas
        assert requested_gpus <= n_gpus, 'No single step can use more gpus than available'
        # Since I can't break up replicas into different load groups, you may need to make duplicate tasks so that they can be split
        if (load_group_gpus + requested_gpus) > n_gpus:
            load_groups.append(load_group)
            load_group = [step.name]
            load_group_gpus = requested_gpus
        else:
            load_group.append(step.name)
            load_group_gpus += requested_gpus
    load_groups.append(load_group)
    return load_groups

def data_router(step_distribution: list[float], k: int = 1, invalidate_cache: bool = False) -> Callable:
    '''
    Given a list of downstream steps and a distribution, per batch, sample k downstream steps to route to

    Use this as a step in a pipeline to e.g. split the data from a step to different language models

    Args:
        invalidate_cache: If true, reroute batches instead of using the cached routed_to.
            This is needed if you update the routing function but still want to resume progress
            where possible.
    '''
    @routing_batch_function(invalidate_cache=invalidate_cache)
    def router(steps: list[str]) -> list[str]:
        return random.choices(steps, weights=step_distribution, k=k)
    return router

def multi_branch_data_router(
    branch_configs: list[dict]
) -> Callable:
    '''
    Route data to multiple branches, with each branch having its own step distribution and sample size.
    
    Args:
        branch_configs: List of dicts, each containing:
            - step_names: List of step names belonging to this branch
            - distribution: Weights for steps within this branch
            - k: Number of steps to sample for this branch
            
    Returns:
        A router function that splits incoming steps by branch and samples from each branch
    '''
    # set of all step names
    step_map = {step_name for branch in branch_configs for step_name in branch['step_names']}
            
    @routing_batch_function()
    def router(steps: list[str]) -> list[str]:
        # verify we have all the same steps
        assert len(steps) == len(step_map) and all(step in step_map for step in steps)
        
        # sample from each branch
        result = []
        for branch in branch_configs:
            result.extend(
                random.choices(
                    branch['step_names'], 
                    weights=branch['distribution'], 
                    k=branch['k']
                )
            )
        return result
    
    return router

def make_lms(config: Config, stage: Stage, use_cache: bool = True) -> list[OpenAILM]:
    '''initialize lms for a stage'''
    return [
        OpenAILM(
            stage=stage, 
            lm_config=lm_config, 
            model=lm_config.path, 
            generation_kwargs={
                'temperature': lm_config.temperature, 
                'max_new_tokens': lm_config.max_new_tokens,
            },
            debug_with_running_vllm=config.debug_with_running_vllm,
            use_cache=use_cache,
        ) 
        for lm_config in stage.lm_configs
    ]
