import random
import os

from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDicts,
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)
from distilabel.steps import (
    NoOp,
)

from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils
from distilabel.pydantics import Config, Stage
from distilabel.models.llms import OpenAILM

from distilabel.configs.lm_filter_dataset import config

def add_hard_negs_back(dataset: Dataset, filtered_dataset: Dataset):
    '''
    Add hard negatives back to the dataset.
    '''
    filtered_dataset = filtered_dataset.remove_columns(['references_model_name']).rename_column('source', 'image_filename')
    filtered_dataset = filtered_dataset.map(lambda x: {'image_filename': x['image_filename'][0]}, num_proc=16)
    not_references = set(filtered_dataset['image_filename'])
    references = set(dataset['image_filename']) - not_references

    # cache the dataset_filename_to_idx since this is a slow operation
    path = 'out/lm_filter_extras/dataset_filename_to_idx.json'
    if not os.path.exists(path):
        dataset_filename_to_idx = utils.generate_field_to_idx(dataset, 'image_filename')
        utils.save_json(path, dataset_filename_to_idx)
    else:
        dataset_filename_to_idx = utils.load_json(path)
    filtered_dataset_filename_to_idx = utils.generate_field_to_idx(filtered_dataset, 'image_filename')

    dataset_idx_to_filtered_dataset_idx = {
        dataset_filename_to_idx[fn]: filtered_dataset_filename_to_idx[fn] for fn in not_references
    }

    filtered_dataset = filtered_dataset.map(
        lambda x: {
            'hard_negs_idx_img_img': [
                dataset_idx_to_filtered_dataset_idx[hni] 
                for hni in dataset[dataset_filename_to_idx[x['image_filename']]]['hard_negatives_idx_img_img'] 
                if hni in dataset_idx_to_filtered_dataset_idx
            ],
            'hard_negs_idx_txt_img': [
                dataset_idx_to_filtered_dataset_idx[hni] 
                for hni in dataset[dataset_filename_to_idx[x['image_filename']]]['hard_negatives_idx_txt_img'] 
                if hni in dataset_idx_to_filtered_dataset_idx
            ]
        }, num_proc=64
    )
    return filtered_dataset


STAGE = 0
'''tracks the current stage of the pipeline'''


def run_pipeline(config: Config):
    global STAGE
    random.seed(0)
    
    stages = config.stages
    dataset = load_from_disk('/mnt/nfs/dse/scraped_v0.3_with_txt_img_neg')
    # dataset = dataset.select(range(1_000_000))
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'image_filename'])
    dataset = dataset.rename_column('image_filename', 'source')
    dataset = dataset.map(lambda x: {'source': [x['source']]}, num_proc=16)

    with Pipeline(
        name="remove_reference_pages",
        description="Use a LM to find reference/bibliography pages in the scraped data and remove them.",
        cache_dir='out/remove_references',

    ) as pipeline:
        ################## STAGE 0 ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=256)  # cols: ['source', 'question', ...]
        data_router = pipe_utils.data_router(
            step_distribution=[lm_config.data_ratio for lm_config in stage.lm_configs]
        )
        lms = pipe_utils.make_lms(config, stage, use_cache=False)  # use_cache=False  # turn off lm level caching
        label_references = [
            LMGenerationTask(
                name=f"label_references_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=256,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'system': 'references_system', 'model_name': 'references_model_name'},
                # use_cache=False,  # turn off batch level caching
                **lm.lm_config.task_kwargs,
            ) 
            for i, lm in enumerate(lms)
        ]  # cols: ['source', ...] -> ['references_system', 'references_model_name', ...]
        drop_reference_pages = FilterRows(  # drop rows where the answer is None (structured output failed)
            name="drop_reference_pages",
            cols=['is_references_page'],
            condition=utils.logical_and_filters(
                utils.generation_is_structured,  # structured output successful
                utils.logical_not_filter(utils.cols_true),  # is not a references page
            ),
            input_batch_size=256,
        )  # cols: ['is_references_page', ...] -> ['is_references_page', ...]

        ## Pipeline
        (
            load_data >> data_router >> label_references >> drop_reference_pages
        )
    
    distiset = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(  # handles breaking up steps so each load_group has enough gpus
                # data_router is not included because it's not quite a step, but it does actually still run
                [load_data, *label_references, drop_reference_pages],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,  # False to turn off distiset level caching
        invalidate_distiset=False,
        use_fs_to_pass_data=False,  # True will keep data not being actively read/modified in the fsspec fs (which can be local disk or others)
    )
    return distiset

if __name__ == "__main__":
    distiset = run_pipeline(config)['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])  # don't need this for this pipeline

    dataset = load_from_disk('/mnt/nfs/dse/scraped_v0.3_with_txt_img_neg')
    distiset = add_hard_negs_back(dataset, distiset)
    
    if not os.path.exists('out/scraped_v0.3_with_txt_img_neg_references_filtered'):
        distiset.save_to_disk('out/scraped_v0.3_with_txt_img_neg_references_filtered')
    