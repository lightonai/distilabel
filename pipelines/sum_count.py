import random
from pathlib import Path

from distilabel.pipeline import Pipeline
from datasets import load_from_disk, Dataset

from distilabel.steps import (
    StepResources, 
    LoadDataFromDataset,
    FilterRows,
    ListToRows,
    NoOp,
    ConcatenateBranches,
    Map,
)
from distilabel.steps.tasks import (
    Task,
    LMGenerationTask,
)
from distilabel.prompt_sampler import PromptSampler

from distilabel.utils.pipelines.sum_count_utils import format_distiset
from distilabel.pydantics import Config
from distilabel import utils
import distilabel.utils.pipe_utils as pipe_utils

from distilabel.configs.sum_count import (
    config, 
    PDF_ROOT, 
    sum_count_prompt_sampler_config, 
    DS_PATH,
    CACHE_DIR,
    EXCLUDE_PDFS,
)

def get_ds(dataset: Dataset, n: int) -> Dataset:
    '''
    Get a dataset of n random pdfs, selecting all pages from each
    '''
    fn_to_page_count = utils.count_all_pages(
        pdf_root=PDF_ROOT,
        cache_dir=CACHE_DIR,
        n_jobs=64,
    )
    pdfs = random.sample(list([k for k, v in fn_to_page_count.items() if v < 100]), n // 2)
    hn_row_idxs = random.sample(range(len(dataset)), n // 2)

    # sample one system prompt for each pdf
    prompt_sampler = PromptSampler(
        sum_count_prompt_sampler_config, 
        Path(config.stages[0].default_system_template_path).read_text(),
    )
    system_prompts = [prompt_sampler.generate_prompt() for _ in range(len(pdfs) + len(hn_row_idxs))]

    ifn_col = dataset['image_filename']
    def sample_max_n_neg_fns(row_idx: int, n: int) -> list[str]:
        row = dataset[row_idx]
        negs = row['hard_negatives_idx_img_img'] + row['hard_negatives_idx_txt_img']
        negs = random.sample(negs, k=random.randint(min(5, n), min(n, len(negs))))
        return [ifn_col[neg] for neg in negs]

    hn_rows = []
    for system_prompt, row_idx, row_id in zip(system_prompts[len(pdfs):], hn_row_idxs, range(len(pdfs), len(pdfs) + len(hn_row_idxs))):
        hn_rows.append({
            'source': [ifn_col[row_idx]],
            'count_system': system_prompt,
            'row_id': row_id,
        })
        hn_rows.extend([
            {
                'source': [neg],
                'count_system': system_prompt,
                'row_id': row_id,
            }
            for neg in sample_max_n_neg_fns(row_idx, 63)
        ])


    dataset = Dataset.from_list(
        [
            {'source': [utils.page_path(pdf, i)], 'count_system': system_prompt, 'row_id': row_id}
            for system_prompt, pdf, row_id in zip(system_prompts, pdfs, range(len(pdfs)))
            for i in range(fn_to_page_count[pdf])
        ] + hn_rows
    )

    return dataset

STAGE = 0
'''tracks the current stage of the pipeline'''
BATCH_SIZE = 256

def run_pipeline(config: Config):
    global STAGE
    random.seed(1)
    
    stages = config.stages
    # assuming they share the same prompt because I am manually sampling system prompts in get_ds
    assert len(stages) == 1 and (lm_configs.system_template_path == stages[0].default_system_template_path for lm_configs in stages[0].lm_configs)

    dataset = load_from_disk(DS_PATH)
    # dataset = get_ds(2_000_000)
    dataset = get_ds(dataset, 2)
    dataset = utils.remove_pdfs_from_dataset(dataset, EXCLUDE_PDFS)

    with Pipeline(
        name='sum_count',
        description=(
            'Count the number of a given object in the page. This is used to make the task: count the number of objects in the document.'
        ),
        cache_dir=Path(CACHE_DIR) / 'sum_count',
    ) as pipeline:
        ################## STAGE 0: TRANSCRIBE THE PAGE ##################
        stage = stages[STAGE]
        load_data = LoadDataFromDataset(name="load_data", dataset=dataset, batch_size=BATCH_SIZE)

        lms = pipe_utils.make_lms(config, stage, use_cache=False)
        generate_count = [
            LMGenerationTask(
                name=f"count_{i}",
                stage=stage,
                llm=lm,
                lm_config=lm.lm_config,
                system_col='count_system',
                input_formatter=lm.format_input,
                parallel_input_formatter=lm.parallel_format_inputs,
                input_batch_size=BATCH_SIZE,
                resources=StepResources(replicas=lm.lm_config.replicas, gpus=lm.lm_config.tp_size),
                output_mappings={'model_name': 'count_model_name'},
                invalidate_cache=True,
                **lm.lm_config.task_kwargs,
            )
            for i, lm in enumerate(lms)
        ]
        filter_count = FilterRows(
            name='filter_count',
            cols=['count'],
            condition=utils.logical_and_filters(
                utils.not_empty_string,
                utils.generation_is_structured,
            ),
            input_batch_size=BATCH_SIZE,
        )

        ## Pipeline
        load_data >> generate_count >> filter_count

    distiset = pipeline.run(
        load_groups=(
            pipe_utils.steps_to_load_groups(
                [
                    load_data,
                    *generate_count,
                    filter_count,
                ],
                len(stage.available_gpus),
            )
        ),
        use_cache=True,
        invalidate_distiset=False,
    )
    return distiset

if __name__ == '__main__':
    distiset: Dataset = run_pipeline(config)['default']['train']
    distiset = distiset.remove_columns(['distilabel_metadata'])  # don't need this for this pipeline
    distiset = format_distiset(distiset)
    distiset.save_to_disk(Path(CACHE_DIR) / 'sum_count_ds')
