from datasets import load_from_disk
from distilabel import utils

from distilabel.configs.lc_sft.synthetic_cot import (
    config,
    SP_DS_PATH,
    CACHE_DIR,
    IMAGES_DS_PATH,
)

from distilabel.pipelines.lc_sft.synthetic_cot import run_pipeline


if __name__ == '__main__':
    cols_to_keep = ['source', 'question', 'split', 'question_model_name', 'evidence', 'answer', 'answer_model_name']
    sp_ds_dict = load_from_disk(SP_DS_PATH)

    # input ds doesn't matter, just getting the cached distiset
    distiset, cost_tracker = run_pipeline(config, sp_ds_dict['distractors_short'])
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    # format to vision generic
    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split'], 
        n_workers=16,
    )

    # split between short and long
    recursive_hn = distiset.filter(
        utils.hf_batched(
            lambda row: 'recursive_hn' in row['split']
        ),
        batched=True,
        num_proc=1,
    )
    recursive_doc = distiset.filter(
        utils.hf_batched(
            lambda row: 'recursive_doc' in row['split']
        ),
        batched=True,
        num_proc=1,
    )

    recursive_hn.save_to_disk(CACHE_DIR / 'recursive_hn_vds')
    recursive_doc.save_to_disk(CACHE_DIR / 'recursive_doc_vds')
