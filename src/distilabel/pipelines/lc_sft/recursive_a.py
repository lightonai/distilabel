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
    cols_to_keep = ['source', 'question', 'split', 'question_model_name', 'evidence', 'answer', 'answer_model_name', 'question_source']
    sp_ds_dict = load_from_disk(SP_DS_PATH)
    sp_ds_dict['distractors_short'] = sp_ds_dict['distractors_short'].add_column('question_source', [None] * len(sp_ds_dict['distractors_short']))

    # input ds doesn't matter, just getting the cached distiset
    distiset, cost_tracker = run_pipeline(config, sp_ds_dict['distractors_short'])
    print(f"Cost: {dict(cost_tracker)}")
    distiset = distiset['default']['train']

    # format to vision generic
    distiset = utils.format_distiset(
        distiset, 
        images_ds_path=IMAGES_DS_PATH,
        path_substitution=config.path_substitution,
        cols_to_keep=['answer_model_name', 'split', 'question_source'], 
        n_workers=16,
    )

    recursive = distiset.filter(utils.hf_batched(lambda row: 'distractors' not in row['split']), batched=True)
    recursive.save_to_disk(CACHE_DIR / 'recursive_vds')
    # sp = distiset.filter(utils.hf_batched(lambda row: 'sp' in row['split']), batched=True)
    # sp.save_to_disk(CACHE_DIR / 'recursive_sp_vds')
    # mp = distiset.filter(utils.hf_batched(lambda row: 'mp' in row['split']), batched=True)
    # mp.save_to_disk(CACHE_DIR / 'recursive_mp_vds')
