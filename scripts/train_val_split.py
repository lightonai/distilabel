"""
Script to split the dataset splits in out/mp_synthetic_data_vis_format into train and val sections.
"""

from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict

def split_dataset_into_train_val(
    dataset: Dataset, 
    train_ratio: float = 0.8, 
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    # Use the built-in train_test_split method with consistent seed
    split_dataset = dataset.train_test_split(train_size=train_ratio, seed=seed)
    return split_dataset['train'], split_dataset['test']

def create_train_val_splits(input_path: str, output_path: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Split all splits in a dataset directory into train and validation sections.
    
    Args:
        input_path: Path to the input dataset directory
        output_path: Path where the new dataset with train/val splits will be saved
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducible splits
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load the dataset
    dataset = load_from_disk(str(input_path))
    splits = list(dataset.keys())
    
    train = DatasetDict()
    val = DatasetDict()
    
    # Process each split
    for split_name in splits:
        train_data, val_data = split_dataset_into_train_val(dataset[split_name], train_ratio, seed)
        train[split_name] = train_data
        val[split_name] = val_data
    
    # Save the train and val datasets
    train.save_to_disk(f'{output_path}_train')
    val.save_to_disk(f'{output_path}_val')

def main():
    INPUT_PATH = "out/mp_synthetic_data_vis_format"
    OUTPUT_PATH = "out/mp_synthetic_data_vis_format"
    TRAIN_RATIO = 0.8  # 80% train, 20% validation
    SEED = 0  # For reproducible splits
    
    create_train_val_splits(INPUT_PATH, OUTPUT_PATH, TRAIN_RATIO, SEED)

if __name__ == "__main__":
    main()
