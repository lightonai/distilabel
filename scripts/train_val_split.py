"""
Script to split the dataset splits in out/mp_synthetic_data_vis_format into train and val sections.
"""

from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
import random

def split_dataset_into_train_val(
    dataset: Dataset, 
    val_samples: int, 
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation with a fixed number of validation samples.
    
    Args:
        dataset: The dataset to split
        val_samples: Number of samples to use for validation
        seed: Random seed for reproducible splits
        
    Returns:
        tuple of (train_dataset, val_dataset)
        
    Raises:
        ValueError: If dataset doesn't have enough samples for the requested validation size
    """
    total_samples = len(dataset)
    
    if val_samples >= total_samples:
        raise ValueError(
            f"Not enough samples in dataset. Requested {val_samples} validation samples "
            f"but dataset only has {total_samples} total samples."
        )
    
    random.seed(seed)
    
    all_indices = list(range(total_samples))
    val_indices = random.sample(all_indices, val_samples)
    train_indices = list(set(all_indices) - set(val_indices))
    
    return dataset.select(train_indices), dataset.select(val_indices)

def create_train_val_splits(input_path: str, output_path: str, val_samples: int, seed: int = 42):
    """
    Split all splits in a dataset directory into train and validation sections.
    
    Args:
        input_path: Path to the input dataset directory
        output_path: Path where the new dataset with train/val splits will be saved
        val_samples: Number of samples to use for validation in each split
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
        train_data, val_data = split_dataset_into_train_val(dataset[split_name], val_samples, seed)
        train[split_name] = train_data
        val[split_name] = val_data
    
    # Save the train and val datasets
    train.save_to_disk(f'{output_path}_train')
    val.save_to_disk(f'{output_path}_val')

def main():
    INPUT_PATH = "out/mp_synthetic_data_vis_format"
    OUTPUT_PATH = "out/mp_synthetic_data_vis_format"
    VAL_SAMPLES = 20  # Number of samples to use for validation in each split
    SEED = 0  # For reproducible splits
    
    create_train_val_splits(INPUT_PATH, OUTPUT_PATH, VAL_SAMPLES, SEED)

if __name__ == "__main__":
    main()
