from typing import Dict, Tuple
from torch.utils.data import ConcatDataset, Dataset 
from ..models.prepare_processors import get_processor
from .ja_csv_datasets import JapaneseCSVDataset
from .llava_datasets import LlavaDataset
from .m3it_datasets import M3ITDataset
from my_custom_dataset import MyCSVDataset
import yaml

def get_each_dataset(dataset_config: Dict, processor, max_length: int) -> Tuple[Dataset, Dataset]:
    if dataset_config["dataset_type"] == "m3it":
        train_dataset = M3ITDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = M3ITDataset.create(dataset_config, processor, max_length, "validation")
    elif dataset_config["dataset_type"] == "japanese_csv":
        train_dataset = JapaneseCSVDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = JapaneseCSVDataset.create(dataset_config, processor, max_length, "validation")
    elif dataset_config["dataset_type"] == "llava":
        train_dataset = LlavaDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = LlavaDataset.create(dataset_config, processor, max_length, "validation")
    elif dataset_config["dataset_type"] == "my_csv":
        train_dataset = MyCSVDataset.create(dataset_config, processor, max_length, "train")
        val_dataset = MyCSVDataset.create(dataset_config, processor, max_length, "validation")
    else:
        raise ValueError(f"dataset_type: {dataset_config['dataset_type']} is not supported.")

    return train_dataset, val_dataset



def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    processor = get_processor(config["model_config"])
    train_dataset_list = []
    val_dataset_list = []
    max_length = config["model_config"]["max_length"]

    for dataset_config_path in config["dataset_config_path"]:
        try:
            with open(dataset_config_path, "r") as f:
                dataset_config = yaml.safe_load(f)
            train_dataset, val_dataset = get_each_dataset(dataset_config, processor, max_length)
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
        except IsADirectoryError:
            print(f"Error: {dataset_config_path} is a directory, not a file.")

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    return train_dataset, val_dataset
