import pandas as pd
from PIL import Image
from .base_datasets import BaseDataset

class MyCSVDataset(BaseDataset):
    def __init__(self, csv_path, is_inference=False):
        super(MyCSVDataset, self).__init__(is_inference)
        self.data = pd.read_csv(csv_path)

    @classmethod
    def create(cls, dataset_config, processor, max_length, split="train", is_inference=False):
        if split == "train":
            csv_path = dataset_config["train_csv_path"]
        elif split == "validation":
            csv_path = dataset_config["val_csv_path"]
        else:
            raise ValueError(f"Invalid split: {split}")
        return cls(csv_path, is_inference)
    
    def __len__(self):
        return len(self.data)   
        
    def _get_item_train(self, index):
        row = self.data.iloc[index]
        image_path = row["image_path"]
        text = row["text"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Process text and image as needed
        # ...

        return {
            "input_ids": ...,  # Processed text
            "labels": ...,     # Processed text (same as input_ids for language modeling)
            "attention_mask": ...,  # Attention mask for text
            "pixel_values": ...,  # Processed image
        }

    def _get_item_inference(self, index):
        row = self.data.iloc[index]
        image_path = row["image_path"]
        text = row["text"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Process text and image as needed
        # ...

        return {
            "input_ids": ...,  # Processed text
            "labels": None,    # No labels for inference
            "attention_mask": ...,  # Attention mask for text
            "pixel_values": ...,  # Processed image
        }

