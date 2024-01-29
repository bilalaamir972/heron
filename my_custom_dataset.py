import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import torch
from heron.datasets.base_datasets import BaseDataset

class MyCSVDataset(BaseDataset):
    def __init__(self, csv_path, is_inference=False):
        super(MyCSVDataset, self).__init__(is_inference)
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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

        # Resize image
        resized_image = self.image_transform(image)

        # Tokenize text with a fixed max_length
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Specify a fixed max_length for tokenized sequences
        )

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "labels": tokenized_text["input_ids"].squeeze().clone(),
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "pixel_values": resized_image,
        }

    def _get_item_inference(self, index):
        row = self.data.iloc[index]
        image_path = row["image_path"]
        text = row["text"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Resize image
        resized_image = self.image_transform(image)

        # Tokenize text with a fixed max_length
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Specify a fixed max_length for tokenized sequences
        )

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "labels": None,
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "pixel_values": resized_image,
        }
