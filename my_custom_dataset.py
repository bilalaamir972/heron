import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import torch
from heron.datasets.base_datasets import BaseDataset

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        super(CSVDataset, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.max_sequence_length = 512  # Adjust this based on your model's input size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
        )

        # Modify the return statement to include "labels" key with ignore_index
        return {
            "pixel_values": resized_image,
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
            "labels": tokenized_text["input_ids"].squeeze(0),  # Target labels with the same batch size
        }




   
