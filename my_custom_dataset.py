import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import torch

class MyCSVDataset(BaseDataset):
    def __init__(self, csv_path, is_inference=False):
        super(MyCSVDataset, self).__init__(is_inference)
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # ... (rest of the class)

    def _get_item_train(self, index):
        row = self.data.iloc[index]
        image_path = row["image_path"]
        text = row["text"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Tokenize text
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Apply image transformations
        processed_image = self.image_transform(image)

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "labels": tokenized_text["input_ids"].squeeze(),  # For language modeling, adjust as needed
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "pixel_values": processed_image,
        }

    def _get_item_inference(self, index):
        row = self.data.iloc[index]
        image_path = row["image_path"]
        text = row["text"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Tokenize text
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Apply image transformations
        processed_image = self.image_transform(image)

        return {
            "input_ids": tokenized_text["input_ids"].squeeze(),
            "labels": None,  # No labels for inference
            "attention_mask": tokenized_text["attention_mask"].squeeze(),
            "pixel_values": processed_image,
        }
