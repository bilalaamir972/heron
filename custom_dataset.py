import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        super(CSVDataset, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add padding token to the tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
        labels = row["labels"]

        # Load image using PIL
        image = Image.open(image_path).convert("RGB")

        # Resize image
        resized_image = self.image_transform(image)

        # Tokenize text with a fixed max_length
        tokenized_text = self.tokenizer(
            labels,
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
        }
