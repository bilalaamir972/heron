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

        # Reshape the "input_ids" to remove the second dimension
        tokenized_text["input_ids"] = tokenized_text["input_ids"].squeeze(0)

        # In image captioning, you usually have source and target sequences
        # Source sequence (image features)
        image_inputs = {"pixel_values": resized_image}

        # Target sequence (text description)
        text_targets = tokenized_text["input_ids"]

        # Modify the return statement to include "labels" key with ignore_index
        return {
            **image_inputs,
            **tokenized_text,
            "labels": text_targets,
        }
