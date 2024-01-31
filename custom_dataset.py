import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path, is_inference=False):
        super(CSVDataset, self).__init__()
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

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
        padding=True,
        truncation=True,
        max_length=512  # Specify a fixed max_length for tokenized sequences
    )

    # Squeeze unnecessary dimension
    tokenized_text["input_ids"] = tokenized_text["input_ids"].squeeze(0)
    tokenized_text["attention_mask"] = tokenized_text["attention_mask"].squeeze(0)

    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "pixel_values": resized_image,
    }
