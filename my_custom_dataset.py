import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        text = self.data.iloc[idx, 1]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Return data as a dictionary
        sample = {"image": img, "text": text}

        return sample

# Example usage
#csv_dataset_path = "/path/to/your/csv/dataset.csv"
#transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#custom_dataset = CustomCSVDataset(csv_dataset_path, transform=transform)

# Accessing a sample from the dataset
#sample = custom_dataset[0]
#print(sample["image"].shape)  # Tensor representing the image
#print(sample["text"])  # Text corresponding to the image
