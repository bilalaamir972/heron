import torch

class CSVDataset(Dataset):
    def __init__(self, csv_path):
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
            "ignore_index": self.tokenizer.pad_token_id,
        }
