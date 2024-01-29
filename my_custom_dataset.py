def _get_item_train(self, index):
    row = self.data.iloc[index]
    image_path = row["image_path"]
    text = row["text"]

    # Tokenize text with a fixed max_length
    tokenized_text = self.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Specify a fixed max_length for tokenized sequences
    )

    # Load image using PIL
    image = Image.open(image_path).convert("RGB")

    # Apply image transformations
    processed_image = self.image_transform(image)

    return {
        "input_ids": tokenized_text["input_ids"].squeeze(),
        "labels": tokenized_text["input_ids"].squeeze().clone(),
        "attention_mask": tokenized_text["attention_mask"].squeeze(),
        "pixel_values": processed_image,
    }

def _get_item_inference(self, index):
    row = self.data.iloc[index]
    image_path = row["image_path"]
    text = row["text"]

    # Tokenize text with a fixed max_length
    tokenized_text = self.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Specify a fixed max_length for tokenized sequences
    )

    # Load image using PIL
    image = Image.open(image_path).convert("RGB")

    # Apply image transformations
    processed_image = self.image_transform(image)

    return {
        "input_ids": tokenized_text["input_ids"].squeeze(),
        "labels": None,
        "attention_mask": tokenized_text["attention_mask"].squeeze(),
        "pixel_values": processed_image,
    }
