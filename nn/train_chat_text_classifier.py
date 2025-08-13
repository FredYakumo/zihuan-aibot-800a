import json  # For handling potential JSON serialization in labels if needed later
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nn.models import MultiLabelClassifier, TextEmbedder, get_device

# --- 1. Data Preparation ---

# Load label data
label_df = pd.read_csv("train_data/labels.csv")
label_to_id = {row["label_name"]: index for index, row in label_df.iterrows()}
id_to_label = {index: row["label_name"] for index, row in label_df.iterrows()}
num_labels = len(label_to_id)
print(f"Loaded label mapping: {label_to_id}")
print(f"Total number of labels: {num_labels}")

# Load training text data
# Read all CSV files from train_data/train_set directory and merge them
train_set_dir = "train_data/train_set"
csv_files = [f for f in os.listdir(train_set_dir) if f.endswith(".csv")]


# Process multi-labels: convert label string to one-hot encoded list
def process_labels(labels_str, label_map, num_classes):
    # \brief Print labels_str if its type is float.
    if not isinstance(labels_str, str):
        raise ValueError(f"labels_str is not string: {labels_str}")

    label_ids = []
    for label in labels_str.split("|"):
        label = label.strip()
        if not label:
            continue
        if label not in label_map:
            raise ValueError(
                f"Label '{label}' not found in label mapping. Please check your training data or labels.csv."
            )
        label_ids.append(label_map[label])

    one_hot = [0] * num_classes
    for idx in label_ids:
        if 0 <= idx < num_classes:
            one_hot[idx] = 1
    return one_hot


try:
    all_rows = []
    for csv_file in csv_files:
        file_path = os.path.join(train_set_dir, csv_file)
        print(f"Loading train set {file_path}...")
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # CSV line number is index + 2 (1 for header, 1 for 0-based index)
            one_hot_label = process_labels(row["labels"], label_to_id, num_labels)
            all_rows.append({"text": row["text"], "one_hot_labels": one_hot_label})
    train_df = pd.DataFrame(all_rows)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

print("\nProcessed training data example:")
print(train_df.head())
print(f"train set count: {len(csv_files)}, total sample count: {len(train_df)}")

# --- 2. Text Tokenization and Vectorization Model ---
# Use the TextEmbedder class from models.py to load the text tokenization and vectorization model

# --- 5. Device Selection ---
# Move the model to the corresponding device for use
device = get_device()
print(f"\nCurrent device: {device}")

text_embedder = TextEmbedder(device=device)
tokenizer = text_embedder.tokenizer
# Use the underlying transformer model from TextEmbedder
embedding_model = (
    text_embedder.model.model
)  # Get the actual AutoModel from TextEmbeddingModel


# --- 3. Dataset and DataLoader ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = torch.tensor(self.labels[item], dtype=torch.float32)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding=False,  # No padding at tokenization stage
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels,
        }


# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].to_numpy(),
    train_df["one_hot_labels"].to_numpy(),
    test_size=0.2,
    random_state=42,
)

MAX_LEN = 128
BATCH_SIZE = 1


# Custom collate function for handling sequences of different lengths
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences to the length of the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)

    # Stack labels into a tensor
    labels = torch.stack(labels)

    return {
        "text": texts,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


train_dataset = TextDataset(
    texts=train_texts, labels=train_labels, tokenizer=tokenizer, max_len=MAX_LEN
)

val_dataset = TextDataset(
    texts=val_texts, labels=val_labels, tokenizer=tokenizer, max_len=MAX_LEN
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# --- 4. Model Definition ---
model = MultiLabelClassifier(embedding_model, num_labels)
model = model.to(device)

# --- 6. Training Configuration ---
EPOCHS = 20  # Training epochs
LEARNING_RATE = 2e-5  # Learning rate

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# For multi-label classification, BCEWithLogitsLoss is usually used because it has an integrated Sigmoid activation function
# and is numerically more stable than a separate Sigmoid + BCELoss
loss_fn = nn.BCEWithLogitsLoss().to(device)


# --- 7. Training Function ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    losses = []
    progress_bar = tqdm(
        data_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False
    )
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return sum(losses) / len(losses)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for d in progress_bar:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Convert logits to probabilities, then convert to predicted labels based on a threshold
            # The sigmoid here is to convert logits to probabilities between 0 and 1
            # In BCEWithLogitsLoss, this Sigmoid is built-in, so outputs are logits
            predictions = torch.sigmoid(outputs).cpu().numpy()
            true_labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

    avg_loss = sum(losses) / len(losses)
    # For multi-label classification, micro or macro F1 score is usually used
    # Need to convert probability predictions to binary labels, here simply use 0.5 as the threshold
    # Or you can adjust the threshold according to the actual application
    all_predictions_binary = (torch.tensor(all_predictions) > 0.5).long().numpy()
    f1_micro = f1_score(all_true_labels, all_predictions_binary, average="micro")
    f1_macro = f1_score(all_true_labels, all_predictions_binary, average="macro")

    return avg_loss, f1_micro, f1_macro


# --- 8. Training Loop ---
print("\nStarting training...")
best_f1_micro = -1

# For plotting
train_losses = []
val_losses = []
val_f1_micros = []
val_f1_macros = []

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
    train_loss = train_epoch(
        model, train_dataloader, loss_fn, optimizer, device, epoch + 1, EPOCHS
    )
    val_loss, val_f1_micro, val_f1_macro = eval_model(
        model, val_dataloader, loss_fn, device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1_micros.append(val_f1_micro)
    val_f1_macros.append(val_f1_macro)

    print(
        f"Epoch {epoch + 1} Summary: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val F1 (Micro): {val_f1_micro:.4f}, Val F1 (Macro): {val_f1_macro:.4f}"
    )

    # Save the best model (based on validation set Micro F1 score)
    if val_f1_micro > best_f1_micro:
        best_f1_micro = val_f1_micro
        torch.save(model.state_dict(), "best_model_state.bin")
        print(f"  -> New best model saved with F1 (Micro): {best_f1_micro:.4f}")

print("\nTraining finished.")

# Plotting and saving the metrics chart
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(15, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, "o-", label="Training Loss")
plt.plot(epochs_range, val_losses, "o-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot F1 Scores
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_f1_micros, "o-", label="Validation F1 (Micro)")
plt.plot(epochs_range, val_f1_macros, "o-", label="Validation F1 (Macro)")
plt.title("Validation F1 Scores")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

plt.suptitle("Training Metrics")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig("training_metrics.png")
print("\nMetrics chart saved to training_metrics.png")
plt.show()

# --- 9. Convert the model to ONNX model ---
# Load the best model state
model.load_state_dict(torch.load("best_model_state.bin"))
model.eval()  # Switch to evaluation mode

# Create a dummy input for ONNX export
# input_ids: (batch_size, sequence_length)
# attention_mask: (batch_size, sequence_length)
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN)).to(device)
dummy_attention_mask = torch.ones(1, MAX_LEN, dtype=torch.long).to(device)

onnx_model_path = "multi_label_classifier.onnx"

print(f"\nStart exporting the ONNX model to {onnx_model_path}...")

try:
    # Make sure the model is on the CPU for ONNX export, or make sure your ONNX Runtime environment supports CUDA ONNX models
    # It is usually recommended to export on the CPU for wider compatibility
    model.to("cpu")
    dummy_input_ids = dummy_input_ids.to("cpu")
    dummy_attention_mask = dummy_attention_mask.to("cpu")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=11,  # It is recommended to use a stable and widely supported opset version
        do_constant_folding=True,  # Optimization: perform constant folding
    )
    print(f"ONNX model successfully exported to {onnx_model_path}")
except Exception as e:
    print(f"ONNX model export failed: {e}")
