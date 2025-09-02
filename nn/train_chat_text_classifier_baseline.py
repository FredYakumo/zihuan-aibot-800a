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
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nn.models import (
    CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH,
    TEXT_EMBEDDING_DEFAULT_MODEL_NAME,
    TEXT_EMBEDDING_OUTPUT_LENGTH,
    MultiLabelClassifierBaseline,
    TextEmbedder,
    get_device,
)

# --- Hyperparameters ---
# Data loading
TRAIN_SET_DIR = "train_data/train_set"
LABELS_FILE = "train_data/labels.csv"
TEST_SIZE = 0.2
RANDOM_SEED = 42
MIN_TEXT_LENGTH = 1  # For duplicating short texts


BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0  # maximum norm for gradient clipping
PREDICTION_THRESHOLD = 0.5  # Threshold for binary prediction in evaluation

# File paths for saving models and metrics
BEST_MODEL_STATE_PATH = "chat_text_classifier_baseline.pt"
TRAINING_METRICS_PLOT_PATH = "chat_text_classifier_baseline_training_metrics.png"


# Data Preparation


label_df = pd.read_csv(LABELS_FILE)
label_to_id = {row["label_name"]: index for index, row in label_df.iterrows()}
id_to_label = {index: row["label_name"] for index, row in label_df.iterrows()}
num_labels = len(label_to_id)
print(f"Loaded label mapping: {label_to_id}")
print(f"Total number of labels: {num_labels}")

# read all CSV files from train_data/train_set directory and merge them
csv_files = [f for f in os.listdir(TRAIN_SET_DIR) if f.endswith(".csv")]


# process multi-labels: convert label string to one-hot encoded list
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
        file_path = os.path.join(TRAIN_SET_DIR, csv_file)
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

# --- Set up device ---
device = get_device()
print(f"\nCurrent device: {device}")

# --- Text Embedder for BGE-M3 ---
# print(f"\nInitializing tokenizer with model: {TEXT_EMBEDDING_DEFAULT_MODEL_NAME}")
# tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_DEFAULT_MODEL_NAME)

text_embedder = TextEmbedder(device=device, mean_pooling=True)


# --- Dataset and DataLoader ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = torch.tensor(self.labels[item], dtype=torch.float32)
        
        # Return token IDs and attention mask directly
        return {
            "text": text,
            "labels": labels,
        }


# split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].to_numpy(),
    train_df["one_hot_labels"].to_numpy(),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
)


def collate_fn(batch):
    """
    Collate function for handling token IDs and attention masks
    """
    texts = [item["text"] for item in batch]
    labels = [item["labels"] for item in batch]

    labels = torch.stack(labels)

    return {
        "text": texts,
        "labels": labels,
    }


train_dataset = TextDataset(
    texts=train_texts,
    labels=train_labels,
    max_len=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH
)

val_dataset = TextDataset(
    texts=val_texts,
    labels=val_labels,
    max_len=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


print(f"\nInitializing MultiLabelClassifier with {num_labels} classes")


model = MultiLabelClassifierBaseline(num_classes=num_labels)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss().to(device)  # Using BCE Loss since sigmoid is already applied in the model


# --- Training  ---


def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    losses = []
    progress_bar = tqdm(
        data_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False
    )
    for d in progress_bar:
        # Forward pass with token IDs and attention mask
        embedding = text_embedder.embed(d["text"])
        outputs = model(embedding)
        
        # Move labels to the same device as outputs
        labels = d["labels"].to(device)
        loss = loss_fn(outputs, labels)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP_NORM
        )  # Gradient clipping
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
            embedding = text_embedder.embed(d["text"])
            outputs = model(embedding)
            # Move labels to the same device as outputs
            labels = d["labels"].to(device)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            predictions = outputs.cpu().numpy()  # Sigmoid is now applied in the model
            
            true_labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

    avg_loss = sum(losses) / len(losses)

    all_predictions_binary = (
        (torch.tensor(all_predictions) > PREDICTION_THRESHOLD).long().numpy()
    )

    f1_micro = f1_score(all_true_labels, all_predictions_binary, average="micro")
    f1_macro = f1_score(all_true_labels, all_predictions_binary, average="macro")

    return avg_loss, f1_micro, f1_macro


# --- Training Loop ---


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

    # save the best model (based on validation set Micro F1 score)
    if val_f1_micro > best_f1_micro:
        best_f1_micro = val_f1_micro
        torch.save(model.state_dict(), BEST_MODEL_STATE_PATH)
        print(f"  -> New best model saved with F1 (Micro): {best_f1_micro:.4f}")

print("\nTraining finished.")

# plotting and saving the metrics chart
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
plt.savefig(TRAINING_METRICS_PLOT_PATH)
print(f"\nMetrics chart saved to {TRAINING_METRICS_PLOT_PATH}")
plt.show()
