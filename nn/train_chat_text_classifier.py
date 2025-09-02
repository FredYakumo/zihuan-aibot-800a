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
    MultiLabelClassifier,
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


BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5
GRAD_CLIP_NORM = 1.0  # maximum norm for gradient clipping
PREDICTION_THRESHOLD = 0.5  # Threshold for binary prediction in evaluation

# File paths for saving models and metrics
BEST_MODEL_STATE_PATH = "best_model_state.bin"
TEXT_EMBEDDING_PT_PATH = "text_embedding.pt"
TEXT_EMBEDDING_ONNX_PATH = "text_embedding.onnx"
MULTI_LABEL_CLASSIFIER_ONNX_PATH = "multi_label_classifier.onnx"
TRAINING_METRICS_PLOT_PATH = "training_metrics.png"

# ONNX export settings
ONNX_OPSET_VERSION = 11


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

# --- Text Tokenization and Vectorization Model ---
# Use the TextEmbedder class from models.py to load the text tokenization and vectorization model


device = get_device()
print(f"\nCurrent device: {device}")

text_embedder = TextEmbedder(device=device)
tokenizer = text_embedder.tokenizer


# --- Dataset and DataLoader ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Create a TextEmbedder with mean_pooling=False to get token-level embeddings
        self.text_embedder = TextEmbedder(device=device, mean_pooling=False)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = torch.tensor(self.labels[item], dtype=torch.float32)

        # ensure text is long enough to tokenize meaningfully
        # if text is very short, duplicate it to create a longer sequence
        if len(text) < MIN_TEXT_LENGTH:
            text = text + " " + text  # Duplicate short texts

        # generate token-level embeddings (not mean-pooled)
        with torch.no_grad():
            # Tokenize manually first to get tokens
            tokens = self.tokenizer(
                text,
                padding="max_length",  # Use max_length to ensure consistent length
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(device)

            embeddings = self.text_embedder.model(
                tokens["input_ids"], tokens["attention_mask"]
            )

        return {
            "text": text,
            "embeddings": embeddings.squeeze(0),  # remove batch dimension
            "attention_mask": tokens["attention_mask"].squeeze(0),
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
    Custom collate function for handling sequences of different lengths
    """
    texts = [item["text"] for item in batch]
    embeddings = [item["embeddings"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Stack embeddings into a tensor (they should all have the same first dimension)
    embeddings = torch.stack(embeddings)

    # Stack attention masks into a tensor
    attention_masks = torch.stack(attention_masks)

    # Stack labels into a tensor
    labels = torch.stack(labels)

    return {
        "text": texts,
        "embeddings": embeddings,
        "attention_mask": attention_masks,
        "labels": labels,
    }


train_dataset = TextDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_len=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH,
)

val_dataset = TextDataset(
    texts=val_texts,
    labels=val_labels,
    tokenizer=tokenizer,
    max_len=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH,
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


print(f"\nInitializing MultiLabelClassifier with {num_labels} classes")
config = AutoConfig.from_pretrained(TEXT_EMBEDDING_DEFAULT_MODEL_NAME)
# embedding_dim = config.hidden_size
embedding_dim = TEXT_EMBEDDING_OUTPUT_LENGTH

model = MultiLabelClassifier(
    embedding_dim=embedding_dim,
    num_classes=num_labels,
    max_seq_len=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH,
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss().to(device)


# --- Training  ---


def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    losses = []
    progress_bar = tqdm(
        data_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False
    )
    for d in progress_bar:
        embeddings = d["embeddings"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(embeddings, attention_mask)
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
            embeddings = d["embeddings"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(embeddings, attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            predictions = torch.sigmoid(outputs).cpu().numpy()
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


print("\nexported model")


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

# # --- Convert the model to ONNX model ---
# # Load the best model state
# model.load_state_dict(torch.load(BEST_MODEL_STATE_PATH))
# model.eval()  # Switch to evaluation mode

# # Create a dummy input for ONNX export
# # Use a dummy embedding tensor with the right dimensions
# dummy_embeddings = torch.randn(
#     1, CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH, embedding_dim
# ).to(device)
# # Create a dummy attention mask
# dummy_attention_mask = torch.ones(
#     1, CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH, dtype=torch.long
# ).to(device)

# onnx_model_path = MULTI_LABEL_CLASSIFIER_ONNX_PATH

# print(f"\nStart exporting the ONNX model to {onnx_model_path}...")

# try:
#     # make sure the model is on the CPU for ONNX export, or make sure your ONNX Runtime environment supports CUDA ONNX models
#     # It is usually recommended to export on the CPU for wider compatibility
#     model.to("cpu")
#     dummy_embeddings = dummy_embeddings.to("cpu")
#     dummy_attention_mask = dummy_attention_mask.to("cpu")

#     torch.onnx.export(
#         model,
#         (dummy_embeddings, dummy_attention_mask),  # Wrap in tuple with attention mask
#         onnx_model_path,
#         input_names=["embeddings", "attention_mask"],
#         output_names=["logits"],
#         dynamic_axes={
#             "embeddings": {0: "batch_size", 1: "sequence_length"},
#             "attention_mask": {0: "batch_size", 1: "sequence_length"},
#             "logits": {0: "batch_size"},
#         },
#         opset_version=ONNX_OPSET_VERSION,  # It is recommended to use a stable and widely supported opset version
#         do_constant_folding=True,  # Optimization: perform constant folding
#     )
#     print(f"ONNX model successfully exported to {onnx_model_path}")
# except Exception as e:
#     print(f"ONNX model export failed: {e}")
