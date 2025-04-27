import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import csv

# Data loading module
def load_data(file_path):
    """Load training data, return a list of texts and labels."""
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            text = row[0].strip().strip('"')
            label = int(row[1].strip())
            texts.append(text)
            labels.append(label)
    return texts, labels

# Text embedding module
class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def embed(self, texts):
        """Generate text embedding vectors."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# Dataset module
class TextClassificationDataset(Dataset):
    """Custom dataset class."""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Model definition module
class ClassificationModel(nn.Module):
    """Neural network model for classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

# Training module
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train and validate the model."""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon GPU")
        device = torch.device('mps')
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation phase
        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Record history
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n')
        
        # Save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')
    
    # Save training history
    pd.DataFrame(history).to_csv("training_metrics.csv", index=False)
    return history

# Visualization module
def visualize_training(history):
    """Plot training curves and show data table."""
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epoch'], history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Print data table
    df = pd.DataFrame(history)
    print("\nTraining metrics summary:")
    print(df.round(4))

# Main program
if __name__ == "__main__":
    # Load data
    texts, labels = load_data('train_data.txt')
    
    # Generate embeddings
    embedder = TextEmbedder()
    print("Generating text embeddings...")
    embeddings = embedder.embed(texts)
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings.numpy(),
        labels,
        test_size=0.2,
        random_state=42
    )
    
    # Create data loaders
    batch_size = 32
    train_dataset = TextClassificationDataset(X_train, y_train)
    val_dataset = TextClassificationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = embeddings.shape[1]
    model = ClassificationModel(input_dim)
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=6,
        lr=0.001
    )
    model_file_name = 'model.pth'
    print(f"Save model to {model_file_name}")
    # Visualize results
    visualize_training(history)

# Prediction function
def predict(text):
    """Make predictions using the trained model."""
    # Initialize components
    embedder = TextEmbedder()
    model = ClassificationModel(embedder.model.config.hidden_size)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    # Generate embeddings
    embedding = embedder.embed([text])
    
    # Make prediction
    with torch.no_grad():
        output = model(embedding)
    return output.item()
