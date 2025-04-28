import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from dataset_loader import ReplyIntentClassifierDataset, load_reply_intent_classifier_data
from models import ReplyIntentClassifierModel, TextEmbedder, get_device


from utils.config_loader import ConfigLoader
from utils.logging_config import logger


device = get_device()

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train and validate the model."""
    
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


        logger.info(f'Epoch {epoch+1}/{epochs}')
        logger.info(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n')
        
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
    logger.info("Visualize training...")
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

    # logger.info data table
    df = pd.DataFrame(history)
    logger.info("\nTraining metrics summary:")
    logger.info(df.round(4))

if __name__ == "__main__":
    logger.info("正在加载ReplyIntentClassifierDataset训练数据...")
    start_time = time.time()
    texts, labels = load_reply_intent_classifier_data('train_data.txt')
    end_time = time.time()
    logger.info(f"训练数据加载完成,耗时{end_time - start_time:.2f}秒")


    logger.info("正在加载Tokenlizer模型...")
    start_time = time.time()
    embedder = TextEmbedder(device=device)
    embeddings = embedder.embed(texts)
    end_time = time.time()
    logger.info(f"Tokenlizer模型加载完成,耗时{end_time - start_time:.2f}秒")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings.cpu().numpy(),
        labels,
        test_size=0.2,
        random_state=42
    )

    # Create data loaders
    batch_size = 32
    train_dataset = ReplyIntentClassifierDataset(X_train, y_train)
    val_dataset = ReplyIntentClassifierDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    input_dim = embeddings.shape[1]
    logger.info(f"模型输入维度: {input_dim}")
    model = ReplyIntentClassifierModel(input_dim).to(device)

    logger.info("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=6,
        lr=0.001
    )
    model_file_name = 'model.pth'
    logger.info(f"Save model to {model_file_name}")

    visualize_training(history)
