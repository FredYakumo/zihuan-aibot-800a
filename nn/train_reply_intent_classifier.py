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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create separate histories for epoch and step metrics
    epoch_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    step_history = {
        'step': [],
        'step_loss': [],
        'step_acc': []
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            step_loss = loss.item()
            predicted = (outputs >= 0.5).float()
            step_acc = (predicted == labels).sum().item() / labels.size(0)
            

            # Record step-level metrics
            step_history['step'].append(global_step)
            step_history['step_loss'].append(step_loss)
            step_history['step_acc'].append(step_acc)
            
            global_step += 1
            
            train_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

            if (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Step Loss: {step_loss:.4f}, Step Acc: {step_acc:.4f}')
        
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
        
        # Record epoch-level metrics
        epoch_history['epoch'].append(epoch+1)
        epoch_history['train_loss'].append(train_loss)
        epoch_history['train_acc'].append(train_acc)
        epoch_history['val_loss'].append(val_loss)
        epoch_history['val_acc'].append(val_acc)


        logger.info(f'Epoch {epoch+1}/{epochs}')
        logger.info(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n')
        
        # Save model
        # if val_acc > best_val_acc:
            # best_val_acc = val_acc
        torch.save(model.state_dict(), 'model.pth')
    
    # Save training histories separately
    pd.DataFrame(epoch_history).to_csv("epoch_metrics.csv", index=False)
    pd.DataFrame(step_history).to_csv("step_metrics.csv", index=False)
    
    # Return both histories as a tuple
    return epoch_history, step_history

# Visualization module
def visualize_training(epoch_history, step_history):
    """Plot training curves and show data table."""
    logger.info("Visualize training...")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epoch_history['epoch'], epoch_history['train_loss'], label='Train Loss')
    plt.plot(epoch_history['epoch'], epoch_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss (by epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epoch_history['epoch'], epoch_history['train_acc'], label='Train Accuracy')
    plt.plot(epoch_history['epoch'], epoch_history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy (by epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(step_history['step'], step_history['step_loss'], label='Step Loss', alpha=0.6)
    plt.title('Training Loss (by step)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(step_history['step'], step_history['step_acc'], label='Step Accuracy', alpha=0.6)
    plt.title('Training Accuracy (by step)')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Display epoch metrics summary
    epoch_df = pd.DataFrame(epoch_history)
    logger.info("\nTraining metrics summary (by epoch):")
    logger.info(epoch_df.round(4))

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
    epoch_history, step_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=6,
        lr=0.001
    )
    model_file_name = 'model.pth'
    logger.info(f"Save model to {model_file_name}")

    visualize_training(epoch_history, step_history)
