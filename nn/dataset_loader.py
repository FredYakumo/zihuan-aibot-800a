import csv
import torch
from torch.utils.data import Dataset

def load_reply_intent_classifier_data(file_path):
    """Load ReplyIntentClassifier training data, return a list of texts and labels."""
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

class ReplyIntentClassifierDataset(Dataset):
    """Custom dataset class."""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]