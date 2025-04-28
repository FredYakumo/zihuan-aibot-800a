from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from utils.logging_config import logger

def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU")
        device = torch.device('mps')
    return device

class ReplyIntentClassifierModel(nn.Module):
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
    


class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', device=torch.device('cpu')):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
    def embed(self, texts):
        """Generate text embedding vectors."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
def load_reply_intent_classifier_model(model_path):
    """Load the trained model"""
    embedder = TextEmbedder()
    model = ReplyIntentClassifierModel(embedder.model.config.hidden_size)
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, embedder