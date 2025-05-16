from transformers import AutoTokenizer, AutoModel, AutoConfig
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

class CosineSimilarityModel(nn.Module):
    def forward(self, a: torch.Tensor, B: torch.Tensor):
        assert a.dim() == 2 and a.shape[0] == 1, "a must be a [1, D] vector"
        assert B.dim() == 2, "B must be a [N, D] matrix"
        
        # avoid div 0
        eps = 1e-8
        
        a_norm = a.norm(dim=1, keepdim=True)  # [1,1]
        
        B_norm = B.norm(dim=1, keepdim=True)  # [N,1]
        B_norm = torch.max(B_norm, torch.tensor(eps))
        
        dot_product = torch.matmul(a, B.t())  # [1,N]
        
        similarity = dot_product / (a_norm * B_norm.t())
        
        # [N]
        return similarity.squeeze(0)

class ReplyIntentClassifierModel(nn.Module):
    """Neural network model for classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()
    
class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)  # This should be your Hugging Face model that accepts tokenized inputs.

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     # Assume you want to mean-pool the output embeddings.
    #     embeddings = outputs.last_hidden_state.mean(dim=1)
    #     return embeddings
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        # Assume you want to mean-pool the output embeddings.
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', device=torch.device('cpu')):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        print(f"Max position embedding: {config.max_position_embeddings}")
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