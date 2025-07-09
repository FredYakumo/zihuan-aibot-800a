from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_config import logger
from transformers.pipelines import pipeline

def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU")
        device = torch.device('mps')
    else:
        logger.info("Using CPU")
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
        
        # a_norm = torch.sqrt(torch.sum(a ** 2, dim=1, keepdim=True))   # shape: [1, 1]
        # B_norm = torch.sqrt(torch.sum(B ** 2, dim=1, keepdim=True))   # shape: [N, 1]
        # 避免除以 0
        B_norm = B_norm.clamp(min=eps)
        
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
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', mean_pooling=True, device=torch.device('cpu')):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)  # This should be your Hugging Face model that accepts tokenized inputs.
        self.config = AutoConfig.from_pretrained(model_name)
        print(f"Max position embedding: {self.config.max_position_embeddings}")
        self.mean_pooling = mean_pooling
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Assume you want to mean-pool the output embeddings.
        if self.mean_pooling:
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings
        else:
            return outputs

class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', device=torch.device('cpu'),mean_pooling=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TextEmbeddingModel(model_name, mean_pooling, device)
        
    def embed(self, texts):
        """Generate text embedding vectors.

        Args:
            texts (list[str]): A list of texts to embed.
            mean_pooling (bool): If true, applies mean pooling to the token embeddings.
                                 Otherwise, returns all token embeddings.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        
        if self.model.mean_pooling:
            return outputs
        else:
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state
    
def load_reply_intent_classifier_model(model_path):
    """Load the trained model"""
    embedder = TextEmbedder()
    model = ReplyIntentClassifierModel(embedder.model.config.hidden_size)
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, embedder

    
class MultiLabelClassifier(nn.Module):
    """
    A multi-label classifier using a pre-trained transformer model and CNN layers.

    This model first generates text embeddings using a pre-trained transformer model.
    Then, it passes these embeddings through multiple parallel 1D convolutional layers
    with different kernel sizes. The output of each convolutional layer is passed
    through a max-pooling layer. Finally, the pooled features are concatenated,
    passed through a dropout layer, and a fully connected layer to produce the
    final logits for each label. This architecture is inspired by TextCNN.
    """
    def __init__(self, embedding_model, num_labels, num_filters=128, filter_sizes=[3, 4, 5], dropout=0.2):
        """
        @param embedding_model The pre-trained transformer model.
        @param num_labels The number of labels for classification.
        @param num_filters The number of filters for each convolutional layer.
        @param filter_sizes A list of kernel sizes for the convolutional layers.
        @param dropout The dropout rate for the final layer.
        """
        super(MultiLabelClassifier, self).__init__()
        self.embedding_model = embedding_model
        bert_hidden_size = self.embedding_model.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=bert_hidden_size,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """
        @param input_ids The input token ids.
        @param attention_mask The attention mask.
        @return Logits for each label.
        """
        # Get embeddings from the transformer model
        # outputs.last_hidden_state shape: (batch_size, sequence_length, embedding_dim)
        outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
        embedded = outputs.last_hidden_state

        # Conv1d expects input of shape (batch_size, in_channels, sequence_length)
        # We need to permute the dimensions of the embeddings
        embedded = embedded.permute(0, 2, 1)

        # Apply convolution and pooling
        # conved is a list of tensors of shape (batch_size, num_filters, new_sequence_length)
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # Apply max-over-time pooling
        # pooled is a list of tensors of shape (batch_size, num_filters)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # Concatenate the pooled features from all filter sizes
        # cat has shape (batch_size, num_filters * len(filter_sizes))
        cat = self.dropout(torch.cat(pooled, dim=1))

        # Pass through the final fully connected layer
        logits = self.fc(cat)
        return logits