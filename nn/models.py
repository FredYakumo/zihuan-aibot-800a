import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForSequenceClassification,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline

from utils.logging_config import logger

TEXT_EMBEDDING_DEFAULT_MODEL_NAME = "BAAI/bge-m3"
# TEXT_EMBEDDING_DEFAULT_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
LTP_MODEL_NAME = "LTP/base"
TEXT_EMBEDDING_INPUT_LENGTH = 8192
TEXT_EMBEDDING_OUTPUT_LENGTH = 1024
CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH = 1024
LTP_MAX_INPUT_LENGTH = 512


def get_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU")
        device = torch.device("mps")
    else:
        logger.info("Using CPU")
    return device


class TextEmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name=TEXT_EMBEDDING_DEFAULT_MODEL_NAME,
        mean_pooling=True,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name).to(
            device
        )  # This should be your Hugging Face model that accepts tokenized inputs.
        self.config = AutoConfig.from_pretrained(model_name)
        print(
            f'TextEmbedding model "{model_name}" - max position embedding: {self.config.max_position_embeddings}'
        )
        self.mean_pooling = mean_pooling
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Always return tensor embeddings only, not the full outputs object
        if self.mean_pooling:
            # Return mean-pooled sentence embeddings [batch_size, hidden_size]
            # Use attention mask to properly handle padding tokens
            token_embeddings = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_embeddings = token_embeddings * attention_mask_expanded
            sum_embeddings = masked_embeddings.sum(dim=1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            # Avoid /0
            seq_lengths = torch.clamp(seq_lengths, min=1.0)

            return sum_embeddings / seq_lengths
        else:
            # Return token-level embeddings [batch_size, seq_len, hidden_size]
            return outputs.last_hidden_state


class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""

    def __init__(
        self,
        model_name=TEXT_EMBEDDING_DEFAULT_MODEL_NAME,
        device=torch.device("cpu"),
        mean_pooling=True,
    ):
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
            # padding="max_length",
            padding=True,
            truncation=True,
            max_length=TEXT_EMBEDDING_INPUT_LENGTH,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            embeddings = self.model(inputs["input_ids"], inputs["attention_mask"])

        # Model now always returns tensor embeddings directly
        return embeddings


class MultiLabelClassifierBaseline(nn.Module):
    """
    Architecture:
        Input (1024-d embedding)
        ↓
        Dense Layer (512 units, activation=ReLU)
        ↓
        Dropout (0.1~0.3)
        ↓
        Dense Layer (num_labels, activation=Sigmoid/Softmax)

    """

    def __init__(
        self,
        embedding_dim=TEXT_EMBEDDING_OUTPUT_LENGTH,
        num_classes=41,
        max_seq_length=CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH,
        dropout_rate=0.2,
        hidden_dim=768,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        # Simple feed-forward network
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using proper initialization techniques"""
        
        # Dense layers - Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        """
        Forward pass for the simple feed-forward classifier

        Args:
            input_ids: Token IDs with shape [batch_size, seq_len]
            attention_mask: Attention mask with shape [batch_size, seq_len]

        Returns:
            Tensor with shape [batch_size, num_classes] containing class probabilities
        """

        
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        logits = self.fc3(hidden)

        # Apply sigmoid for multi-label classification
        return torch.sigmoid(logits)

    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """
        Load a pretrained model from a saved checkpoint.

        Args:
            model_path: Path to the saved model
            device: Device to load the model to (cpu, cuda, mps)

        Returns:
            Loaded model instance
        """
        if device is None:
            device = get_device()

        # Load config and state dict
        checkpoint = torch.load(model_path, map_location=device)

        # Create model with saved hyperparameters
        model = cls(
            embedding_dim=checkpoint.get("embedding_dim", TEXT_EMBEDDING_OUTPUT_LENGTH),
            num_classes=checkpoint.get("num_classes", 41),
            max_seq_length=checkpoint.get(
                "max_seq_length", CHAT_TEXT_CLASSIFIER_MAX_INPUT_LENGTH
            ),
            dropout_rate=checkpoint.get("dropout_rate", 0.2),
            hidden_dim=checkpoint.get("hidden_dim", 512),
        )

        # Load model weights with error handling for architecture changes
        try:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info("Loaded model weights with model_state_dict key")
            else:
                # For backward compatibility with older saved models
                model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded model weights directly from checkpoint")
        except Exception as e:
            logger.warning(
                f"Could not load all weights from checkpoint due to architecture changes: {e}"
            )
            logger.info("Continuing with partially loaded or newly initialized weights")

        model.to(device)
        model.eval()

        return model


class LTPModel(nn.Module):
    """
    LTP (Language Technology Platform) model for Chinese NLP tasks.

    This model supports multiple Chinese NLP tasks including:
    - Word Segmentation (CWS)
    - Part-of-Speech Tagging (POS)
    - Named Entity Recognition (NER)
    - Semantic Role Labeling (SRL)
    - Dependency Parsing (DEP)
    - Semantic Dependency Parsing (SDP)
    """

    def __init__(
        self, model_name=LTP_MODEL_NAME, device=torch.device("cpu"), tasks=None
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tasks = tasks or ["cws", "pos", "ner"]  # Default tasks

        # Try to load LTP pipeline directly
        try:
            from ltp import LTP

            self.ltp_pipeline = LTP(model_name)
            if torch.cuda.is_available() and device.type == "cuda":
                self.ltp_pipeline.to("cuda")  # type: ignore
            self.use_ltp_pipeline = True
            logger.info(f"LTP Model loaded with LTP library: {model_name}")
            logger.info(f"Supported tasks: {self.tasks}")
        except ImportError:
            logger.warning(
                "LTP library not found. LTP model functionality will be limited."
            )
            self.use_ltp_pipeline = False
            # Fallback: try to use a supported BERT-like model for basic inference
            try:
                # Use BERT as fallback since LTP is based on BERT architecture
                fallback_model = "bert-base-chinese"
                self.model = AutoModel.from_pretrained(fallback_model).to(device)
                self.config = AutoConfig.from_pretrained(fallback_model)
                logger.info(f"Using fallback model: {fallback_model}")
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
                raise e

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for LTP model.
        Returns the hidden states that can be used for downstream tasks.
        """
        if self.use_ltp_pipeline:
            # For LTP pipeline, we cannot directly use forward pass
            # This method is mainly for compatibility with export functionality
            logger.warning(
                "Direct forward pass not supported with LTP pipeline. Use process_text instead."
            )
            # Return dummy tensor for export compatibility
            batch_size, seq_len = input_ids.shape
            hidden_size = 768  # Standard BERT hidden size
            return torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        else:
            # Use fallback model
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state


class LTPProcessor:
    """
    LTP processor for Chinese text analysis using the LTP pipeline.

    This class provides a high-level interface for various Chinese NLP tasks
    using the LTP model from Hugging Face.
    """

    def __init__(self, model_name=LTP_MODEL_NAME, device=torch.device("cpu")):
        self.model_name = model_name
        self.device = device

        # Initialize LTP pipeline if available
        try:
            from ltp import LTP

            self.ltp_pipeline = LTP(model_name)
            self.ltp_pipeline.to(device)  # type: ignore
            self.use_pipeline = True
            logger.info("LTP pipeline loaded successfully.")

            # Create a simple LTPModel for export compatibility
            self.model = LTPModel(model_name, device)

            # Try to get tokenizer from LTP or use fallback
            try:
                # LTP might not expose tokenizer directly, use BERT tokenizer as fallback
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                logger.info("Using bert-base-chinese tokenizer as fallback for LTP")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None

        except ImportError:
            logger.info("LTP library not found. Using transformers model only.")
            self.use_pipeline = False

            # Use BERT as fallback
            fallback_model = "bert-base-chinese"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = LTPModel(fallback_model, device)
                logger.info(f"Using fallback model: {fallback_model}")
            except Exception as e:
                logger.error(f"Failed to initialize fallback model: {e}")
                raise e

    def process_text(self, texts, tasks=None):
        """
        Process Chinese text with specified tasks.

        Args:
            texts (str or list[str]): Input text(s) to process
            tasks (list[str]): Tasks to perform. Available:
                              ['cws', 'pos', 'ner', 'srl', 'dep', 'sdp']

        Returns:
            dict: Processing results for each task
        """
        if isinstance(texts, str):
            texts = [texts]

        if tasks is None:
            tasks = ["cws", "pos", "ner"]

        if self.use_pipeline:
            # Use LTP pipeline if available
            results = self.ltp_pipeline.pipeline(texts, tasks=tasks)
            return {
                "cws": results.cws if hasattr(results, "cws") else None,  # type: ignore
                "pos": results.pos if hasattr(results, "pos") else None,  # type: ignore
                "ner": results.ner if hasattr(results, "ner") else None,  # type: ignore
                "srl": results.srl if hasattr(results, "srl") else None,  # type: ignore
                "dep": results.dep if hasattr(results, "dep") else None,  # type: ignore
                "sdp": results.sdp if hasattr(results, "sdp") else None,  # type: ignore
            }
        else:
            # Fallback to basic tokenization
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=LTP_MAX_INPUT_LENGTH,
                return_tensors="pt",
            ).to(  # type: ignore
                self.device
            )  # pyright: ignore[reportOptionalCall]

            with torch.no_grad():
                hidden_states = self.model(
                    inputs["input_ids"], inputs["attention_mask"]
                )

            return {
                "hidden_states": hidden_states,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

    def word_segmentation(self, texts):
        """Perform word segmentation (分词)"""
        return self.process_text(texts, tasks=["cws"])

    def pos_tagging(self, texts):
        """Perform part-of-speech tagging (词性标注)"""
        return self.process_text(texts, tasks=["cws", "pos"])

    def named_entity_recognition(self, texts):
        """Perform named entity recognition (命名实体识别)"""
        return self.process_text(texts, tasks=["cws", "pos", "ner"])

    def full_analysis(self, texts):
        """Perform full NLP analysis"""
        return self.process_text(
            texts, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"]
        )


def load_ltp_model(model_path):
    """Load a trained LTP model"""
    processor = LTPProcessor()
    device = get_device()

    # If there's a fine-tuned model, load it
    if model_path and torch.cuda.is_available():
        try:
            processor.model.load_state_dict(torch.load(model_path, map_location=device))
            processor.model.eval()
            logger.info(f"Loaded fine-tuned LTP model from {model_path}")
        except Exception as e:
            logger.info(f"Failed to load model from {model_path}: {e}")
            logger.info("Using pre-trained model instead")

    return processor


def load_multi_label_classifier(model_path, tokenizer=None):
    """
    Load a trained MultiLabelClassifier model with an optional tokenizer

    Args:
        model_path (str): Path to the saved model
        tokenizer: Optional tokenizer to use with the classifier.
                   If None, a new AutoTokenizer for bge-m3 will be created.

    Returns:
        tuple: (model, tokenizer) - The loaded classifier model and the tokenizer
    """
    device = get_device()

    # Try to load the model
    try:
        model = MultiLabelClassifierBaseline.from_pretrained(model_path, device)
        logger.info(f"Loaded multi-label classifier from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load multi-label classifier: {e}")
        raise e

    # Get or create tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_DEFAULT_MODEL_NAME)
        logger.info(
            f"Created new tokenizer with model {TEXT_EMBEDDING_DEFAULT_MODEL_NAME}"
        )

    return model, tokenizer
