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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze()


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
        print(f"Max position embedding: {self.config.max_position_embeddings}")
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


class MultiLabelClassifier(nn.Module):
    """
    An RNN-based multi-label classifier.

    This model takes embeddings as input [seq_len, embedding_size=1024] and processes them
    through a bidirectional LSTM to produce logits for multi-label classification.
    The RNN architecture allows the model to capture sequential relationships
    and long-range dependencies in the input sequence.
    """

    def __init__(
        self,
        embedding_dim=1024,
        num_classes=10,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # * 2 for bidirectional
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, mask=None):
        """
        Forward pass for the RNN-based classifier.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, seq_len, embedding_dim]
            mask (torch.Tensor, optional): Attention mask for padding. Shape [batch_size, seq_len]
                                         with 0 for padding positions and 1 for non-padding.

        Returns:
            torch.Tensor: Logits for each class with shape [batch_size, num_classes]
        """
        seq_len = x.size(1)

        # truncate sequence if longer than max_seq_len
        if seq_len > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            seq_len = self.max_seq_len
            if mask is not None:
                mask = mask[:, : self.max_seq_len]

        # pack padded sequence for more efficient and correct RNN calculation
        if mask is not None:

            lengths = mask.sum(dim=1).cpu()

            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

            packed_output, (hidden, _) = self.lstm(packed_x)

            # unpack output
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

            # get the last hidden states from both directions
            # hidden shape: [num_layers * num_directions, batch, hidden_size]
            last_hidden = torch.cat(
                [hidden[-2], hidden[-1]], dim=1
            )  # Take last layer, both directions
        else:
            # if no mask, process sequence normally
            output, (hidden, _) = self.lstm(x)
            # get the last hidden states from both directions
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        logits = self.classifier(last_hidden)

        return logits


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
