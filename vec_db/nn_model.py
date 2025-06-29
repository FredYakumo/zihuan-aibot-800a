from nn.models import TextEmbedder, get_device
from utils.logging_config import logger
import time
from utils.config_loader import config

__all__ = ["text_embedder"]

start_time = time.time()
text_embedder = TextEmbedder(device=get_device())
end_time = time.time()
logger.info(f"Text embedder initialized in {end_time - start_time} seconds")