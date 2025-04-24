import yaml
from logging_config import logger

class ConfigLoader:
    def __init__(self, config_path: str):
        with open(config_path) as file:
            logger.info(f"Loading config from {config_path}",)
            self.vector_db_port = self.yaml["vector_db_port"]
            logger.info(f"Vector DB Port: {self.vector_db_port}")

config = ConfigLoader("config.yaml")