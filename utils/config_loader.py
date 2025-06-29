import yaml
from utils.logging_config import logger

class ConfigLoader:
    def __init__(self, config_path: str):
        with open(config_path) as file:
            logger.info(f"Loading config from {config_path}",)
            self.yaml = yaml.safe_load(file)
            self.vector_db_port: int = self.yaml["vector_db_port"]
            logger.info(f"Vector DB Port: {self.vector_db_port}")
            self.knowledge_collection_name: str = self.yaml["knowledge_collection_name"]
            logger.info(f"Knowledge Collection Name: {self.knowledge_collection_name}")
            self.nn = self.yaml["nn"]
            logger.info(f"NN config: {self.nn}")

config = ConfigLoader("config.yaml")