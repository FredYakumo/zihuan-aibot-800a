from weaviate.collections.collection import Collection
from utils.config_loader import config
from utils.logging_config import logger
import weaviate
from weaviate.client import WeaviateClient
import datetime
from typing import List

from pydantic import BaseModel

def get_vec_db_client() -> WeaviateClient:
    return weaviate.connect_to_local(port=config.vector_db_port)

class Knowledge(BaseModel):
    key: str
    value: str
    create_time: datetime.datetime | None
    creator_name: str
    vector_source: str  # "key" or "value" to indicate which field was used for vectorization
    
class VecDBKnowledge(BaseModel):
    key: str
    value: str
    create_time: datetime.datetime | None
    creator_name: str
    vector_source: str
    certainty: float

default_knowledge_schema_name = "AIBotKnowledge"

def create_knowledge_schema(client: WeaviateClient, schema_name: str = config.knowledge_collection_name or default_knowledge_schema_name):
    # Delete existing knowledge class if it exists
    try:
        client.collections.delete(schema_name)
        print(f"Existing {schema_name} class deleted successfully")
    except weaviate.exceptions.UnexpectedStatusCodeException:
        print(f"{schema_name} class didn't exist, proceeding with creation")

    # Define the schema for the knowledge class
    schema_config = {
        "class": schema_name,
        "vectorIndexConfig": {
            "distance": "cosine",
            "ef": -1,
            "efConstruction": 128,
            "maxConnections": 64,
            "vectorCacheMaxObjects": 500000,
            "dimensions": 1024
        },
        "properties": [
            {
                "name": "knowledge_class_filter",
                "dataType": ["string"],
                "description": "知识筛选类别"
            },
            {
                "name": "keyword",
                "dataType": ["string[]"],
                "description": "知识键"
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "知识值"
            },
            {
                "name": "create_time",
                "dataType": ["date"],
                "description": "创建时间"
            },
            {
                "name": "creator_name",
                "dataType": ["string"],
                "description": "创建者名称"
            }
        ],
    }

    # Create the knowledge class in Weaviate
    client.collections.create_from_dict(schema_config)
    print(f"{schema_name} class created in Weaviate schema.")

def get_knowledge_collection(client: WeaviateClient, schema_name: str = config.knowledge_collection_name or default_knowledge_schema_name) -> Collection:
    return client.collections.get(schema_name)

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-knowledge-schema":
        with get_vec_db_client() as client:
            create_knowledge_schema(client)
        logger.info("Knowledge schema created successfully")
    else:
        print("Usage: python schema_collection.py -create-knowledge-schema")

if __name__ == "__main__":
    main()