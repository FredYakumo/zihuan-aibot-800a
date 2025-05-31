import weaviate

from utils.config_loader import config

schema_name = "AIBot_knowledge"

if __name__ == "__main__":
    # Connect to Weaviate instance
    with weaviate.connect_to_local(port=config.vector_db_port) as client:
        
        # Delete existing SeriesNameTranslate class if it exists
        try:
            client.collections.delete(schema_name)
            print(f"Existing {schema_name} class deleted successfully")
        except weaviate.exceptions.UnexpectedStatusCodeException:
            print(f"{schema_name} class didn't exist, proceeding with creation")

        # Define the schema for the SeriesNameTranslate class
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
                    "name": "class_name_list",
                    "dataType": ["string[]"],
                    "description": "类别列表"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "知识内容"
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

        # Create the SeriesNameTranslate class in Weaviate
        client.collections.create_from_dict(schema_config)
        print(f"{schema_name} class created in Weaviate schema.")