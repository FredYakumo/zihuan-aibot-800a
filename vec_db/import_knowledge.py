import sys
import os
import weaviate
import pandas as pd
import jieba
import torch
import time
import pytz
import datetime

from utils.config_loader import config
from utils.logging_config import logger
from utils.string import convert_to_rfc3339
import utils.data_loader as data_loader
from vec_db.schema_collection import get_knowledge_collection, get_vec_db_client
from vec_db.nn_model import text_embedder


from weaviate.classes.query import Filter
from weaviate.collections.collection import Collection

device = text_embedder.model.device

max_batch_size = 32

def insert_knowledge_to_vec_collection(collection: Collection, knowledge_df: pd.DataFrame):
    """
    @brief Inserts knowledge from a DataFrame into the specified Weaviate collection.
    
    @param collection The Weaviate collection to insert data into.
    @param knowledge_df The DataFrame containing the knowledge to be inserted.
    """
    
    df = knowledge_df.copy()
    
    if 'key' not in df.columns:
        if 'value' in df.columns:
            logger.info("'key' column not found. Using 'value' column as 'key'.")
            df['key'] = df['value']
        else:
            logger.error("DataFrame must contain a 'key' or 'value' column for vectorization.")
            return

    # Ensure 'value' column exists.
    if 'value' not in df.columns:
        df['value'] = ''

    # Fill NaNs for all columns we use.
    df['key'] = df['key'].fillna('')
    df['value'] = df['value'].fillna('')

    if "creator_name" not in df.columns:
        df["creator_name"] = ""
    else:
        df["creator_name"] = df["creator_name"].fillna("")

    if "create_time" not in df.columns:
        df["create_time"] = None
    
    df["create_time"] = df["create_time"].apply(
        lambda x: convert_to_rfc3339(str(x)) if pd.notna(x) else datetime.datetime.now(pytz.UTC).isoformat()
    )

    all_keys = df["key"].tolist()
    
    metadata = df[["key", "value", "creator_name", "create_time"]].to_dict('records') # type: ignore

    logger.info(f"Computing key embeddings in batches of {max_batch_size}...")
    start_time = time.time()
    
    # Process embeddings in batches
    key_embeddings = []
    total_batches = (len(all_keys) + max_batch_size - 1) // max_batch_size
    
    for i in range(0, len(all_keys), max_batch_size):
        batch_end = min(i + max_batch_size, len(all_keys))
        batch_keys = all_keys[i:batch_end]
        batch_num = (i // max_batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_keys)} items)...")
        batch_embeddings = text_embedder.embed(batch_keys)
        key_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    key_embeddings = torch.cat(key_embeddings, dim=0)
    
    end_time = time.time()
    logger.info(f"Key embeddings computation complete in {end_time - start_time:.2f} seconds.")

    logger.info("Preparing data for import...")
    processed_data = []
    for i in range(len(metadata)):
        embedding = key_embeddings[i]

        processed_data.append({
            "properties": {
                "key": metadata[i]["key"],
                "value": metadata[i]["value"],
                "create_time": metadata[i]["create_time"],
                "creator_name": metadata[i]["creator_name"],
            },
            "vector": embedding.tolist()
        })
    logger.info("Data preparation complete.")

    logger.info("Importing data into vector database...")
    with collection.batch.fixed_size(10) as batch:
        for item in processed_data:
            batch.add_object(
                properties=item["properties"],
                vector=item["vector"]
            )
            if batch.number_errors > 10:
                logger.error("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        logger.error(f"Number of failed imports: {len(failed_objects)}")
        logger.error(f"First failed object: {failed_objects[0]}")
    else:
        logger.info("Import complete.")
    
def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--import-knowledge":
        if len(sys.argv) > 2:
            data_path = sys.argv[2]
        else:
            print("Usage: python import_knowledge.py --import-knowledge <data_path>")
            return
        knowledge_df = data_loader.load_knowledge_from_path(data_path)
        if knowledge_df is not None:
            with get_vec_db_client() as client:
                insert_knowledge_to_vec_collection(get_knowledge_collection(client), knowledge_df)
            logger.info("Knowledge imported successfully")
        else:
            logger.error("Failed to load knowledge from path: %s", data_path)
    else:
        print("Usage: python import_knowledge.py --import-knowledge <data_path>")


if __name__ == "__main__":
    main()
