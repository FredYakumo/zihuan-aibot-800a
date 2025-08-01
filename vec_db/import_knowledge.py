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
from nn.models import LTPProcessor, get_device


from weaviate.classes.query import Filter
from weaviate.collections.collection import Collection

device = text_embedder.model.device

max_batch_size = 32

# Initialize LTP processor for Chinese NLP tasks
ltp_processor = LTPProcessor(device=device)

def extract_nouns_from_text(text: str) -> list:
    """
    Extract nouns from Chinese text using LTP model
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        list: List of extracted nouns
    """
    if not text or not text.strip():
        return []
    
    try:
        # Use LTP for word segmentation and POS tagging
        result = ltp_processor.pos_tagging([text])
        
        nouns = []
        if 'cws' in result and 'pos' in result and result['cws'] and result['pos']:
            words = result['cws'][0]  # First (and only) sentence's words
            pos_tags = result['pos'][0]  # First (and only) sentence's POS tags
            
            # Extract words with noun POS tags
            # Common Chinese noun POS tags: n, nr, ns, nt, nz, etc.
            noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'ng', 'nh', 'ni', 'nl', 'nm'}
            
            for word, pos in zip(words, pos_tags):
                if pos in noun_tags and len(word.strip()) > 0:
                    nouns.append(word.strip())
        
        return nouns
    except Exception as e:
        logger.warning(f"Failed to extract nouns from text '{text[:50]}...': {e}")
        return []

def parse_spec_keywords(spec_keyword_str: str) -> list:
    """
    Parse spec_keyword string that is pipe-separated
    
    Args:
        spec_keyword_str (str): Pipe-separated string of keywords
        
    Returns:
        list: List of keywords
    """
    if not spec_keyword_str or pd.isna(spec_keyword_str):
        return []
    
    keywords = [kw.strip() for kw in str(spec_keyword_str).split('|') if kw.strip()]
    return keywords

def insert_knowledge_to_vec_collection(collection: Collection, knowledge_df: pd.DataFrame):
    """
    @brief Inserts knowledge from a DataFrame into the specified Weaviate collection.
    
    @param collection The Weaviate collection to insert data into.
    @param knowledge_df The DataFrame containing the knowledge to be inserted.
                       Expected columns for asteria_readme.csv: 
                       - spec_keyword: pipe-separated keywords (can be empty)
                       - content: main content text
                       - create_time: creation timestamp 
                       - dimension: knowledge category/filter
                       - creator_name: creator name
    """
    
    df = knowledge_df.copy()
    
    # Map CSV columns to schema fields
    # For asteria_readme.csv: spec_keyword,content,create_time,dimension,creator_name
    column_mappings = {
        'content': 'content',           # content -> content  
        'dimension': 'knowledge_class_filter',  # dimension -> knowledge_class_filter
        'creator_name': 'creator_name', # creator_name -> creator_name
        'create_time': 'create_time',   # create_time -> create_time
        'spec_keyword': 'spec_keyword'  # spec_keyword -> spec_keyword (for processing)
    }
    
    # Check if required columns exist
    if 'content' not in df.columns:
        logger.error("DataFrame must contain a 'content' column.")
        return
    
    # Fill missing columns with defaults
    for original_col in column_mappings.keys():
        if original_col not in df.columns:
            if original_col == 'spec_keyword':
                df[original_col] = ''
            elif original_col == 'dimension':
                df[original_col] = 'general'
            elif original_col == 'creator_name':
                df[original_col] = 'unknown'
            elif original_col == 'create_time':
                df[original_col] = datetime.datetime.now(pytz.UTC)
    
    # Fill NaNs for all columns we use
    df['content'] = df['content'].fillna('')
    df['spec_keyword'] = df['spec_keyword'].fillna('')
    df['creator_name'] = df['creator_name'].fillna('unknown')
    df['dimension'] = df['dimension'].fillna('general')
    
    # Handle create_time conversion
    df["create_time"] = df["create_time"].apply(
        lambda x: convert_to_rfc3339(str(x)) if pd.notna(x) else datetime.datetime.now(pytz.UTC).isoformat()
    )

    # Prepare texts for embedding and extract keywords
    texts_to_embed = []
    record_metadata = []
    
    logger.info(f"Processing {len(df)} records for keyword extraction and embedding...")
    
    for idx, row in df.iterrows():
        content = row["content"].strip()
        if not content:
            logger.warning(f"Skipping row {idx} with empty content")
            continue
            
        # Extract keywords from spec_keyword (pipe-separated)
        spec_keywords = parse_spec_keywords(row["spec_keyword"])
        
        # Extract nouns from content using LTP
        content_nouns = extract_nouns_from_text(content)
        
        # Combine and deduplicate keywords
        all_keywords = list(set(spec_keywords + content_nouns))
        
        # Filter out empty keywords
        all_keywords = [kw for kw in all_keywords if kw.strip()]
        
        logger.debug(f"Row {idx}: spec_keywords={spec_keywords}, content_nouns={content_nouns[:5]}..., final_keywords={len(all_keywords)}")
        
        # Use content for vectorization
        texts_to_embed.append(content)
        record_metadata.append({
            "knowledge_class_filter": row["dimension"],
            "keyword": all_keywords,
            "content": content,
            "create_time": row["create_time"],
            "creator_name": row["creator_name"]
        })

    if not texts_to_embed:
        logger.error("No valid content found for embedding")
        return

    logger.info(f"Computing embeddings for {len(texts_to_embed)} texts in batches of {max_batch_size}...")
    start_time = time.time()
    
    # Process embeddings in batches
    all_embeddings = []
    total_batches = (len(texts_to_embed) + max_batch_size - 1) // max_batch_size
    
    for i in range(0, len(texts_to_embed), max_batch_size):
        batch_end = min(i + max_batch_size, len(texts_to_embed))
        batch_texts = texts_to_embed[i:batch_end]
        batch_num = (i // max_batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} items)...")
        batch_embeddings = text_embedder.embed(batch_texts)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batch embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    end_time = time.time()
    logger.info(f"Embeddings computation complete in {end_time - start_time:.2f} seconds.")

    logger.info(f"Preparing data for import...")
    processed_data = []
    for i, metadata in enumerate(record_metadata):
        embedding = all_embeddings[i]

        processed_data.append({
            "properties": {
                "knowledge_class_filter": metadata["knowledge_class_filter"],
                "keyword": metadata["keyword"],
                "content": metadata["content"],
                "create_time": metadata["create_time"],
                "creator_name": metadata["creator_name"],
            },
            "vector": embedding.tolist()
        })
    
    logger.info(f"Data preparation complete. Total records to import: {len(processed_data)}")
    logger.info(f"Sample record keywords: {processed_data[0]['properties']['keyword'][:5] if processed_data[0]['properties']['keyword'] else 'No keywords'}")

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
