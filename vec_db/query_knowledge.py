import sys
import os
import weaviate
import torch
import time
from typing import List, Optional

from utils.config_loader import config
from utils.logging_config import logger
from vec_db.schema_collection import get_knowledge_collection, get_vec_db_client
from vec_db.nn_model import text_embedder
from nn.models import LTPProcessor, get_device

from weaviate.classes.query import Filter
from weaviate.collections.collection import Collection
from pydantic import BaseModel
import datetime

# Updated model to match new schema
class VecDBKnowledge(BaseModel):
    knowledge_class_filter: str
    keyword: List[str]
    content: str
    create_time: datetime.datetime | None
    creator_name: str
    certainty: float
    # Additional fields for hybrid search scoring
    vector_score: float = 0.0
    bm25_score: float = 0.0

# Initialize LTP processor for Chinese NLP tasks
device = get_device()
ltp_processor = LTPProcessor(device=device)

def extract_nouns_from_text(text: str) -> List[str]:
    """
    Extract nouns from Chinese text using LTP model
    
    Args:
        text (str): Input Chinese text
        
    Returns:
        List[str]: List of extracted nouns
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
        
        # Fallback: if no nouns found using LTP, use jieba for basic segmentation
        if not nouns and 'cws' in result and result['cws']:
            words = result['cws'][0]
            # Filter out short words and common particles
            stopwords = {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而', '也', '都', '就', '还', '又', '很', '更', '最'}
            for word in words:
                if len(word.strip()) > 1 and word.strip() not in stopwords:
                    nouns.append(word.strip())
        
        # Final fallback: use jieba if LTP completely fails
        if not nouns:
            import jieba
            words = list(jieba.cut(text))
            stopwords = {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而', '也', '都', '就', '还', '又', '很', '更', '最'}
            for word in words:
                if len(word.strip()) > 1 and word.strip() not in stopwords:
                    nouns.append(word.strip())
        
        return nouns
    except Exception as e:
        logger.warning(f"Failed to extract nouns from text '{text[:50]}...': {e}")
        # Ultimate fallback: return the original text as a single keyword
        return [text.strip()] if text.strip() else []

def query_knowledge_by_vector(
    collection: Collection, 
    query_text: str, 
    limit: int = 10,
    certainty_threshold: float = 0.85
) -> List[VecDBKnowledge]:
    """
    @brief Query knowledge from the vector database using semantic similarity.
    
    @param collection The Weaviate collection to query from.
    @param query_text The text query to search for.
    @param limit Maximum number of results to return.
    @param certainty_threshold Minimum certainty score for results.
    @return List of VecDBKnowledge objects containing matching knowledge.
    """
    
    logger.info(f"Computing query embedding for: '{query_text}'")
    start_time = time.time()
    
    # Get embedding for the query text
    query_embedding = text_embedder.embed([query_text])
    query_vector = query_embedding[0].tolist()
    
    end_time = time.time()
    logger.info(f"Query embedding computed in {end_time - start_time:.3f} seconds.")
    
    logger.info(f"Searching for similar knowledge with certainty >= {certainty_threshold}...")
    
    # Perform vector search
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        return_metadata=["certainty"]
    )
    
    results = []
    for obj in response.objects:
        certainty = obj.metadata.certainty or 0.0
        
        if certainty >= certainty_threshold:
            knowledge = VecDBKnowledge(
                knowledge_class_filter=obj.properties.get("knowledge_class_filter", ""),
                keyword=obj.properties.get("keyword", []),
                content=obj.properties.get("content", ""),
                create_time=obj.properties.get("create_time"),
                creator_name=obj.properties.get("creator_name", ""),
                certainty=certainty,
                vector_score=certainty,  # Set vector score for pure vector search
                bm25_score=0.0
            )
            results.append(knowledge)
    
    logger.info(f"Found {len(results)} knowledge items above certainty threshold.")
    return results

def query_knowledge_by_bm25(
    collection: Collection,
    query_text: str,
    limit: int = 10
) -> List[VecDBKnowledge]:
    """
    @brief Query knowledge using BM25 search on keywords extracted from query text.
    
    @param collection The Weaviate collection to query from.
    @param query_text The text query to extract keywords from.
    @param limit Maximum number of results to return.
    @return List of VecDBKnowledge objects containing matching knowledge.
    """
    
    logger.info(f"Extracting keywords from query: '{query_text}'")
    
    # Extract nouns from query text
    query_keywords = extract_nouns_from_text(query_text)
    
    if not query_keywords:
        logger.warning("No keywords extracted from query text")
        return []
    
    logger.info(f"Extracted keywords: {query_keywords}")
    
    # Perform BM25 search on keyword field
    response = collection.query.bm25(
        query=" ".join(query_keywords),
        query_properties=["keyword"],
        limit=limit,
        return_metadata=["score"]
    )
    
    results = []
    for obj in response.objects:
        # BM25 score is different from certainty, normalize it
        score = obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0
        # Convert BM25 score to a certainty-like value (0-1 range)
        certainty = min(score / 10.0, 1.0) if score > 0 else 0.0
        
        knowledge = VecDBKnowledge(
            knowledge_class_filter=obj.properties.get("knowledge_class_filter", ""),
            keyword=obj.properties.get("keyword", []),
            content=obj.properties.get("content", ""),
            create_time=obj.properties.get("create_time"),
            creator_name=obj.properties.get("creator_name", ""),
            certainty=certainty,
            vector_score=0.0,
            bm25_score=certainty  # Set BM25 score for pure BM25 search
        )
        results.append(knowledge)
    
    logger.info(f"Found {len(results)} knowledge items using BM25 search.")
    return results

def query_knowledge_hybrid(
    collection: Collection,
    query_text: str,
    vector_limit: int = 10,
    bm25_limit: int = 10,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    min_combined_score: float = 0.5
) -> List[VecDBKnowledge]:
    """
    @brief Query knowledge using hybrid search: vector similarity + BM25 keyword search.
    
    @param collection The Weaviate collection to query from.
    @param query_text The text query to search for.
    @param vector_limit Maximum number of results from vector search.
    @param bm25_limit Maximum number of results from BM25 search.
    @param vector_weight Weight for vector search results in final scoring.
    @param bm25_weight Weight for BM25 search results in final scoring.
    @param min_combined_score Minimum combined score threshold for results.
    @return List of VecDBKnowledge objects containing matching knowledge, sorted by combined score.
    """
    
    logger.info(f"Performing hybrid search for: '{query_text}'")
    logger.info(f"Vector weight: {vector_weight}, BM25 weight: {bm25_weight}")
    
    # Perform vector search with lower certainty threshold for hybrid approach
    vector_results = query_knowledge_by_vector(
        collection, query_text, 
        limit=vector_limit, 
        certainty_threshold=0.6  # Lower threshold for hybrid
    )
    
    # Perform BM25 search
    bm25_results = query_knowledge_by_bm25(collection, query_text, limit=bm25_limit)
    
    # Combine results using content as unique key
    combined_results = {}
    
    # Add vector search results
    for result in vector_results:
        content_key = result.content
        if content_key not in combined_results:
            # Create a new result with updated scores
            combined_results[content_key] = VecDBKnowledge(
                knowledge_class_filter=result.knowledge_class_filter,
                keyword=result.keyword,
                content=result.content,
                create_time=result.create_time,
                creator_name=result.creator_name,
                certainty=result.certainty,
                vector_score=result.certainty,
                bm25_score=0.0
            )
        else:
            # Update vector score if higher
            if result.certainty > combined_results[content_key].vector_score:
                # Create updated result with new vector score
                existing = combined_results[content_key]
                combined_results[content_key] = VecDBKnowledge(
                    knowledge_class_filter=existing.knowledge_class_filter,
                    keyword=existing.keyword,
                    content=existing.content,
                    create_time=existing.create_time,
                    creator_name=existing.creator_name,
                    certainty=existing.certainty,
                    vector_score=result.certainty,
                    bm25_score=existing.bm25_score
                )
    
    # Add BM25 search results
    for result in bm25_results:
        content_key = result.content
        if content_key not in combined_results:
            # Create a new result with updated scores
            combined_results[content_key] = VecDBKnowledge(
                knowledge_class_filter=result.knowledge_class_filter,
                keyword=result.keyword,
                content=result.content,
                create_time=result.create_time,
                creator_name=result.creator_name,
                certainty=result.certainty,
                vector_score=0.0,
                bm25_score=result.certainty
            )
        else:
            # Update BM25 score if higher
            if result.certainty > combined_results[content_key].bm25_score:
                # Create updated result with new BM25 score
                existing = combined_results[content_key]
                combined_results[content_key] = VecDBKnowledge(
                    knowledge_class_filter=existing.knowledge_class_filter,
                    keyword=existing.keyword,
                    content=existing.content,
                    create_time=existing.create_time,
                    creator_name=existing.creator_name,
                    certainty=existing.certainty,
                    vector_score=existing.vector_score,
                    bm25_score=result.certainty
                )
    
    # Calculate combined scores and filter
    final_results = []
    for content_key, result in combined_results.items():
        vector_score = result.vector_score
        bm25_score = result.bm25_score
        
        # Calculate weighted combined score
        combined_score = (vector_weight * vector_score) + (bm25_weight * bm25_score)
        
        if combined_score >= min_combined_score:
            # Create final result with combined score
            final_result = VecDBKnowledge(
                knowledge_class_filter=result.knowledge_class_filter,
                keyword=result.keyword,
                content=result.content,
                create_time=result.create_time,
                creator_name=result.creator_name,
                certainty=combined_score,
                vector_score=vector_score,
                bm25_score=bm25_score
            )
            final_results.append(final_result)
    
    # Sort by combined score (certainty) in descending order
    final_results.sort(key=lambda x: x.certainty, reverse=True)
    
    logger.info(f"Hybrid search found {len(final_results)} knowledge items above combined score threshold.")
    logger.info(f"Vector results: {len(vector_results)}, BM25 results: {len(bm25_results)}, Combined: {len(final_results)}")
    
    return final_results

def query_knowledge_by_filter(
    collection: Collection,
    keyword_filter: Optional[str] = None,
    content_filter: Optional[str] = None,
    creator_filter: Optional[str] = None,
    class_filter: Optional[str] = None,
    limit: int = 10
) -> List[VecDBKnowledge]:
    """
    @brief Query knowledge using property filters.
    
    @param collection The Weaviate collection to query from.
    @param keyword_filter Filter by keyword containing this string.
    @param content_filter Filter by content containing this string.
    @param creator_filter Filter by creator name.
    @param class_filter Filter by knowledge class.
    @param limit Maximum number of results to return.
    @return List of VecDBKnowledge objects containing matching knowledge.
    """
    
    logger.info("Querying knowledge by filters...")
    
    # Build filter conditions
    filters = []
    if keyword_filter:
        filters.append(Filter.by_property("keyword").contains_any([keyword_filter]))
    if content_filter:
        filters.append(Filter.by_property("content").contains_any([content_filter]))
    if creator_filter:
        filters.append(Filter.by_property("creator_name").equal(creator_filter))
    if class_filter:
        filters.append(Filter.by_property("knowledge_class_filter").equal(class_filter))
    
    # Combine filters with AND operation
    combined_filter = None
    if len(filters) == 1:
        combined_filter = filters[0]
    elif len(filters) > 1:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
    
    # Perform filtered search
    if combined_filter:
        response = collection.query.fetch_objects(
            where=combined_filter,
            limit=limit
        )
    else:
        response = collection.query.fetch_objects(limit=limit)
    
    results = []
    for obj in response.objects:
        knowledge = VecDBKnowledge(
            knowledge_class_filter=obj.properties.get("knowledge_class_filter", ""),
            keyword=obj.properties.get("keyword", []),
            content=obj.properties.get("content", ""),
            create_time=obj.properties.get("create_time"),
            creator_name=obj.properties.get("creator_name", ""),
            certainty=1.0,  # No certainty for filter-based queries
            vector_score=0.0,
            bm25_score=0.0
        )
        results.append(knowledge)
    
    logger.info(f"Found {len(results)} knowledge items matching filters.")
    return results

def display_results(results: List[VecDBKnowledge]):
    """
    @brief Display query results in a formatted way.
    
    @param results List of VecDBKnowledge objects to display.
    """
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results)} results:")
    print(f"{'='*80}")
    
    for i, knowledge in enumerate(results, 1):
        print(f"\n[{i}] Class: {knowledge.knowledge_class_filter}")
        print(f"    Keywords: {', '.join(knowledge.keyword[:10])}{'...' if len(knowledge.keyword) > 10 else ''}")
        print(f"    Content: {knowledge.content[:200]}{'...' if len(knowledge.content) > 200 else ''}")
        print(f"    Creator: {knowledge.creator_name}")
        print(f"    Create Time: {knowledge.create_time}")
        
        # Show certainty/score info
        if knowledge.certainty < 1.0:
            print(f"    Combined Score: {knowledge.certainty:.3f}")
        
        # Show detailed scores for hybrid search
        if knowledge.vector_score > 0.0 or knowledge.bm25_score > 0.0:
            print(f"    Vector Score: {knowledge.vector_score:.3f}, BM25 Score: {knowledge.bm25_score:.3f}")
            
        print("-" * 80)

def interactive_query():
    """
    @brief Interactive query mode for exploring the knowledge base.
    """
    print("=== Knowledge Base Interactive Query ===")
    print("Commands:")
    print("  hybrid <text>     - Hybrid search (vector + BM25 on keywords)")
    print("  vector <text>     - Search by semantic similarity only")
    print("  bm25 <text>       - Search by BM25 on keywords only")
    print("  filter <content>  - Search by content filter")
    print("  keyword <keyword> - Search by keyword filter")
    print("  creator <name>    - Search by creator name")
    print("  class <class>     - Search by knowledge class")
    print("  quit              - Exit")
    print("=" * 40)
    
    with get_vec_db_client() as client:
        collection = get_knowledge_collection(client)
        
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                
                if not user_input or user_input.lower() == 'quit':
                    break
                
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                
                if command == 'hybrid' and len(parts) > 1:
                    query_text = parts[1]
                    results = query_knowledge_hybrid(collection, query_text)
                    display_results(results)
                    
                elif command == 'vector' and len(parts) > 1:
                    query_text = parts[1]
                    results = query_knowledge_by_vector(collection, query_text)
                    display_results(results)
                    
                elif command == 'bm25' and len(parts) > 1:
                    query_text = parts[1]
                    results = query_knowledge_by_bm25(collection, query_text)
                    display_results(results)
                    
                elif command == 'filter' and len(parts) > 1:
                    content_filter = parts[1]
                    results = query_knowledge_by_filter(collection, content_filter=content_filter)
                    display_results(results)
                    
                elif command == 'keyword' and len(parts) > 1:
                    keyword_filter = parts[1]
                    results = query_knowledge_by_filter(collection, keyword_filter=keyword_filter)
                    display_results(results)
                    
                elif command == 'creator' and len(parts) > 1:
                    creator_filter = parts[1]
                    results = query_knowledge_by_filter(collection, creator_filter=creator_filter)
                    display_results(results)
                    
                elif command == 'class' and len(parts) > 1:
                    class_filter = parts[1]
                    results = query_knowledge_by_filter(collection, class_filter=class_filter)
                    display_results(results)
                    
                else:
                    print("Invalid command. Use 'hybrid <text>', 'vector <text>', 'bm25 <text>', 'filter <content>', 'keyword <keyword>', 'creator <name>', 'class <class>', or 'quit'.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during query: {e}")
                print(f"Error: {e}")
    
    print("\nGoodbye!")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python query_knowledge.py --hybrid <query_text>")
        print("  python query_knowledge.py --vector <query_text>")
        print("  python query_knowledge.py --bm25 <query_text>")
        print("  python query_knowledge.py --filter <content_filter>")
        print("  python query_knowledge.py --keyword <keyword_filter>")
        print("  python query_knowledge.py --creator <creator_name>")
        print("  python query_knowledge.py --class <class_filter>")
        print("  python query_knowledge.py --interactive")
        return
    
    command = sys.argv[1]
    
    if command == "--interactive":
        interactive_query()
        return
    
    if len(sys.argv) < 3:
        print("Error: Missing query parameter")
        return
    
    query_param = sys.argv[2]
    
    with get_vec_db_client() as client:
        collection = get_knowledge_collection(client)
        
        if command == "--hybrid":
            results = query_knowledge_hybrid(collection, query_param)
            display_results(results)
            
        elif command == "--vector":
            results = query_knowledge_by_vector(collection, query_param)
            display_results(results)
            
        elif command == "--bm25":
            results = query_knowledge_by_bm25(collection, query_param)
            display_results(results)
            
        elif command == "--filter":
            results = query_knowledge_by_filter(collection, content_filter=query_param)
            display_results(results)
            
        elif command == "--keyword":
            results = query_knowledge_by_filter(collection, keyword_filter=query_param)
            display_results(results)
            
        elif command == "--creator":
            results = query_knowledge_by_filter(collection, creator_filter=query_param)
            display_results(results)
            
        elif command == "--class":
            results = query_knowledge_by_filter(collection, class_filter=query_param)
            display_results(results)
            
        else:
            print(f"Unknown command: {command}")
            print("Use --hybrid, --vector, --bm25, --filter, --keyword, --creator, --class, or --interactive")

if __name__ == "__main__":
    main()
