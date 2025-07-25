#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for text embedding model functionality.
This script verifies the Python implementation of text embedding models
to help debug potential issues in the C++ implementation.
"""

import torch
import numpy as np
import time
from typing import List, Tuple
from nn.models import TextEmbedder, CosineSimilarityModel, get_device

def cosine_similarity_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """
    CPU implementation of cosine similarity for verification.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def test_batch_text_embedding():
    """Test batch text embedding functionality similar to C++ TestBatchTextEmbeddingONNX."""
    print("=== Testing Batch Text Embedding ===")
    
    # Test data - same as C++ test
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    target_text = "杀猪"
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize text embedder with mean pooling
    embedder = TextEmbedder(device=device, mean_pooling=True)
    embedder.model.eval()
    
    print("Computing embeddings for similarity test...")
    
    # 计算目标文本的嵌入向量
    target_embedding = embedder.embed([target_text])
    print(f"Target embedding dim: {target_embedding.shape[1]}")
    
    # 计算候选文本的嵌入向量
    start_embed = time.time()
    batch_embeddings = embedder.embed(batch_text)
    end_embed = time.time()
    embed_duration = int((end_embed - start_embed) * 1000)
    
    print(f"Batch embedding computation took {embed_duration} ms")
    print(f"Generated {batch_embeddings.shape[0]} embeddings with dimension {batch_embeddings.shape[1]}")
    
    for i, text in enumerate(batch_text):
        print(f"Text: \"{text}\" - Embedding dim: {batch_embeddings.shape[1]}")
    
    return target_embedding, batch_embeddings

def test_cosine_similarity_model(target_embedding: torch.Tensor, batch_embeddings: torch.Tensor):
    """Test cosine similarity model functionality."""
    
    # Test data
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    target_text = "杀猪"
    
    print("Computing cosine similarities using PyTorch model...")
    
    # Initialize cosine similarity model
    device = target_embedding.device
    cosine_model = CosineSimilarityModel().to(device)
    cosine_model.eval()
    
    # Compute similarities using the model
    start_time = time.time()
    with torch.no_grad():
        similarity_scores = cosine_model(target_embedding, batch_embeddings)
    end_time = time.time()
    duration = int((end_time - start_time) * 1000)
    
    print(f"PyTorch cosine similarity computation took {duration} ms")
    print("Similarity results:")
    
    # Convert to numpy for logging
    similarity_scores_np = similarity_scores.cpu().numpy()
    
    for i, (text, score) in enumerate(zip(batch_text, similarity_scores_np)):
        print(f"  \"{target_text}\" <-> \"{text}\": {score:.6f}")
    
    # 验证结果
    assert len(similarity_scores_np) == len(batch_text), "Similarity scores count mismatch"
    
    # 验证相似度范围在[-1, 1]之间, 容许误差
    for i, score in enumerate(similarity_scores_np):
        assert score >= -1.0 - 1e-6, f"Score {i} too low: {score}"
        assert score <= 1.0 + 1e-6, f"Score {i} too high: {score}"
    
    # 验证"杀猪"与自身的相似度最高
    max_score_idx = np.argmax(similarity_scores_np)
    assert batch_text[max_score_idx] == "杀猪", f"Expected '杀猪' to have highest similarity, got '{batch_text[max_score_idx]}'"
    
    print("PyTorch cosine similarity test completed successfully!")
    return similarity_scores_np

def test_token_level_embeddings():
    """Test token-level embeddings (without mean pooling)."""
    print("=== Testing Token-Level Embeddings ===")
    
    # Test data
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    
    # Get device
    device = get_device()
    
    # Initialize text embedder without mean pooling
    print("Testing token-level embeddings without mean pooling...")
    token_embedder = TextEmbedder(device=device, mean_pooling=False)
    token_embedder.model.eval()
    
    # Test individual token embeddings
    start_time = time.time()
    individual_token_embeddings = []
    for text in batch_text:
        token_embeddings = token_embedder.embed([text])  # [1, seq_len, hidden_size]
        individual_token_embeddings.append(token_embeddings[0])  # Remove batch dimension
    end_time = time.time()
    individual_time = int((end_time - start_time) * 1000)
    
    # Test batch token embeddings
    start_time = time.time()
    batch_token_embeddings = token_embedder.embed(batch_text)  # [batch_size, seq_len, hidden_size]
    end_time = time.time()
    batch_time = int((end_time - start_time) * 1000)
    
    print(f"Individual token embedding computation took {individual_time} ms")
    print(f"Batch token embedding computation took {batch_time} ms")
    print(f"Speed ratio (individual/batch): {individual_time/batch_time:.2f}x")
    
    print("Token-level embedding test completed successfully!")
    return individual_token_embeddings, batch_token_embeddings

def main():
    """Main test function."""
    print("Starting Python text embedding model tests...")
    print("This test replicates the C++ unit test functionality to help debug potential issues.")
    
    try:
        # Test 1: Batch text embedding (similar to C++ TestBatchTextEmbeddingONNX)
        target_embedding, batch_embeddings = test_batch_text_embedding()
        
        # Test 2: Cosine similarity model
        pytorch_scores = test_cosine_similarity_model(target_embedding, batch_embeddings)
        
        # Test 3: Token-level embeddings
        individual_tokens, batch_tokens = test_token_level_embeddings()
        
        print("=== Summary ===")
        print("All Python text embedding model tests completed successfully!")
        print("If C++ tests are failing but Python tests pass, the issue is likely in:")
        print("  1. ONNX model export/conversion")
        print("  2. C++ model loading or inference")
        print("  3. C++ tokenization or preprocessing")
        print("  4. C++ tensor handling or data type conversions")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
