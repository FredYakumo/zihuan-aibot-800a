#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed analysis script for comparing individual vs batch inference results.
This script helps identify why there might be differences between single and batch inference.
"""

import torch
import numpy as np
import time
from typing import List, Tuple
from .models import TextEmbedder, CosineSimilarityModel, get_device

def analyze_embedding_differences():
    """Analyze the differences between individual and batch embedding inference."""
    print("=== Detailed Analysis: Individual vs Batch Embedding Inference ===")
    
    # Test data
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    target_text = "杀猪"
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize text embedder with mean pooling
    embedder = TextEmbedder(device=device, mean_pooling=True)
    embedder.model.eval()
    
    print("\n=== Step 1: Computing Individual Embeddings ===")
    individual_embeddings = []
    individual_times = []
    
    for i, text in enumerate(batch_text):
        start_time = time.time()
        embedding = embedder.embed([text])  # Single text in list
        end_time = time.time()
        
        individual_embeddings.append(embedding[0])  # Remove batch dimension
        individual_times.append((end_time - start_time) * 1000)
        
        print(f"Individual[{i}] \"{text}\": embedding shape {embedding.shape}, time {individual_times[i]:.2f}ms")
    
    total_individual_time = sum(individual_times)
    print(f"Total individual inference time: {total_individual_time:.2f}ms")
    
    print("\n=== Step 2: Computing Batch Embeddings ===")
    start_batch = time.time()
    batch_embeddings = embedder.embed(batch_text)
    end_batch = time.time()
    batch_time = (end_batch - start_batch) * 1000
    
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Batch inference time: {batch_time:.2f}ms")
    print(f"Speed improvement: {total_individual_time / batch_time:.2f}x")
    
    print("\n=== Step 3: Analyzing Embedding Vector Differences ===")
    max_diff_overall = 0.0
    max_diff_text_idx = -1
    
    for i, text in enumerate(batch_text):
        individual_vec = individual_embeddings[i].cpu().numpy()
        batch_vec = batch_embeddings[i].cpu().numpy()
        
        # Calculate vector differences
        diff = individual_vec - batch_vec
        mean_abs_diff = np.mean(np.abs(diff))
        max_abs_diff = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        
        print(f"\nText[{i}]: \"{text}\"")
        print(f"  Individual embedding shape: {individual_vec.shape}")
        print(f"  Batch embedding shape: {batch_vec.shape}")
        print(f"  Mean absolute difference: {mean_abs_diff:.8f}")
        print(f"  Max absolute difference: {max_abs_diff:.8f}")
        print(f"  RMSE: {rmse:.8f}")
        
        if max_abs_diff > max_diff_overall:
            max_diff_overall = max_abs_diff
            max_diff_text_idx = i
        
        # Show first 10 elements comparison
        print(f"  First 10 elements comparison:")
        for j in range(min(10, len(individual_vec))):
            print(f"    [{j}]: Individual={individual_vec[j]:.8f}, Batch={batch_vec[j]:.8f}, Diff={diff[j]:.8f}")
    
    print(f"\nOverall maximum difference: {max_diff_overall:.8f} (Text[{max_diff_text_idx}])")
    
    print("\n=== Step 4: Computing Target Embedding ===")
    # Test target embedding consistency
    target_individual = embedder.embed([target_text])[0]
    target_from_batch = embedder.embed([target_text] + batch_text)[0]  # Target as first in batch
    
    target_diff = target_individual.cpu().numpy() - target_from_batch.cpu().numpy()
    target_mean_diff = np.mean(np.abs(target_diff))
    target_max_diff = np.max(np.abs(target_diff))
    
    print(f"Target text \"{target_text}\":")
    print(f"  Individual vs batch-first mean diff: {target_mean_diff:.8f}")
    print(f"  Individual vs batch-first max diff: {target_max_diff:.8f}")
    
    return individual_embeddings, batch_embeddings, target_individual

def analyze_cosine_similarity_differences(individual_embeddings, batch_embeddings, target_embedding):
    """Analyze cosine similarity differences between individual and batch approaches."""
    print("\n=== Step 5: Analyzing Cosine Similarity Differences ===")
    
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    target_text = "杀猪"
    
    # Method 1: Individual embeddings + batch cosine similarity
    print("\nMethod 1: Individual embeddings -> Batch cosine similarity")
    individual_tensor = torch.stack(individual_embeddings)  # Stack individual embeddings
    
    cosine_model = CosineSimilarityModel().to(target_embedding.device)
    cosine_model.eval()
    
    with torch.no_grad():
        scores_individual = cosine_model(target_embedding.unsqueeze(0), individual_tensor)
    
    print("Similarity results (Individual embeddings):")
    for i, (text, score) in enumerate(zip(batch_text, scores_individual.cpu().numpy())):
        print(f"  \"{target_text}\" <-> \"{text}\": {score:.6f}")
    
    # Method 2: Batch embeddings + batch cosine similarity
    print("\nMethod 2: Batch embeddings -> Batch cosine similarity")
    with torch.no_grad():
        scores_batch = cosine_model(target_embedding.unsqueeze(0), batch_embeddings)
    
    print("Similarity results (Batch embeddings):")
    for i, (text, score) in enumerate(zip(batch_text, scores_batch.cpu().numpy())):
        print(f"  \"{target_text}\" <-> \"{text}\": {score:.6f}")
    
    # Method 3: Manual CPU cosine similarity for verification
    print("\nMethod 3: Manual CPU cosine similarity (Individual embeddings)")
    target_np = target_embedding.cpu().numpy().flatten()
    
    cpu_scores_individual = []
    for i, embedding in enumerate(individual_embeddings):
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Manual cosine similarity calculation
        dot_product = np.dot(target_np, embedding_np)
        norm_target = np.linalg.norm(target_np)
        norm_embedding = np.linalg.norm(embedding_np)
        
        if norm_target == 0.0 or norm_embedding == 0.0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_target * norm_embedding)
        
        cpu_scores_individual.append(similarity)
        print(f"  \"{target_text}\" <-> \"{batch_text[i]}\": {similarity:.6f}")
    
    # Method 4: Manual CPU cosine similarity for batch embeddings
    print("\nMethod 4: Manual CPU cosine similarity (Batch embeddings)")
    cpu_scores_batch = []
    for i in range(batch_embeddings.shape[0]):
        embedding_np = batch_embeddings[i].cpu().numpy().flatten()
        
        # Manual cosine similarity calculation
        dot_product = np.dot(target_np, embedding_np)
        norm_target = np.linalg.norm(target_np)
        norm_embedding = np.linalg.norm(embedding_np)
        
        if norm_target == 0.0 or norm_embedding == 0.0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_target * norm_embedding)
        
        cpu_scores_batch.append(similarity)
        print(f"  \"{target_text}\" <-> \"{batch_text[i]}\": {similarity:.6f}")
    
    # Compare all methods
    print("\n=== Comparison of All Methods ===")
    print("Text | Individual->PyTorch | Batch->PyTorch | Individual->CPU | Batch->CPU | Diff(Ind-Batch)")
    print("-" * 100)
    
    for i, text in enumerate(batch_text):
        ind_pytorch = scores_individual[i].item()
        batch_pytorch = scores_batch[i].item()
        ind_cpu = cpu_scores_individual[i]
        batch_cpu = cpu_scores_batch[i]
        diff = abs(ind_pytorch - batch_pytorch)
        
        print(f"{i:4d} | {ind_pytorch:15.6f} | {batch_pytorch:14.6f} | {ind_cpu:15.6f} | {batch_cpu:10.6f} | {diff:13.6f}")
    
    return scores_individual, scores_batch, cpu_scores_individual, cpu_scores_batch

def analyze_tokenization_differences():
    """Analyze potential tokenization differences between individual and batch processing."""
    print("\n=== Step 6: Analyzing Tokenization Differences ===")
    
    batch_text = ["如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"]
    
    # Initialize tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
    
    print("Individual tokenization:")
    individual_tokens = []
    for i, text in enumerate(batch_text):
        tokens = tokenizer(
            [text], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        individual_tokens.append(tokens)
        print(f"Text[{i}] \"{text}\":")
        print(f"  Input IDs shape: {tokens['input_ids'].shape}")
        print(f"  Input IDs: {tokens['input_ids'].tolist()}")
        print(f"  Attention mask: {tokens['attention_mask'].tolist()}")
        print()
    
    print("Batch tokenization:")
    batch_tokens = tokenizer(
        batch_text, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    print(f"Batch input IDs shape: {batch_tokens['input_ids'].shape}")
    print(f"Batch attention mask shape: {batch_tokens['attention_mask'].shape}")
    
    for i in range(len(batch_text)):
        print(f"Text[{i}] \"{batch_text[i]}\":")
        print(f"  Input IDs: {batch_tokens['input_ids'][i].tolist()}")
        print(f"  Attention mask: {batch_tokens['attention_mask'][i].tolist()}")
        
        # Compare with individual tokenization
        individual_ids = individual_tokens[i]['input_ids'][0].tolist()
        batch_ids = batch_tokens['input_ids'][i].tolist()
        
        if individual_ids != batch_ids:
            print(f"  ⚠️  TOKENIZATION DIFFERENCE DETECTED!")
            print(f"    Individual: {individual_ids}")
            print(f"    Batch:      {batch_ids}")
        else:
            print(f"  ✅ Tokenization matches")
        print()

def main():
    """Main analysis function."""
    print("Starting detailed embedding analysis...")
    print("This analysis compares individual vs batch inference to identify sources of differences.")
    
    try:
        # Step 1-3: Analyze embedding differences
        individual_embeddings, batch_embeddings, target_embedding = analyze_embedding_differences()
        
        # Step 4-5: Analyze cosine similarity differences
        scores_ind, scores_batch, cpu_scores_ind, cpu_scores_batch = analyze_cosine_similarity_differences(
            individual_embeddings, batch_embeddings, target_embedding)
        
        # Step 6: Analyze tokenization
        analyze_tokenization_differences()
        
        print("\n=== Analysis Summary ===")
        print("Key findings:")
        print("1. If embedding vectors are identical, differences come from numerical precision")
        print("2. If tokenization differs, padding/truncation causes the differences")
        print("3. If neither differs significantly, the issue may be in model eval/training mode")
        print("4. Batch processing can introduce subtle numerical differences due to:")
        print("   - Different floating point operation order")
        print("   - Padding token influence on attention mechanisms")
        print("   - Batch normalization behavior differences")
        
        return True
        
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
