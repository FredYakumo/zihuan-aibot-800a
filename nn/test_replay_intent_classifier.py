import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModel
from nn.models import ReplyIntentClassifierModel, TextEmbedder, load_reply_intent_classifier_model, get_device


def batch_predict(texts, model, embedder):
    """Batch prediction"""
    with torch.no_grad():
        embeddings = embedder.embed(texts)
        probabilities = model(embeddings).cpu().numpy()
    return probabilities

def read_text_file(file_path):
    """Read text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文本分类预测')
    parser.add_argument('model_path', type=str, help='模型文件路径')
    parser.add_argument('input', type=str, help='预测文本或.txt文件路径')
    args = parser.parse_args()

    try:
        # Load model
        model, embedder = load_reply_intent_classifier_model(args.model_path)
        
        # Determine input type
        if args.input.endswith('.txt') and os.path.exists(args.input):
            # Batch prediction mode
            texts = read_text_file(args.input)
            if not texts:
                print("错误：文本文件为空或只包含空行")
                exit()
                
            # Predict in batches to prevent memory overflow
            batch_size = 32
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                probs = batch_predict(batch, model, embedder)
                results.extend(zip(batch, probs))
            
            # Output results
            print("\n批量预测结果：")
            print("=" * 60)
            for text, prob in results:
                truncated_text = text[:20] + '...' if len(text) > 20 else text
                print(f"文本: {truncated_text}".ljust(30-len(truncated_text)) + 
                      f"概率: {prob:>6.4f}  "
                      f"分类: {'1' if prob >= 0.5 else '0'}")
            print(f"\n总计预测 {len(results)} 条文本")
            
        else:
            # Single prediction mode
            prob = batch_predict([args.input], model, embedder)[0]
            print(f"\n预测文本：{args.input}")
            print(f"属于'1'类别的概率：{prob:.4f}")
            print(f"分类结果：{'1' if prob >= 0.5 else '1'}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {args.input}")
    except Exception as e:
        print(f"运行时错误：{str(e)}")
