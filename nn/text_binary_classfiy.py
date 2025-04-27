import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModel

class TextEmbedder:
    """Text embedding generator"""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def embed(self, texts):
        """Generate text embeddings in batch"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

class ClassificationModel(torch.nn.Module):
    """Classification model structure"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

def load_model(model_path):
    """Load the trained model"""
    embedder = TextEmbedder()
    model = ClassificationModel(embedder.model.config.hidden_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, embedder

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
        model, embedder = load_model(args.model_path)
        
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
                print(f"文本：{text[:20] + '...' if len(text) > 20 else text}".ljust(55) + 
                     f"概率：{prob:.4f}".ljust(15) + 
                     f"分类：{'1' if prob >= 0.5 else '0'}")
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
