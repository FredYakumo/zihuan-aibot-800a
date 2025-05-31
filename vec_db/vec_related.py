from transformers import AutoTokenizer, AutoModel
import torch
from nn.models import CosineSimilarityModel

tokenizer = AutoTokenizer.from_pretrained("GanymedeNil/text2vec-large-chinese")
model = AutoModel.from_pretrained("GanymedeNil/text2vec-large-chinese")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


sentences = ["ISO/IEC C++ Unofficial 🌱②", "复位: 十瓶药是ISO/IEC C++ Unofficial 🌱②群里一个人做的神秘游戏"]
t = "ISO/IEC C++ Unofficial 🌱②"
encoded_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_target = tokenizer(t, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output_sentences = model(**encoded_sentences)
    model_output_target = model(**encoded_target)
# Perform pooling. In this case, mean pooling.


cosine_similarity_model = CosineSimilarityModel()

sentence_embeddings = mean_pooling(model_output_sentences, encoded_sentences['attention_mask'])
target_embeddings = mean_pooling(model_output_target, encoded_target['attention_mask'])

result = cosine_similarity_model(target_embeddings, sentence_embeddings)
print(f"Similarity between sentences: {result}")