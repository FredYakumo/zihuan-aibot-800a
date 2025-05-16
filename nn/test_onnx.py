# import onnxruntime as ort

from nn.models import TextEmbedder

# model = ort.InferenceSession("cosine_sim.onnx")

target = "杀猪盘"

tokenizer = TextEmbedder().tokenizer

token = tokenizer(target, return_tensors='pt', padding=True, truncation=True)
emb = TextEmbedder(device="mps")
emb.eval()
value_list = emb.model(token["input_ids"], token["attention_mask"])

print(value_list)