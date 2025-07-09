import torch
import pnnx
from nn.models import load_reply_intent_classifier_model, TextEmbeddingModel, TextEmbedder, CosineSimilarityModel


if __name__ == "__main__":
    # model = load_reply_intent_classifier_model("model.pth")[0]
    # torch.onnx.export(model, torch.randn(1024), "model.onnx", opset_version=11)
    net = load_reply_intent_classifier_model("model.pth")[0]
    x = torch.rand(1024)
    mod = torch.jit.trace(net, x)
    mod.save("exported_model/model.pt")
    
    # emb = TextEmbeddingModel()
    # embedder = TextEmbedder();
    # emb.eval()
    # dummy_input = embedder.tokenizer("233", return_tensors="pt", padding=True, truncation=True, max_length=1024)
    # # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"], dummy_input["attention_mask"]))
    # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"]))
    embedder = TextEmbedder(mean_pooling=False, device="cpu")
    token = embedder.tokenizer("0" * 1024, return_tensors='pt', padding=True, truncation=True)
    print(token)
    torch.onnx.export(embedder.model, (token["input_ids"], token["attention_mask"])
                      , "exported_model/text_embedding.onnx" , opset_version=14, input_names=["input_ids", "attention_mask"],
                      output_names=["output"],
                      dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    })
    embedder.tokenizer.save_pretrained("tokenizer")
    
    embedder = TextEmbedder(mean_pooling=True)
    token = embedder.tokenizer("0" * 1024, return_tensors='pt', padding=True, truncation=True)
    print(token)
    torch.onnx.export(embedder.model, (token["input_ids"], token["attention_mask"])
                      , "exported_model/text_embedding_mean_pooling.onnx" , opset_version=14, input_names=["input_ids", "attention_mask"],
                      output_names=["output"],
                      dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    })
    


    dynamic_axes = {
        "target": {1: "feature_dim"},
        "value_list": {0: "batch_size", 1: "feature_dim"},
        "output": {0: "batch_size"}
    }
    model = CosineSimilarityModel()
    
    input_target = torch.randn(1, 1024)
    input_value_list = torch.randn(3, 1024)
    torch.onnx.export(
  model,
        (input_target, input_value_list),
        "exported_model/cosine_sim.onnx",
        input_names=["target", "value_list"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )
    
    pnnx.export(model, "exported_model/cosine_sim.pt", (input_target, input_value_list))