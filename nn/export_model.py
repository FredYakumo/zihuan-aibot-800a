import torch
import pnnx
import os
from nn.models import load_reply_intent_classifier_model, TextEmbeddingModel, TextEmbedder, CosineSimilarityModel


if __name__ == "__main__":
    # Create exported_model directory if it doesn't exist
    os.makedirs("exported_model", exist_ok=True)
    
    print("Exporting models to exported_model directory...")
    
    # model = load_reply_intent_classifier_model("model.pth")[0]
    # torch.onnx.export(model, torch.randn(1024), "model.onnx", opset_version=11)
    print("Exporting reply intent classifier model...")
    net = load_reply_intent_classifier_model("model.pth")[0]
    x = torch.rand(1024)
    mod = torch.jit.trace(net, x)
    mod.save("exported_model/model.pt")
    print("   ✓ Saved to exported_model/model.pt")
    
    # emb = TextEmbeddingModel()
    # embedder = TextEmbedder();
    # emb.eval()
    # dummy_input = embedder.tokenizer("233", return_tensors="pt", padding=True, truncation=True, max_length=1024)
    # # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"], dummy_input["attention_mask"]))
    # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"]))
    
    print("Exporting text embedding model (without mean pooling)...")
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
    print("   ✓ Saved ONNX to exported_model/text_embedding.onnx")
    
    # Export TorchScript model for text embedding (without mean pooling)
    embedder.model.eval()
    traced_model = torch.jit.trace(embedder.model, (token["input_ids"], token["attention_mask"]), strict=False)
    traced_model.save("exported_model/text_embedding.pt")
    print("   ✓ Saved TorchScript to exported_model/text_embedding.pt")
    
    embedder.tokenizer.save_pretrained("tokenizer")
    print("   ✓ Saved tokenizer to tokenizer/")
    
    print("Exporting text embedding model with mean pooling...")
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
    print("   ✓ Saved ONNX to exported_model/text_embedding_mean_pooling.onnx")
    
    # Export TorchScript model for text embedding with mean pooling
    embedder.model.eval()
    traced_model_mean_pooling = torch.jit.trace(embedder.model, (token["input_ids"], token["attention_mask"]), strict=False)
    traced_model_mean_pooling.save("exported_model/text_embedding_mean_pooling.pt")
    print("   ✓ Saved TorchScript to exported_model/text_embedding_mean_pooling.pt")
    


    print("Exporting cosine similarity model...")
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
    print("   ✓ Saved ONNX to exported_model/cosine_sim.onnx")
    
    # Export TorchScript model for cosine similarity
    model.eval()
    traced_cosine_model = torch.jit.trace(model, (input_target, input_value_list))
    traced_cosine_model.save("exported_model/cosine_sim.bin")
    print("   ✓ Saved TorchScript to exported_model/cosine_sim.bin")
    
    print("All models exported successfully!")
    print("ONNX models for ONNX Runtime backend:")
    print("  - exported_model/text_embedding.onnx")
    print("  - exported_model/text_embedding_mean_pooling.onnx") 
    print("  - exported_model/cosine_sim.onnx")
    print("TorchScript models for LibTorch backend:")
    print("  - exported_model/text_embedding.pt")
    print("  - exported_model/text_embedding_mean_pooling.pt")
    print("  - exported_model/cosine_sim.bin")
    print("  - exported_model/model.pt")