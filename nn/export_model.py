import torch
import pnnx
import os
from nn.models import load_reply_intent_classifier_model, TextEmbeddingModel, TextEmbedder, CosineSimilarityModel,TEXT_EMBEDDING_INPUT_LENGTH


if __name__ == "__main__":
    # Create exported_model directory if it doesn't exist
    os.makedirs("exported_model", exist_ok=True)
    
    print("Exporting models to exported_model directory...")
    
    # model = load_reply_intent_classifier_model("model.pth")[0]
    # torch.onnx.export(model, torch.randn(TEXT_EMBEDDING_INPUT_LENGTH), "model.onnx", opset_version=11)
    print("Exporting reply intent classifier model...")
    net = load_reply_intent_classifier_model("model.pth")[0]
    x = torch.rand(TEXT_EMBEDDING_INPUT_LENGTH)
    mod = torch.jit.trace(net, x)
    mod.save("exported_model/model.pt")
    print("   ✓ Saved to exported_model/model.pt")
    
    # emb = TextEmbeddingModel()
    # embedder = TextEmbedder();
    # emb.eval()
    # dummy_input = embedder.tokenizer("233", return_tensors="pt", padding=True, truncation=True, max_length=TEXT_EMBEDDING_INPUT_LENGTH)
    # # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"], dummy_input["attention_mask"]))
    # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"]))
    
    print("Exporting text embedding model (without mean pooling)...")
    embedder = TextEmbedder(mean_pooling=False, device="cpu")
    token = embedder.tokenizer("0" * TEXT_EMBEDDING_INPUT_LENGTH, return_tensors='pt', padding="max_length", truncation=True)
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
    
    # Export tokenizer
    embedder.tokenizer.save_pretrained("exported_model/tokenizer")
    print("   ✓ Saved tokenizer to exported_model/tokenizer/")
    
    print("Exporting text embedding model with mean pooling...")
    embedder = TextEmbedder(mean_pooling=True)
    token = embedder.tokenizer("0" * TEXT_EMBEDDING_VEC_LENGTH, return_tensors='pt', padding=True, truncation=True)
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
    
    input_target = torch.randn(1, TEXT_EMBEDDING_VEC_LENGTH)
    input_value_list = torch.randn(3, TEXT_EMBEDDING_VEC_LENGTH)
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
    traced_cosine_model.save("exported_model/cosine_sim.pt")
    print("   ✓ Saved TorchScript to exported_model/cosine_sim.pt")
    
    print("All models exported successfully!")
    print("ONNX models for ONNX Runtime backend:")
    print("  - exported_model/text_embedding.onnx")
    print("  - exported_model/text_embedding_mean_pooling.onnx") 
    print("  - exported_model/cosine_sim.onnx")
    print("TorchScript models for LibTorch backend:")
    print("  - exported_model/text_embedding.pt")
    print("  - exported_model/text_embedding_mean_pooling.pt")
    print("  - exported_model/cosine_sim.pt")
    print("  - exported_model/model.pt")
    print("Tokenizer:")
    print("  - exported_model/tokenizer/")