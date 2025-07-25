import torch
import pnnx
import os
from nn.models import (
    load_reply_intent_classifier_model, 
    TextEmbeddingModel, 
    TextEmbedder, 
    CosineSimilarityModel,
    LTPModel,
    LTPProcessor,
    TEXT_EMBEDDING_INPUT_LENGTH, 
    TEXT_EMBEDDING_OUTPUT_LENGTH,
    LTP_MAX_INPUT_LENGTH
)
from utils.logging_config import logger


if __name__ == "__main__":
    # Create exported_model directory if it doesn't exist
    os.makedirs("exported_model", exist_ok=True)
    
    logger.info("Exporting models to exported_model directory...")
    
    # model = load_reply_intent_classifier_model("model.pth")[0]
    # torch.onnx.export(model, torch.randn(TEXT_EMBEDDING_INPUT_LENGTH), "model.onnx", opset_version=11)
    logger.info("Exporting reply intent classifier model...")
    net = load_reply_intent_classifier_model("model.pth")[0]
    x = torch.rand(TEXT_EMBEDDING_OUTPUT_LENGTH)
    mod = torch.jit.trace(net, x)
    mod.save("exported_model/model.pt")
    logger.info("   ✓ Saved to exported_model/model.pt")
    
    # emb = TextEmbeddingModel()
    # embedder = TextEmbedder();
    # emb.eval()
    # dummy_input = embedder.tokenizer("233", return_tensors="pt", padding=True, truncation=True, max_length=TEXT_EMBEDDING_INPUT_LENGTH)
    # # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"], dummy_input["attention_mask"]))
    # opt_model = pnnx.export(emb, "embedding.pt", (dummy_input["input_ids"]))
    
    logger.info("Exporting text embedding model (without mean pooling)...")
    embedder = TextEmbedder(mean_pooling=False, device="cpu")
    token = embedder.tokenizer("0" * TEXT_EMBEDDING_INPUT_LENGTH, return_tensors='pt', 
                            #    padding="max_length", 
                               padding=True,
                               truncation=True)
    logger.info(token)
    torch.onnx.export(embedder.model, (token["input_ids"], token["attention_mask"])
                      , "exported_model/text_embedding.onnx" , opset_version=14, input_names=["input_ids", "attention_mask"],
                      output_names=["output"],
                      dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    })
    logger.info("   ✓ Saved ONNX to exported_model/text_embedding.onnx")
    
    # Export TorchScript model for text embedding (without mean pooling)
    embedder.model.eval()
    traced_model = torch.jit.trace(embedder.model, (token["input_ids"], token["attention_mask"]), strict=False)
    traced_model.save("exported_model/text_embedding.pt")
    logger.info("   ✓ Saved TorchScript to exported_model/text_embedding.pt")
    
    # Export tokenizer
    embedder.tokenizer.save_pretrained("exported_model/tokenizer")
    logger.info("   ✓ Saved tokenizer to exported_model/tokenizer/")
    
    logger.info("Exporting text embedding model with mean pooling...")
    embedder = TextEmbedder(mean_pooling=True)
    token = embedder.tokenizer("0" * TEXT_EMBEDDING_INPUT_LENGTH, return_tensors='pt', 
                               padding=True, 
                               truncation=True)
    logger.info(token)
    torch.onnx.export(embedder.model, (token["input_ids"], token["attention_mask"])
                      , "exported_model/text_embedding_mean_pooling.onnx" , opset_version=14, input_names=["input_ids", "attention_mask"],
                      output_names=["output"],
                      dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    })
    logger.info("   ✓ Saved ONNX to exported_model/text_embedding_mean_pooling.onnx")
    
    # Export TorchScript model for text embedding with mean pooling
    embedder.model.eval()
    traced_model_mean_pooling = torch.jit.trace(embedder.model, (token["input_ids"], token["attention_mask"]), strict=False)
    traced_model_mean_pooling.save("exported_model/text_embedding_mean_pooling.pt")
    logger.info("   ✓ Saved TorchScript to exported_model/text_embedding_mean_pooling.pt")
    


    logger.info("Exporting cosine similarity model...")
    dynamic_axes = {
        "target": {1: "feature_dim"},
        "value_list": {0: "batch_size", 1: "feature_dim"},
        "output": {0: "batch_size"}
    }
    model = CosineSimilarityModel()
    
    input_target = torch.randn(1, TEXT_EMBEDDING_OUTPUT_LENGTH)
    input_value_list = torch.randn(3, TEXT_EMBEDDING_OUTPUT_LENGTH)
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
    logger.info("   ✓ Saved ONNX to exported_model/cosine_sim.onnx")
    
    # Export TorchScript model for cosine similarity
    model.eval()
    traced_cosine_model = torch.jit.trace(model, (input_target, input_value_list))
    traced_cosine_model.save("exported_model/cosine_sim.pt")
    logger.info("   ✓ Saved TorchScript to exported_model/cosine_sim.pt")
    
    logger.info("Exporting LTP model...")
    try:
        ltp_processor = LTPProcessor(device="cpu")
        
        if ltp_processor.tokenizer is None:
            logger.warning("LTP tokenizer not available, skipping LTP model export")
        else:
            # Create dummy input for LTP model
            dummy_text = "这是一个测试句子，用于导出LTP模型。" * 10  # Make it reasonably long
            token_ltp = ltp_processor.tokenizer(
                dummy_text, 
                return_tensors='pt', 
                padding=True,
                truncation=True,
                max_length=LTP_MAX_INPUT_LENGTH
            )
            
            logger.info(f"LTP input shape: {token_ltp['input_ids'].shape}")
            
            # Export LTP model to ONNX
            try:
                torch.onnx.export(
                    ltp_processor.model, 
                    (token_ltp["input_ids"], token_ltp["attention_mask"]),
                    "exported_model/ltp_model.onnx", 
                    opset_version=14, 
                    input_names=["input_ids", "attention_mask"],
                    output_names=["output"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "seq_len"},
                        "attention_mask": {0: "batch_size", 1: "seq_len"},
                        "output": {0: "batch_size", 1: "seq_len"}
                    }
                )
                logger.info("   ✓ Saved ONNX to exported_model/ltp_model.onnx")
            except Exception as e:
                logger.warning(f"Failed to export LTP model to ONNX: {e}")
            
            # Export LTP model to TorchScript
            try:
                ltp_processor.model.eval()
                traced_ltp_model = torch.jit.trace(
                    ltp_processor.model, 
                    (token_ltp["input_ids"], token_ltp["attention_mask"]), 
                    strict=False
                )
                traced_ltp_model.save("exported_model/ltp_model.pt")
                logger.info("   ✓ Saved TorchScript to exported_model/ltp_model.pt")
            except Exception as e:
                logger.warning(f"Failed to export LTP model to TorchScript: {e}")
            
            # Export LTP tokenizer
            try:
                ltp_processor.tokenizer.save_pretrained("exported_model/ltp_tokenizer")
                logger.info("   ✓ Saved LTP tokenizer to exported_model/ltp_tokenizer/")
            except Exception as e:
                logger.warning(f"Failed to save LTP tokenizer: {e}")
                
    except Exception as e:
        logger.error(f"Failed to initialize LTP processor for export: {e}")
        logger.info("Skipping LTP model export")
    
    logger.info("All models exported successfully!")
    logger.info("ONNX models for ONNX Runtime backend:")
    logger.info("  - exported_model/text_embedding.onnx")
    logger.info("  - exported_model/text_embedding_mean_pooling.onnx") 
    logger.info("  - exported_model/cosine_sim.onnx")
    if os.path.exists("exported_model/ltp_model.onnx"):
        logger.info("  - exported_model/ltp_model.onnx")
    logger.info("TorchScript models for LibTorch backend:")
    logger.info("  - exported_model/text_embedding.pt")
    logger.info("  - exported_model/text_embedding_mean_pooling.pt")
    logger.info("  - exported_model/cosine_sim.pt")
    if os.path.exists("exported_model/ltp_model.pt"):
        logger.info("  - exported_model/ltp_model.pt")
    logger.info("  - exported_model/model.pt")
    logger.info("Tokenizers:")
    logger.info("  - exported_model/tokenizer/")
    if os.path.exists("exported_model/ltp_tokenizer"):
        logger.info("  - exported_model/ltp_tokenizer/")