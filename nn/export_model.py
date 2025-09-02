import torch
import pnnx
import os
from nn.models import (
    TextEmbeddingModel, 
    TextEmbedder, 
    CosineSimilarityModel,
    LTPModel,
    LTPProcessor,
    MultiLabelClassifierBaseline,
    load_multi_label_classifier,
    TEXT_EMBEDDING_INPUT_LENGTH, 
    TEXT_EMBEDDING_OUTPUT_LENGTH,
    LTP_MAX_INPUT_LENGTH
)
from utils.logging_config import logger


def export_chat_text_classifier_baseline():
    """Export chat text classifier baseline model"""
    if not os.path.exists("chat_text_classifier_baseline.pt"):
        logger.warning("No chat_text_classifier_baseline.pt found, skipping")
        return
        
    logger.info("Exporting chat text classifier baseline model...")
    try:
        # Load the model using the proper loading function
        model, _ = load_multi_label_classifier("chat_text_classifier_baseline.pt")
        
        # Create dummy input for export
        dummy_input = torch.randn(1, TEXT_EMBEDDING_OUTPUT_LENGTH)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            "exported_model/chat_text_classifier_baseline.onnx",
            input_names=["embedding"],
            output_names=["output"],
            dynamic_axes={
                "embedding": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        logger.info("   ✓ Saved ONNX to exported_model/chat_text_classifier_baseline.onnx")
        
        # Export to TorchScript
        model.eval()
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save("exported_model/chat_text_classifier_baseline.pt")
        logger.info("   ✓ Saved TorchScript to exported_model/chat_text_classifier_baseline.pt")
        
    except Exception as e:
        logger.error(f"Failed to export chat text classifier baseline: {e}")


def export_text_embedding_model():
    """Export text embedding model without mean pooling"""
    logger.info("Exporting text embedding model (without mean pooling)...")
    embedder = TextEmbedder(mean_pooling=False, device="cpu")
    token = embedder.tokenizer("0" * TEXT_EMBEDDING_INPUT_LENGTH, return_tensors='pt', 
                            #    padding="max_length", 
                               padding=True,
                               truncation=True)
    logger.info(token)
    
    # Export to ONNX
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


def export_ltp_model():
    """Export LTP model"""
    logger.info("Exporting LTP model...")
    try:
        ltp_processor = LTPProcessor(device="cpu")
        
        if ltp_processor.tokenizer is None:
            logger.warning("LTP tokenizer not available, skipping LTP model export")
            return
            
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


if __name__ == "__main__":
    # Create exported_model directory if it doesn't exist
    os.makedirs("exported_model", exist_ok=True)
    
    logger.info("Exporting models to exported_model directory...")
    
    export_chat_text_classifier_baseline()
    export_text_embedding_model()
    export_ltp_model()
    