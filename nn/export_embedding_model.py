"""
Export text embedding model separately.

This script extracts and exports the text embedding model as a standalone model.
This allows the text embedding functionality to be used separately from the classifier,
reducing memory usage when only embeddings are needed.
"""

import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nn.models import (
    TEXT_EMBEDDING_DEFAULT_MODEL_NAME,
    TextEmbedder,
    TextEmbeddingModel,
    get_device,
)


def main():
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load text embedding model
    print(f"Loading text embedding model: {TEXT_EMBEDDING_DEFAULT_MODEL_NAME}")
    text_embedder = TextEmbedder(device=device)
    text_embedding_model = text_embedder.model
    tokenizer = text_embedder.tokenizer

    # Set to evaluation mode
    text_embedding_model.eval()

    # Create dummy inputs for embedding model
    MAX_LEN = 128
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN)).to(device)
    dummy_attention_mask = torch.ones(1, MAX_LEN, dtype=torch.long).to(device)

    # Test the model with dummy inputs
    with torch.no_grad():
        outputs = text_embedding_model(dummy_input_ids, dummy_attention_mask)
        print(f"Test output shape: {outputs.shape}")

    # Export to ONNX
    try:
        embedding_onnx_path = "exported_model/text_embedding.onnx"
        print(f"Exporting to ONNX: {embedding_onnx_path}")

        # Move to CPU for export
        text_embedding_model.to("cpu")
        dummy_input_ids = dummy_input_ids.to("cpu")
        dummy_attention_mask = dummy_attention_mask.to("cpu")

        torch.onnx.export(
            text_embedding_model,
            (dummy_input_ids, dummy_attention_mask),
            embedding_onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=11,
            do_constant_folding=True,
        )

        # Save PyTorch model
        torch.save(
            text_embedding_model.state_dict(), "exported_model/text_embedding.pt"
        )

        print(
            f"Text embedding model successfully exported to {embedding_onnx_path} and exported_model/text_embedding.pt"
        )
    except Exception as e:
        print(f"Text embedding model export failed: {e}")


if __name__ == "__main__":
    main()
