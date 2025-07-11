#pragma once

#include "neural_network/nn.h"
#include "neural_network/text_model.h"
#include <memory>

namespace neural_network {
    struct ModelSet {
        ModelSet(neural_network::Device device = neural_network::Device::CPU);

        std::unique_ptr<neural_network::TextEmbeddingWithMeanPoolingModel> text_embedding_model;
        std::unique_ptr<neural_network::CosineSimilarityONNXModel> cosine_similarity_model;
        std::shared_ptr<tokenizers::Tokenizer> tokenizer;
        neural_network::TokenizerWrapper tokenizer_wrapper;
    };

    void init_model_set(Device device = Device::CPU);
    ModelSet &get_model_set();
} // namespace neural_network
