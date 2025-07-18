#pragma once

#include "neural_network/nn.h"
#include "neural_network/text_model.h"
#include <memory>

namespace neural_network {
    struct ModelSet {
        ModelSet(neural_network::Device device = neural_network::Device::CPU);

        std::unique_ptr<neural_network::TextEmbeddingWithMeanPoolingModel> text_embedding_model;
#ifdef __USE_ONNX_RUNTIME__
        std::unique_ptr<neural_network::CosineSimilarityModel> cosine_similarity_model;
#endif
#ifdef __USE_LIBTORCH__
        std::unique_ptr<neural_network::CosineSimilarityModel> cosine_similarity_model;
#endif
        std::shared_ptr<tokenizers::Tokenizer> tokenizer;
        neural_network::TokenizerWrapper tokenizer_wrapper;
    };

    void init_model_set(Device device = Device::CPU);
    ModelSet &get_model_set();
} // namespace neural_network
