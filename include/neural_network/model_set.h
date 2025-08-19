#pragma once

#include "neural_network/nn.h"
#include "neural_network/text_model/text_embedding_model.h"
#include "neural_network/text_model/tokenizer_wrapper.h"
#include <memory>

#ifdef __USE_PADDLE_INFERENCE__
#include <neural_network/text_model/lac/lac.h>
#endif

namespace neural_network {
    struct ModelSet {
        ModelSet(neural_network::Device device = neural_network::Device::CPU);

        std::unique_ptr<neural_network::TextEmbeddingModel> text_embedding_model;

        std::shared_ptr<tokenizers::Tokenizer> tokenizer;
        neural_network::TokenizerWrapper tokenizer_wrapper;

#ifdef __USE_PADDLE_INFERENCE__
        std::unique_ptr<neural_network::lac::LAC> lac_model;
#endif
    };

    void init_model_set(Device device = Device::CPU);
    ModelSet &get_model_set();
} // namespace neural_network
