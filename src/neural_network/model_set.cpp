#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include "neural_network/text_model.h"
#include <exception>
#include <memory>
#include <stdexcept>

namespace neural_network {
    ModelSet::ModelSet(neural_network::Device device)
        : text_embedding_model("models/text_embedding_mean_pooling.onnx", device),
          cosine_similarity_model("models/cosine_sim.onnx", device),
          tokenizer(neural_network::load_tokenizers("tokenizer/tokenizer.json")),
          tokenizer_wrapper(tokenizer, neural_network::TokenizerConfig()) {}

    std::unique_ptr<ModelSet> model_set = nullptr;

    void init_model_set(Device device) { model_set = std::make_unique<ModelSet>(device); }

    ModelSet &get_model_set() {
        if (!model_set) {
            throw std::runtime_error("ModelSet has not been initialized. Call init_model_set() first.");
        }
        return *model_set;
    }
} // namespace neural_network