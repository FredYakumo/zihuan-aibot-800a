#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include "neural_network/text_model.h"
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>

constexpr const char *TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH = "models/text_embedding_mean_pooling.onnx";
constexpr const char *COSINE_SIMILARITY_MODEL_PATH = "models/cosine_sim.onnx";

namespace neural_network {
    ModelSet::ModelSet(neural_network::Device device)
        : tokenizer(neural_network::load_tokenizers("tokenizer/tokenizer.json")),
          tokenizer_wrapper(tokenizer, neural_network::TokenizerConfig()) {

        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        this->text_embedding_model = std::make_unique<neural_network::TextEmbeddingWithMeanPoolingModel>(
            TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH, device);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Loading text embedding model from {} successfully in {} ms",
                     TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH, duration.count());

        start_time = std::chrono::high_resolution_clock::now();
        this->cosine_similarity_model =
            std::make_unique<neural_network::CosineSimilarityONNXModel>(COSINE_SIMILARITY_MODEL_PATH, device);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Loading cosine similarity model from {} successfully in {} ms", COSINE_SIMILARITY_MODEL_PATH,
                     duration.count());
    }

    std::unique_ptr<ModelSet> model_set = nullptr;

    void init_model_set(Device device) { model_set = std::make_unique<ModelSet>(device); }

    ModelSet &get_model_set() {
        if (!model_set) {
            throw std::runtime_error("ModelSet has not been initialized. Call init_model_set() first.");
        }
        return *model_set;
    }
} // namespace neural_network