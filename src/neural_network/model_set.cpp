#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include "neural_network/text_model/text_embedding_model.h"
#include "neural_network/text_model/text_embedding_with_mean_pooling_model.h"
#include "neural_network/text_model/tokenizer_wrapper.h"
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>

#ifdef __USE_ONNX_RUNTIME__
constexpr const char *TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH = "exported_model/text_embedding_mean_pooling.onnx";
constexpr const char *COSINE_SIMILARITY_MODEL_PATH = "exported_model/cosine_sim.onnx";
constexpr const char *LTP_MODEL_PATH = "exported_model/ltp_model.onnx";
#elif defined(__USE_LIBTORCH__)
constexpr const char *TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH = "exported_model/text_embedding_mean_pooling.pt";
constexpr const char *COSINE_SIMILARITY_MODEL_PATH = "exported_model/cosine_sim.pt";
constexpr const char *LTP_MODEL_PATH = "exported_model/ltp_model.pt";
#endif

namespace neural_network {
    ModelSet::ModelSet(neural_network::Device device)
        : tokenizer(neural_network::load_tokenizers("exported_model/tokenizer/tokenizer.json")),
          tokenizer_wrapper(tokenizer, neural_network::TokenizerConfig()) {

        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        this->text_embedding_model = std::make_unique<neural_network::TextEmbeddingWithMeanPoolingModel>(
            TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH, device);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Loading text embedding model from {} successfully in {} ms",
                     TEXT_EMBEDDING_MEAN_POOLING_MODEL_PATH, duration.count());

        start_time = std::chrono::high_resolution_clock::now();
#ifdef __USE_ONNX_RUNTIME__
        this->cosine_similarity_model =
            std::make_unique<neural_network::CosineSimilarityModel>(COSINE_SIMILARITY_MODEL_PATH, device);
#elif defined(__USE_LIBTORCH__)
        this->cosine_similarity_model =
            std::make_unique<neural_network::CosineSimilarityModel>(COSINE_SIMILARITY_MODEL_PATH, device);
#endif
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