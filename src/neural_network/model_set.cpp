#include "neural_network/model_set.h"
#include "config.h"
#include "neural_network/nn.h"
#include "neural_network/text_model/text_embedding_model.h"
#include "neural_network/text_model/tokenizer_wrapper.h"
#include <chrono>
#include <memory>
#include <spdlog/spdlog.h>
#include <stdexcept>

#ifdef __USE_PADDLE_INFERENCE__
#include "neural_network/text_model/lac/lac.h"
#endif

#ifdef __USE_ONNX_RUNTIME__
constexpr const char *TEXT_EMBEDDING_PATH = "exported_model/text_embedding.onnx";
#elif defined(__USE_LIBTORCH__)
constexpr const char *TEXT_EMBEDDING_MODEL_PATH = "exported_model/text_embedding.pt";
#endif

namespace neural_network {
    ModelSet::ModelSet(neural_network::Device device)
        : tokenizer(neural_network::load_tokenizers("exported_model/tokenizer/tokenizer.json")),
          tokenizer_wrapper(tokenizer, neural_network::TokenizerConfig()) {

        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        this->text_embedding_model =
            std::make_unique<neural_network::TextEmbeddingModel>(TEXT_EMBEDDING_MODEL_PATH, device);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Loading text embedding model from {} successfully in {} ms", TEXT_EMBEDDING_MODEL_PATH,
                     duration.count());

#ifdef __USE_PADDLE_INFERENCE__
        // Load LAC model using path from config
        const std::string &lac_model_path = Config::instance().lac_model_path;
        start_time = std::chrono::high_resolution_clock::now();
        this->lac_model = std::make_unique<neural_network::lac::LAC>(lac_model_path, neural_network::lac::CODE_UTF8);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        spdlog::info("Loading LAC model from {} successfully in {} ms", lac_model_path, duration.count());
#endif
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