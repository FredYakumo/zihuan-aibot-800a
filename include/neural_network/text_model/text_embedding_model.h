#pragma once
#include "../nn.h"
#include <string>
#include <vector>

#ifdef __USE_LIBTORCH__
#include <torch/script.h>
#include <c10/core/Device.h>
#endif

namespace neural_network {
    constexpr size_t EMBEDDING_MAX_INPUT_LENGTH = 8192;
    constexpr size_t EMBEDDING_OUTPUT_LENGTH = 1024;

#ifdef __USE_ONNX_RUNTIME__
    class TextEmbeddingModel {
      public:
        /**
         * @brief Construct a new Text Embedding Model object
         *
         * @param model_path Path to onnx model
         * @param device Device to run model
         */
        TextEmbeddingModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Get the token embeddings for a given text.
         * @param text Input text
         * @return A matrix representing the embeddings for each token.
         */
        emb_mat_t embed(const std::string &text);

        /**
         * @brief Get the token embeddings for multiple texts.
         * @param texts List of input texts
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A list of matrices representing the embeddings for each text.
         */
        std::vector<emb_mat_t> embed(const std::vector<std::string> &texts,
                                     size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);
        /**
         * @brief Get the token embeddings for a given text.
         * @return A matrix representing the embeddings for each token.
         */
        emb_mat_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the token embeddings for multiple texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A list of matrices representing the embeddings for each text
         */
        std::vector<emb_mat_t> embed(const std::vector<token_id_list_t> &token_ids,
                                     const std::vector<attention_mask_list_t> &attention_mask,
                                     size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

      private:
        Ort::MemoryInfo m_memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Session m_session;
        std::vector<const char *> m_input_names;
        std::vector<Ort::AllocatedStringPtr> m_input_names_ptr;
        std::vector<const char *> m_output_names;
        std::vector<Ort::AllocatedStringPtr> m_output_names_ptr;
    };
#endif

#ifdef __USE_LIBTORCH__
    class TextEmbeddingModel {
      public:
        /**
         * @brief Construct a new Text Embedding Model object
         *
         * @param model_path Path to torchscript model
         * @param device Device to run model
         */
        TextEmbeddingModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Get the token embeddings for a given text.
         * @param text Input text
         * @return A matrix representing the embeddings for each token.
         */
        emb_mat_t embed(const std::string &text);

        /**
         * @brief Get the token embeddings for multiple texts.
         * @param texts List of input texts
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A list of matrices representing the embeddings for each text.
         */
        std::vector<emb_mat_t> embed(const std::vector<std::string> &texts,
                                     size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

        /**
         * @brief Get the token embeddings for a given tokenized text.
         * @return A matrix representing the embeddings for each token.
         */
        emb_mat_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the token embeddings for multiple tokenized texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A list of matrices representing the embeddings for each text
         */
        std::vector<emb_mat_t> embed(const std::vector<token_id_list_t> &token_ids,
                                     const std::vector<attention_mask_list_t> &attention_mask,
                                     size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

      private:
        torch::jit::script::Module m_module;
        torch::Device m_device;
    };
#endif
} // namespace neural_network
