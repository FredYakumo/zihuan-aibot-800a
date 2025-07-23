#pragma once
#include "nn.h"
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
         * @param model_path P`ath to onnx model
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

        // /**
        //  * @brief Get the sentence embedding for a given text using mean pooling.
        //  * @return A vector representing the sentence embedding.
        //  */
        // emb_vec_t embed_with_mean_pooling(const token_id_list_t &token_ids,
        //                                   const attention_mask_list_t &attention_mask) {
        //     emb_mat_t token_embeddings = embed(token_ids, attention_mask);
        //     return neural_network::mean_pooling(token_embeddings, attention_mask);
        // }

      private:
        Ort::MemoryInfo m_memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Session m_session;
        std::vector<const char *> m_input_names;
        std::vector<Ort::AllocatedStringPtr> m_input_names_ptr;
        std::vector<const char *> m_output_names;
        std::vector<Ort::AllocatedStringPtr> m_output_names_ptr;
    };

    class TextEmbeddingWithMeanPoolingModel {
      public:
        /**
         * @brief Construct a new Text Embedding Model object
         *
         * @param model_path Path to onnx model
         * @param device Device to run model
         */
        TextEmbeddingWithMeanPoolingModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Get the sentence embedding for a given text.
         * @param text Input text
         * @return A vector representing the sentence embedding.
         */
        emb_vec_t embed(const std::string &text);

        /**
         * @brief Get the sentence embeddings for multiple texts.
         * @param texts List of input texts
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<std::string> &texts, size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

        /**
         * @brief Get the sentence embedding for a given tokenized text.
         * @return A vector representing the sentence embedding.
         */
        emb_vec_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the sentence embeddings for multiple tokenized texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<token_id_list_t> &token_ids,
                        const std::vector<attention_mask_list_t> &attention_mask,
                        size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

        // /**
        //  * @brief Get the sentence embedding for a given text using mean pooling.
        //  * @return A vector representing the sentence embedding.
        //  */
        // emb_vec_t embed_with_mean_pooling(const token_id_list_t &token_ids,
        //                                   const attention_mask_list_t &attention_mask) {
        //     emb_mat_t token_embeddings = embed(token_ids, attention_mask);
        //     return neural_network::mean_pooling(token_embeddings, attention_mask);
        // }

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

    class TextEmbeddingWithMeanPoolingModel {
      public:
        /**
         * @brief Construct a new Text Embedding Model object
         *
         * @param model_path Path to torchscript model
         * @param device Device to run model
         */
        TextEmbeddingWithMeanPoolingModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Get the sentence embedding for a given text.
         * @param text Input text
         * @return A vector representing the sentence embedding.
         */
        emb_vec_t embed(const std::string &text);

        /**
         * @brief Get the sentence embeddings for multiple texts.
         * @param texts List of input texts
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<std::string> &texts, size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

        /**
         * @brief Get the sentence embedding for a given tokenized text.
         * @return A vector representing the sentence embedding.
         */
        emb_vec_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the sentence embeddings for multiple tokenized texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @param max_batch_size Maximum batch size for processing (default: DEFAULT_MAX_BATCH_SIZE)
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<token_id_list_t> &token_ids,
                        const std::vector<attention_mask_list_t> &attention_mask,
                        size_t max_batch_size = DEFAULT_MAX_BATCH_SIZE);

      private:
        torch::jit::script::Module m_module;
        torch::Device m_device;
    };

#endif

    struct TokenizerConfig {
        bool add_special_tokens = true;
        // [CLS]
        int32_t cls_token_id = 101;
        // [SEP]
        int32_t sep_token_id = 102;
    };

    using token_id_data_t = int32_t;
    using token_id_vec_t = std::vector<token_id_data_t>;
    using token_id_vec_with_mask_t = std::pair<token_id_vec_t, token_id_vec_t>;

    inline std::shared_ptr<tokenizers::Tokenizer> load_tokenizers(const std::string &path) {
        spdlog::info("Load tokenizers from {}", path);
        auto start = std::chrono::high_resolution_clock::now();

        auto blob = load_bytes_from_file(path);
        auto tokenizers = tokenizers::Tokenizer::FromBlobJSON(blob);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        spdlog::info("Load time: {} ms", duration);

        return std::move(tokenizers);
    }

    class TokenizerWrapper {
      public:
        TokenizerWrapper(std::shared_ptr<tokenizers::Tokenizer> tokenizer, TokenizerConfig config)
            : m_tokenizer(tokenizer), m_config(std::move(config)) {}
        inline token_id_vec_with_mask_t encode(const std::string &text, std::optional<std::pair<size_t, token_id_data_t>> padding = std::nullopt) const {
            token_id_vec_t tokens = m_tokenizer->Encode(text);

            if (m_config.add_special_tokens) {
                tokens.insert(tokens.begin(), m_config.cls_token_id);
                tokens.push_back(m_config.sep_token_id);
            }

            // Create attention mask (1 for real tokens, 0 for padding)
            token_id_vec_t attention_mask(tokens.size(), 1);
            
            // Apply padding if enabled in config and target length is specified
            if (padding && tokens.size() < padding->first) {
                size_t original_size = tokens.size();
                tokens.resize(padding->first, padding->second);
                attention_mask.resize(padding->first, 0); // Padding positions get 0 in attention mask
            }

            return {tokens, attention_mask};
        }

      private:
        std::shared_ptr<tokenizers::Tokenizer> m_tokenizer;
        TokenizerConfig m_config;
    };
} // namespace neural_network