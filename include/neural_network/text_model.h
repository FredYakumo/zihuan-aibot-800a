#pragma once
#include "nn.h"
#include <string>
#include <vector>

namespace neural_network {


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
         * @return A list of matrices representing the embeddings for each text.
         */
        std::vector<emb_mat_t> embed(const std::vector<std::string> &texts);
        /**
         * @brief Get the token embeddings for a given text.
         * @return A matrix representing the embeddings for each token.
         */
        emb_mat_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the token embeddings for multiple texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @return A list of matrices representing the embeddings for each text
         */
        std::vector<emb_mat_t> embed(const std::vector<token_id_list_t> &token_ids,
                                     const std::vector<attention_mask_list_t> &attention_mask);

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
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<std::string> &texts);

        /**
         * @brief Get the sentence embedding for a given tokenized text.
         * @return A vector representing the sentence embedding.
         */
        emb_vec_t embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask);

        /**
         * @brief Get the sentence embeddings for multiple tokenized texts.
         * @param token_ids List of token ID sequences
         * @param attention_mask List of attention mask sequences
         * @return A matrix where each row is a sentence embedding.
         */
        emb_mat_t embed(const std::vector<token_id_list_t> &token_ids,
                        const std::vector<attention_mask_list_t> &attention_mask);

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
        inline token_id_vec_t encode(const std::string &text) const {
            token_id_vec_t tokens = m_tokenizer->Encode(text);

            if (m_config.add_special_tokens) {
                tokens.insert(tokens.begin(), m_config.cls_token_id);
                tokens.push_back(m_config.sep_token_id);
            }
            return tokens;
        }

        inline std::vector<token_id_vec_t> encode_batch(const std::vector<std::string> &texts) const {
            return m_tokenizer->EncodeBatch(texts);
        }

        inline token_id_vec_with_mask_t encode_with_mask(const std::string &text) const {
            auto tokens = encode(text);
            token_id_vec_t mask(tokens.size(), 1);

            return {tokens, mask};
        }

        inline std::vector<token_id_vec_with_mask_t> batch_encode(const std::vector<std::string> &batch_text,
                                                                  std::optional<token_id_data_t> padding = 0) const {
            size_t max_vec_dim = 0;
            std::vector<token_id_vec_with_mask_t> res;
            for (const auto &text : batch_text) {
                token_id_vec_with_mask_t i = encode_with_mask(text);
                if (i.first.size() > max_vec_dim) {
                    max_vec_dim = i.first.size();
                }
                res.push_back(std::move(i));
            }
            if (padding) {
                for (auto &emb_with_mask : res) {
                    if (emb_with_mask.first.size() < max_vec_dim) {
                        emb_with_mask.first.resize(max_vec_dim, *padding);
                    }
                    if (emb_with_mask.second.size() < max_vec_dim) {
                        emb_with_mask.second.resize(max_vec_dim, *padding);
                    }
                }
            }
            return res;
        }

      private:
        std::shared_ptr<tokenizers::Tokenizer> m_tokenizer;
        TokenizerConfig m_config;
    };
} // namespace neural_network