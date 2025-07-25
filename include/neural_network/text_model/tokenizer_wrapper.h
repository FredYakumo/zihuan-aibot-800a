#pragma once
#include "../nn.h"
#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace neural_network {
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
