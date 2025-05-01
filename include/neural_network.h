#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <fstream>
#include <memory>
#include <numeric>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <tokenizers_cpp.h>
#include <utility>
#include <vector>

namespace neural_network {
    inline std::string load_bytes_from_file(const std::string &path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            spdlog::error("Load bytes from file {} failed.", path);
            return "";
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

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

        inline token_id_vec_with_mask_t encode_with_mask(const std::string &text) const {
            auto tokens = encode(text);
            token_id_vec_t mask(tokens.size(), 1);

            return {tokens, mask};
        }

        inline std::vector<token_id_vec_with_mask_t> batch_encode(const std::vector<std::string> &batch_text,
                                                             std::optional<token_id_data_t> padding = 0) {
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

    using emb_vec_t = std::vector<float>;

    class ONNXEmbedder {
      public:
        ONNXEmbedder(const std::string &model_path)
            : m_env(ORT_LOGGING_LEVEL_WARNING, "Embedding"),
              m_session(m_env, model_path.c_str(), Ort::SessionOptions{}) {

            // Get input/output info

            Ort::AllocatorWithDefaultOptions allocator;
            m_input_names.reserve(2);
            for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
                Ort::AllocatedStringPtr name_ptr = m_session.GetInputNameAllocated(i, allocator);
                m_input_names.push_back(name_ptr.get());
                m_input_names_ptr.emplace_back(std::move(name_ptr));
            }

            m_output_names.reserve(1);
            for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
                Ort::AllocatedStringPtr name_ptr = m_session.GetOutputNameAllocated(i, allocator);
                m_output_names.push_back(name_ptr.get());
                m_output_names_ptr.emplace_back(std::move(name_ptr));
            }
        }

        // Exec embedding inference
        emb_vec_t embed(const std::vector<int32_t> &token_ids, const std::vector<int32_t> &attention_mask) {
            if (token_ids.size() != attention_mask.size()) {
                throw std::invalid_argument("token_ids, attention_mask length not match");
            }

            const size_t seq_len = token_ids.size();
            const std::vector<int64_t> input_shape = {1, static_cast<token_id_data_t>(seq_len)};

            // Prepare input tensor
            Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            std::vector<Ort::Value> input_tensors;
            std::vector<int64_t> input_ids(token_ids.cbegin(), token_ids.cend());
            std::vector<int64_t> masks(attention_mask.cbegin(), attention_mask.cend());

            // Create input_ids, attention_mask tensor
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));

            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, masks.data(), masks.size(),
                                                                         input_shape.data(), input_shape.size()));

            // inference
            auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                                input_tensors.size(), m_output_names.data(), 1);

            // get token embeddings [batch_size, seq_len, hidden_size]
            float *token_embeddings = output_tensors[0].GetTensorMutableData<float>();
            auto embeddings_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            const int64_t hidden_size = embeddings_shape[2];

            return mean_pooling(token_embeddings, masks, seq_len, hidden_size);
        }

      private:
        inline std::vector<float> mean_pooling(float *token_embeddings, const std::vector<int64_t> &attention_mask,
                                               size_t seq_len, size_t hidden_size) {
            std::vector<float> pooled(hidden_size, 0.0f);
            float sum_mask = 0.0f;

            for (size_t i = 0; i < seq_len; ++i) {
                if (attention_mask[i] == 0)
                    continue;

                const float *emb = token_embeddings + i * hidden_size;
                sum_mask += 1.0f;

                for (size_t j = 0; j < hidden_size; ++j) {
                    pooled[j] += emb[j];
                }
            }

            const float epsilon = 1e-9f;
            if (sum_mask > epsilon) {
                for (auto &val : pooled) {
                    val /= sum_mask;
                }
            }

            return pooled;
        }
        Ort::Env m_env;
        Ort::Session m_session;
        std::vector<const char *> m_input_names;
        std::vector<Ort::AllocatedStringPtr> m_input_names_ptr;
        std::vector<const char *> m_output_names;
        std::vector<Ort::AllocatedStringPtr> m_output_names_ptr;
    };

    inline float vector_norm(const float *vec, size_t size) {
        if (size == 0)
            throw std::invalid_argument("vector_norm: size can't be 0");

        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum);
    }

    inline float dot_product(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("dot_product: size must > 0");

        return std::inner_product(a, a + size, b, 0.0f);
    }

    inline float cosine_similarity(const float *emb1, const float *emb2, size_t dim) {
        if (emb1 == nullptr || emb2 == nullptr) {
            throw std::invalid_argument("consine_similarity: emb can't be nullptr");
        }
        if (dim == 0) {
            throw std::invalid_argument("consine_similarity: vec dim must > 0");
        }

        const float dot = dot_product(emb1, emb2, dim);

        const float norm1 = vector_norm(emb1, dim);
        const float norm2 = vector_norm(emb2, dim);

        const float epsilon = 1e-8f;
        const float denominator = norm1 * norm2;
        if (denominator < epsilon) {
            // 处理零向量情况：定义零向量与任何向量相似度为0
            return 0.0f;
        }

        // normalize
        return std::max(-1.0f, std::min(1.0f, dot / denominator));
    }

    inline float cosine_similarity_with_padding(const std::vector<float> &emb1, const std::vector<float> &emb2) {
        const size_t dim1 = emb1.size();
        const size_t dim2 = emb2.size();
        const size_t max_dim = std::max(dim1, dim2);

        std::vector<float> padded_emb1(max_dim, 0.0f);
        std::vector<float> padded_emb2(max_dim, 0.0f);

        if (dim1 > 0) {
            std::copy(emb1.begin(), emb1.end(), padded_emb1.begin());
        }
        if (dim2 > 0) {
            std::copy(emb2.begin(), emb2.end(), padded_emb2.begin());
        }

        return cosine_similarity(padded_emb1.data(), padded_emb2.data(), max_dim);
    }

} // namespace neural_network
#endif