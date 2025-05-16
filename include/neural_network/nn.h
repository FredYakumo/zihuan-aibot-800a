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

    Ort::Env &get_onnx_runtime();

    void init_onnx_runtime();

    Ort::SessionOptions get_onnx_session_opts();

    Ort::SessionOptions get_onnx_session_opts_core_ml();

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
            : m_session(get_onnx_runtime(), model_path.c_str(), Ort::SessionOptions{}) {

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

            std::vector<Ort::Value> input_tensors;
            std::vector<int64_t> input_ids(token_ids.cbegin(), token_ids.cend());
            std::vector<int64_t> masks(attention_mask.cbegin(), attention_mask.cend());

            // Create input_ids, attention_mask tensor
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                m_memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));

            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(m_memory_info, masks.data(), masks.size(),
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

        Ort::MemoryInfo m_memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Session m_session;
        std::vector<const char *> m_input_names;
        std::vector<Ort::AllocatedStringPtr> m_input_names_ptr;
        std::vector<const char *> m_output_names;
        std::vector<Ort::AllocatedStringPtr> m_output_names_ptr;
    };

    constexpr size_t COSINE_SIMILARITY_INPUT_EMB_SIZE = 1024;
    class CosineSimilarityONNXModel {
      public:
        CosineSimilarityONNXModel(const std::string &model_path, const Ort::SessionOptions &options = get_onnx_session_opts())
            : m_session(get_onnx_runtime(), model_path.c_str(), options), m_allocator() {
            Ort::AllocatorWithDefaultOptions allocator;

            for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
                Ort::AllocatedStringPtr name_ptr = m_session.GetInputNameAllocated(i, allocator);
                const char *name = name_ptr.get();
                spdlog::debug("CosineSimilarityModel: input name: {}", name);
                m_input_names.push_back(name);
                m_input_names_ptr.emplace_back(std::move(name_ptr));
            }
            m_output_names.reserve(1);
            m_output_names_ptr.reserve(1);
            for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
                Ort::AllocatedStringPtr name_ptr = m_session.GetOutputNameAllocated(i, allocator);
                const char *name = name_ptr.get();
                spdlog::debug("CosineSimilarityModel: output name: {}", name);
                m_output_names_ptr.emplace_back(std::move(name_ptr));
            }
        }

        std::vector<float> inference(std::vector<float> target, std::vector<std::vector<float>> value_list) {
            // 展平二维value_list到一维连续数组
            std::vector<float> flattened_values;
            for (const auto &vec : value_list) {
                // 确保每个子vector都是COSINE_SIMILARITY_INPUT_EMB_SIZE维
                if (vec.size() != COSINE_SIMILARITY_INPUT_EMB_SIZE) {
                    throw std::invalid_argument(fmt::format("All vectors in value_list must be size {}", COSINE_SIMILARITY_INPUT_EMB_SIZE));
                }
                flattened_values.insert(flattened_values.end(), vec.begin(), vec.end());
            }

            // 创建target张量 (shape [1, COSINE_SIMILARITY_INPUT_EMB_SIZE])
            std::vector<int64_t> target_shape{1, COSINE_SIMILARITY_INPUT_EMB_SIZE};
            auto target_tensor = Ort::Value::CreateTensor<float>(m_memory_info, target.data(), target.size(),
                                                                 target_shape.data(), target_shape.size());

            // 创建value张量 (shape [num_samples, COSINE_SIMILARITY_INPUT_EMB_SIZE])
            std::vector<int64_t> value_shape {static_cast<int64_t>(value_list.size()), COSINE_SIMILARITY_INPUT_EMB_SIZE};
            auto value_tensor =
                Ort::Value::CreateTensor<float>(m_memory_info, flattened_values.data(), flattened_values.size(),
                                                value_shape.data(), value_shape.size());

            // 准备输入张量列表
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(target_tensor));
            input_tensors.push_back(std::move(value_tensor));

            const char *input_names[] = {"target", "value_list"};
            const char *output_names[] = {"output"};

            auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(),
                                                 input_tensors.size(), output_names, 1);

            // 检查输出有效性
            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                throw std::runtime_error("Inference failed: invalid output tensors");
            }

            // 获取输出数据
            float *output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            // 计算输出元素总数
            size_t output_size = 1;
            for (auto dim : output_shape) {
                if (dim < 0) {
                    throw std::runtime_error("Dynamic dimensions in output are not supported");
                }
                output_size *= dim;
            }

            // 拷贝结果到vector
            return std::vector<float>(output_data, output_data + output_size);
        }

      private:
        Ort::Session m_session;
        Ort::MemoryInfo m_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions m_allocator;
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