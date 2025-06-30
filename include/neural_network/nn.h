#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <fstream>
#include <numeric>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <tokenizers_cpp.h>
#include <utility>
#include <vector>

namespace neural_network {

    Ort::Env &get_onnx_runtime();

    void init_onnx_runtime();

    Ort::SessionOptions get_onnx_session_opts();

    Ort::SessionOptions get_onnx_session_opts_tensorrt();

    Ort::SessionOptions get_onnx_session_opts_core_ml();

    /**
     * @brief Device type for model inference
     *
     */
    enum class Device { CPU, CUDA, TensorRT, CoreML };

    inline Ort::SessionOptions get_session_options(Device device) {
        switch (device) {
        case Device::CUDA:
        case Device::TensorRT:
            // Assuming TensorRT uses CUDA options as a base
            return get_onnx_session_opts_tensorrt();
        case Device::CoreML:
            return get_onnx_session_opts_core_ml();
        case Device::CPU:
        default:
            return get_onnx_session_opts();
        }
    }

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

    constexpr size_t COSINE_SIMILARITY_INPUT_EMB_SIZE = 1024;
    class CosineSimilarityONNXModel {
      public:
        CosineSimilarityONNXModel(const std::string &model_path,
                                  const Ort::SessionOptions &options = get_onnx_session_opts())
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
                m_output_names.push_back(name);
                m_output_names_ptr.emplace_back(std::move(name_ptr));
            }
        }

        std::vector<float> inference(std::vector<float> target, std::vector<std::vector<float>> value_list) {
            // 展平二维value_list到一维连续数组
            std::vector<float> flattened_values;
            for (const auto &vec : value_list) {
                // 确保每个子vector都是COSINE_SIMILARITY_INPUT_EMB_SIZE维
                if (vec.size() != COSINE_SIMILARITY_INPUT_EMB_SIZE) {
                    throw std::invalid_argument(
                        fmt::format("All vectors in value_list must be size {}", COSINE_SIMILARITY_INPUT_EMB_SIZE));
                }
                flattened_values.insert(flattened_values.end(), vec.begin(), vec.end());
            }

            // 创建target张量 (shape [1, COSINE_SIMILARITY_INPUT_EMB_SIZE])
            std::vector<int64_t> target_shape{1, COSINE_SIMILARITY_INPUT_EMB_SIZE};
            auto target_tensor = Ort::Value::CreateTensor<float>(m_memory_info, target.data(), target.size(),
                                                                 target_shape.data(), target_shape.size());

            // 创建value张量 (shape [num_samples, COSINE_SIMILARITY_INPUT_EMB_SIZE])
            std::vector<int64_t> value_shape{static_cast<int64_t>(value_list.size()), COSINE_SIMILARITY_INPUT_EMB_SIZE};
            auto value_tensor =
                Ort::Value::CreateTensor<float>(m_memory_info, flattened_values.data(), flattened_values.size(),
                                                value_shape.data(), value_shape.size());

            // 准备输入张量列表
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(target_tensor));
            input_tensors.push_back(std::move(value_tensor));

            auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                                input_tensors.size(), m_output_names.data(), m_output_names.size());

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

    inline float dot_product(const float *a, const float *b, size_t size) {
        if (size == 0)
            throw std::invalid_argument("dot_product: size must > 0");

        return std::inner_product(a, a + size, b, 0.0f);
    }

    inline std::vector<float> mean_pooling(const float *token_embeddings, const int32_t *attention_mask, size_t seq_len,
                                           size_t hidden_size) {
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

    inline std::vector<float> mean_pooling(const std::vector<std::vector<float>> &token_embeddings,
                                           const std::vector<int32_t> &attention_mask) {
        if (token_embeddings.empty()) {
            return {};
        }
        if (token_embeddings.size() != attention_mask.size()) {
            throw std::invalid_argument("token_embeddings and attention_mask must have the same size.");
        }
        size_t hidden_size = token_embeddings[0].size();
        std::vector<float> pooled(hidden_size, 0.0f);
        float sum_mask = 0.0f;

        for (size_t i = 0; i < token_embeddings.size(); ++i) {
            if (attention_mask[i] == 0)
                continue;

            sum_mask += 1.0f;
            for (size_t j = 0; j < hidden_size; ++j) {
                pooled[j] += token_embeddings[i][j];
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

    inline float vector_norm(const float *vec, size_t size) {
        if (size == 0)
            throw std::invalid_argument("vector_norm: size can't be 0");

        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum);
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