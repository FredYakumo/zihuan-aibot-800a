#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <fstream>
#include <numeric>
#ifdef __USE_ONNX_RUNTIME__
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif
#include <spdlog/spdlog.h>
#include <tokenizers_cpp.h>
#include <utility>
#include <vector>
#ifdef __USE_LIBTORCH__
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#endif

namespace neural_network {
    using token_id_list_t = std::vector<int32_t>;
    using attention_mask_list_t = std::vector<int32_t>;
    using emb_vec_t = std::vector<float>;
    using emb_mat_t = std::vector<emb_vec_t>;

    constexpr size_t DEFAULT_MAX_BATCH_SIZE = 512;

#ifdef __USE_ONNX_RUNTIME__
    Ort::Env &get_onnx_runtime();

    void init_onnx_runtime();

    Ort::SessionOptions get_onnx_session_opts_cpu();

    Ort::SessionOptions get_onnx_session_opts_cuda();

    Ort::SessionOptions get_onnx_session_opts_tensorrt();

    Ort::SessionOptions get_onnx_session_opts_core_ml();
#endif

    /**
     * @brief Device type for model inference
     *
     */
    enum class Device { CPU, CUDA, TensorRT, CoreML, MPS };

    constexpr Device USE_DEVICE = Device::CPU;

#ifndef __USE_LIBTORCH__
    inline Ort::SessionOptions get_session_options(Device device) {
        switch (device) {
        case Device::CUDA:
        case Device::TensorRT:
            return get_onnx_session_opts_tensorrt();
        case Device::CoreML:
            return get_onnx_session_opts_core_ml();
        case Device::CPU:
        default:
            return get_onnx_session_opts_cpu();
        }
    }
#endif

#ifdef __USE_LIBTORCH__
    inline torch::Device get_torch_device(Device device) {
        switch (device) {
        case Device::CUDA:
            return torch::kCUDA;
        case Device::TensorRT:
            return torch::kCUDA; // TensorRT runs on CUDA
        case Device::CoreML:
            return torch::kMPS; // CoreML is not supported by PyTorch
        case Device::MPS:
            return torch::kMPS;
        case Device::CPU:
        default:
            return torch::kCPU;
        }
    }
#endif

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

#ifdef __USE_ONNX_RUNTIME__

#endif // __USE_ONNX_RUNTIME__

    namespace cpu {

        inline std::vector<float> mean_pooling(const float *token_embeddings, const int32_t *attention_mask,
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
    } // namespace cpu
} // namespace neural_network
#endif