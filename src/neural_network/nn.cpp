#include "neural_network/nn.h"
#include <cinttypes>
#include <memory>
#include <stdexcept>

#ifdef __USE_LIBTORCH__
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <fmt/format.h>
#endif

#ifdef __USE_ONNX_RUNTIME__

std::unique_ptr<Ort::Env> g_onnx_runtime_ptr;

void neural_network::init_onnx_runtime() {
    g_onnx_runtime_ptr = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AIBot-800a");
}

Ort::Env &neural_network::get_onnx_runtime() {
    if (g_onnx_runtime_ptr == nullptr) {
        throw std::runtime_error("get_onnx_runtime(): onnx_runtime hasn't initialized.");
    }
    return *g_onnx_runtime_ptr;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_cpu() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_core_ml() {
    Ort::SessionOptions opts;
    std::unordered_map<std::string, std::string> provider_options;
    // provider_options["ModelFormat"] = "MLProgram";
    provider_options["RequireStaticInputShapes"] = "0";
    provider_options["EnableOnSubgraphs"] = "0";
    provider_options["MLComputeUnits"] = "ALL";
    // opts.SetIntraOpNumThreads(8);
    opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

    // 添加 CoreML 执行提供程序
    opts.AppendExecutionProvider("CoreML", provider_options);
    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_cuda() {
    Ort::SessionOptions opts;
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    opts.AppendExecutionProvider_CUDA(cuda_options);
    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_tensorrt() {
    Ort::SessionOptions opts;
    OrtTensorRTProviderOptions tensorrt_options;
    tensorrt_options.device_id = 0;
    tensorrt_options.trt_engine_cache_enable = 1;
    tensorrt_options.trt_engine_cache_path = "models/tensorrt_cache";
    tensorrt_options.trt_fp16_enable = 1;
    tensorrt_options.trt_int8_enable = 0;
    tensorrt_options.trt_int8_calibration_table_name = "";
    tensorrt_options.trt_max_workspace_size = 1 << 30; // 1GB
    tensorrt_options.trt_min_subgraph_size = 3; // 最小子图大小
    opts.AppendExecutionProvider_TensorRT(tensorrt_options);
    return opts;
}

#endif // __USE_ONNX_RUNTIME__

#ifdef __USE_LIBTORCH__

neural_network::CosineSimilarityModel::CosineSimilarityModel(const std::string &model_path, Device device) 
    : m_device(get_torch_device(device)) {
    try {
        m_module = torch::jit::load(model_path, m_device);
        m_module.eval(); // Set to evaluation mode
        
        // Log model information for debugging
        auto parameters = m_module.parameters();
        auto param_iter = parameters.begin();
        size_t param_count = 0;
        for (auto iter = parameters.begin(); iter != parameters.end(); ++iter) {
            param_count++;
        }
        
        auto buffers = m_module.buffers();
        size_t buffer_count = 0;
        for (auto iter = buffers.begin(); iter != buffers.end(); ++iter) {
            buffer_count++;
        }
        
        spdlog::info("Successfully loaded PyTorch CosineSimilarityModel from: {}", model_path);
        spdlog::info("Model has {} parameters and {} buffers", param_count, buffer_count);
        
        // For CosineSimilarityModel, having no parameters is normal since it only performs mathematical operations
        if (param_count == 0 && buffer_count == 0) {
            spdlog::debug("Model has no parameters or buffers - this is normal for mathematical operation models like CosineSimilarityModel");
        }
        
    } catch (const std::exception &e) {
        spdlog::error("Failed to load PyTorch CosineSimilarityModel from {}: {}", model_path, e.what());
        throw;
    }
}

neural_network::emb_vec_t neural_network::CosineSimilarityModel::inference(emb_vec_t target, emb_mat_t value_list) {
    // Validate input dimensions
    if (target.size() != COSINE_SIMILARITY_INPUT_EMB_SIZE) {
        throw std::invalid_argument(
            fmt::format("Target vector must be size {}, got {}", COSINE_SIMILARITY_INPUT_EMB_SIZE, target.size()));
    }
    
    for (const auto &vec : value_list) {
        if (vec.size() != COSINE_SIMILARITY_INPUT_EMB_SIZE) {
            throw std::invalid_argument(
                fmt::format("All vectors in value_list must be size {}, got {}", COSINE_SIMILARITY_INPUT_EMB_SIZE, vec.size()));
        }
    }

    if (value_list.empty()) {
        return {};
    }

    // Prepare input tensors
    // Use the device from constructor
    
    // Create target tensor (shape: [1, COSINE_SIMILARITY_INPUT_EMB_SIZE])
    torch::Tensor target_tensor = torch::tensor(target, torch::dtype(torch::kFloat32))
                                    .unsqueeze(0)  // Add batch dimension
                                    .to(m_device);

    // Create value tensor (shape: [num_samples, COSINE_SIMILARITY_INPUT_EMB_SIZE])
    std::vector<float> flattened_values;
    flattened_values.reserve(value_list.size() * COSINE_SIMILARITY_INPUT_EMB_SIZE);
    for (const auto &vec : value_list) {
        flattened_values.insert(flattened_values.end(), vec.begin(), vec.end());
    }
    
    torch::Tensor value_tensor = torch::tensor(flattened_values, torch::dtype(torch::kFloat32))
                                   .view({static_cast<int64_t>(value_list.size()), 
                                         static_cast<int64_t>(COSINE_SIMILARITY_INPUT_EMB_SIZE)})
                                   .to(m_device);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(target_tensor);
    inputs.push_back(value_tensor);

    torch::jit::IValue output = m_module.forward(inputs);
    torch::Tensor output_tensor = output.toTensor();

    // Convert back to vector
    output_tensor = output_tensor.to(torch::kCPU);
    auto output_ptr = output_tensor.data_ptr<float>();
    auto output_size = output_tensor.numel();

    return emb_vec_t(output_ptr, output_ptr + output_size);
}

#endif // __USE_LIBTORCH__