#include "neural_network/nn.h"
#include <cinttypes>
#include <memory>
#include <stdexcept>

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