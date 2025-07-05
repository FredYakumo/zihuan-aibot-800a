#include "neural_network/nn.h"
#include <cinttypes>
#include <memory>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
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
neural_network::CosineSimilarityONNXModel cosine_similarity_onnx_model{"models/cosine_similarity.onnx", neural_network::get_onnx_session_opts()};

Ort::SessionOptions neural_network::get_onnx_session_opts() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(8);
    opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);

    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_core_ml() {
    Ort::SessionOptions opts;
    std::unordered_map<std::string, std::string> provider_options;
    provider_options["ModelFormat"] = "MLProgram";
    provider_options["RequireStaticInputShapes"] = "0";
    provider_options["EnableOnSubgraphs"] = "0";
    provider_options["MLComputeUnits"] = "ALL";
    opts.SetIntraOpNumThreads(8);
    opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_FATAL);
    
    // 添加 CoreML 执行提供程序
    opts.AppendExecutionProvider("CoreML", provider_options);
    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_tensorrt() {
    Ort::SessionOptions opts;
    // OrtCUDAProviderOptionsV2 tensorrt_options;
    // tensorrt_options.device_id = 0;                      // 指定 GPU 设备编号
    // tensorrt_options.trt_max_workspace_size = 1 << 30;     // 设置最大工作内存（例如：1GB）
    // tensorrt_options.trt_fp16_enable = 0;                // 是否启用 FP16 模式, 0：禁用；1：启用
    // tensorrt_options.trt_int8_enable = 0;                // 是否启用 INT8 模式
    // tensorrt_options.trt_engine_cache_enable = 0;        // 是否启用引擎缓存
    // tensorrt_options.trt_context_memory_sharing_enable = 0;// 是否启用上下文内存共享
    // opts.AppendExecutionProvider_TensorRT_V2(opts, tensorrt_options);

    // opts.AppendExecutionProvider_CUDA_V2()
    
    return opts;
}