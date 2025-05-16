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

Ort::SessionOptions neural_network::get_onnx_session_opts() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(8);
    return opts;
}

Ort::SessionOptions neural_network::get_onnx_session_opts_core_ml() {
    Ort::SessionOptions opts;

    std::unordered_map<std::string, std::string> provider_options;
    provider_options["ModelFormat"] = "MLProgram";
    // provider_options["MLComputeUnits"] = "ALL";
    provider_options["RequireStaticInputShapes"] = "0";
    provider_options["EnableOnSubgraphs"] = "0";
    // provider_options["UseCoreMLFallback"] = "0";
    provider_options["MLComputeUnits"] = "ALL";
    opts.SetIntraOpNumThreads(8);

    // 添加 CoreML 执行提供程序
    opts.AppendExecutionProvider("CoreML", provider_options);
    return opts;
}