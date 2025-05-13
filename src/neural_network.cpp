#include "neural_network/nn.h"
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
    opts.SetIntraOpNumThreads(4);
    return opts;
}