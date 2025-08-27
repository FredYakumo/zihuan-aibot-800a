#include "cli_handler.h"
#include "constants.hpp"
#include "database.h"
#include "neural_network/nn.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <spdlog/spdlog.h>

bool CLIHandler::handle_table_init(int argc, char *argv[]) {
    if (argc <= 1) return false;

    const char *cmd = argv[1];
    if (std::strcmp(cmd, "init_message_table") == 0) {
        database::get_global_db_connection().create_message_record_table();
        spdlog::info("Init message record table successed.");
        return true;
    } else if (std::strcmp(cmd, "init_tools_call_record_table") == 0) {
        database::get_global_db_connection().create_tools_call_record_table();
        spdlog::info("Init tools call record table successed.");
        return true;
    } else if (std::strcmp(cmd, "init_user_preference_table") == 0) {
        database::get_global_db_connection().create_user_preference_table();
        spdlog::info("Init user preference table successed.");
        return true;
    } else if (std::strcmp(cmd, "init_user_protait_table") == 0 || std::strcmp(cmd, "init_user_protait_table") == 0) {
        // Keep original typo alias 'init_user_protait_table' and earlier 'init_user_protait_table'
        database::get_global_db_connection().create_user_protait_table();
        spdlog::info("Init user protait table successed.");
        return true;
    } else if (std::strcmp(cmd, "init_all_tables") == 0) {
        database::get_global_db_connection().create_message_record_table();
        database::get_global_db_connection().create_tools_call_record_table();
        database::get_global_db_connection().create_user_preference_table();
        database::get_global_db_connection().create_user_protait_table();
        spdlog::info("Init all tables successed.");
        return true;
    }
    return false;
}

std::optional<neural_network::Device> CLIHandler::parse_device(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--use-coreml") == 0) {
            return neural_network::Device::CoreML;
        } else if (std::strcmp(argv[i], "--use-mps") == 0) {
            return neural_network::Device::MPS;
        } else if (std::strcmp(argv[i], "--use-cuda") == 0) {
            return neural_network::Device::CUDA;
        } else if (std::strcmp(argv[i], "--use-tensorrt") == 0) {
            return neural_network::Device::TensorRT;
        } else if (std::strcmp(argv[i], "--use-cpu") == 0) {
            return neural_network::Device::CPU;
        }
    }
    return std::nullopt;
}

neural_network::Device CLIHandler::determine_device(int argc, char *argv[]) {
    auto device = parse_device(argc, argv).value_or(neural_network::Device::CPU);
    switch (device) {
    case neural_network::Device::CoreML:
        spdlog::info("Using Apple CoreML for neural network inference.");
        break;
    case neural_network::Device::MPS:
        spdlog::info("Using Apple Metal Performance Shaders for neural network inference.");
        break;
    case neural_network::Device::CUDA:
        spdlog::info("Using CUDA for neural network inference.");
        break;
    case neural_network::Device::TensorRT:
        spdlog::info("Using TensorRT for neural network inference.");
        break;
    case neural_network::Device::CPU:
        spdlog::info("Using CPU for neural network inference.");
        break;
    default:
        break; // Future devices
    }
    return device;
}

namespace {
    uint64_t s_bot_id = 0;
}

std::optional<uint64_t> CLIHandler::parse_bot_id(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            char *end = nullptr;
            const auto *val = argv[i + 1];
            if (*val == '\0') return std::nullopt;
            errno = 0;
            unsigned long long parsed = std::strtoull(val, &end, 10);
            if (errno != 0 || end == val || *end != '\0') {
                spdlog::error("Invalid bot id: {}", val);
                return std::nullopt;
            }
            return static_cast<uint64_t>(parsed);
        }
    }
    return std::nullopt;
}

void CLIHandler::process_args(int argc, char *argv[]) {
    if (auto bot_id_opt = parse_bot_id(argc, argv)) {
        s_bot_id = *bot_id_opt;
        spdlog::info("Bot id set from CLI: {}", s_bot_id);
    } else if (s_bot_id == 0) {
        spdlog::warn("No bot id specified via -l <bot_id>; current value is 0. Some features may fail.");
    } else {
        spdlog::info("Bot id retained from previous runtime value: {}", s_bot_id);
    }
}

uint64_t CLIHandler::get_bot_id() { return s_bot_id; }

bool CLIHandler::handle_version_request(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-v") == 0) {
            std::cout << "ZiHuanAIBot version: " << BUILD_VERSION_STRING << std::endl;
            std::cout << "Repository: " << DREPOS_ADDR_STRING << std::endl;
            
            // Platform information
            std::cout << "Platform: ";
#ifdef PLATFORM_MACOS
            std::cout << "macOS";
#elif defined(PLATFORM_LINUX)
            std::cout << "Linux";
#elif defined(PLATFORM_WINDOWS)
            std::cout << "Windows";
#else
            std::cout << "Unknown";
#endif
            std::cout << std::endl;
            
            // Build type
            std::cout << "Build Type: ";
#ifdef DEBUG_BUILD
            std::cout << "Debug";
#else
            std::cout << "Release";
#endif
            std::cout << std::endl;
            
            // AI inference backend
            std::cout << "AI Backend: ";
#ifdef __USE_LIBTORCH__
            std::cout << "LibTorch";
#else
            std::cout << "ONNX Runtime";
#endif
            std::cout << std::endl;
            
            // Custom macros
            std::cout << "Features: ";
            bool first = true;
#ifdef AIBOT_VERSION_800A
            if (!first) std::cout << ", ";
            std::cout << "AIBot 800A";
            first = false;
#endif
#ifdef __USE_PADDLE_INFERENCE__
            if (!first) std::cout << ", ";
            std::cout << "Paddle Inference";
            first = false;
#endif
            if (first) {
                std::cout << "None";
            }
            std::cout << std::endl;
            
            return true;
        }
    }
    return false;
}
