#include "cli_handler.h"
#include "database.h"
#include "neural_network/nn.h"
#include <cstring>
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
