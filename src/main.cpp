
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "config.h"
#include "database.h"
#include "event.h"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

int main(int argc, char *argv[]) {
    const auto log_level = std::getenv("LOG_LEVEL");

    int rotation_hour = 0;
    int rotation_minute = 0;

    auto daily_file_sink =
        std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/aibot/aibot_800a.txt",
                                                            rotation_hour, rotation_minute);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks {console_sink, daily_file_sink};
    auto logger = std::make_shared<spdlog::logger>("aibot_800a", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    
    
    if (log_level != nullptr) {
        try {
            // Convert string to lowercase for case-insensitive comparison
            std::string level_str(log_level);
            std::transform(level_str.begin(), level_str.end(), level_str.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            // Map string values to spdlog levels
            if (level_str == "trace") {
                spdlog::set_level(spdlog::level::trace);
            } else if (level_str == "debug") {
                spdlog::set_level(spdlog::level::debug);
            } else if (level_str == "info") {
                spdlog::set_level(spdlog::level::info);
            } else if (level_str == "warn" || level_str == "warning") {
                spdlog::set_level(spdlog::level::warn);
            } else if (level_str == "error") {
                spdlog::set_level(spdlog::level::err);
            } else if (level_str == "critical" || level_str == "fatal") {
                spdlog::set_level(spdlog::level::critical);
            } else if (level_str == "off") {
                spdlog::set_level(spdlog::level::off);
            } else {
                spdlog::warn("Unknown LOG_LEVEL value: {}, using default", log_level);
            }
        } catch (const std::exception &e) {
            spdlog::error("Failed to parse LOG_LEVEL: {}", e.what());
        }
    }
    spdlog::info("Set logging level to {}", spdlog::level::to_string_view(spdlog::get_level()));

    Config::init();
    bot_cmd::init_command_map();
    try {

        database::init_db_connection();
        spdlog::info("Init database connection successed.");
    } catch (std::exception &e) {
        spdlog::error("Init database connection error: {}", e.what());
    }

    if (argc > 1) {
        if (strcmp(argv[1], "init_message_table") == 0) {
            database::get_global_db_connection().create_message_record_table();
            spdlog::info("Init message record table successed.");
            return 0;
        } else if (strcmp(argv[1], "init_tools_call_record_table") == 0) {
            database::get_global_db_connection().create_tools_call_record_table();
            spdlog::info("Init tools call record table successed.");
            return 0;
        } else if (strcmp(argv[1], "init_user_preference_table") == 0) {
            database::get_global_db_connection().create_user_preference_table();
            spdlog::info("Init user preference table successed.");
            return 0;
        } else if (strcmp(argv[1], "init_user_protait_table") == 0) {
            database::get_global_db_connection().create_user_protait_table();
            spdlog::info("Init user protait table successed.");
            return 0;
        } else if (strcmp(argv[1], "init_all_tables") == 0) {
            database::get_global_db_connection().create_message_record_table();
            database::get_global_db_connection().create_tools_call_record_table();
            database::get_global_db_connection().create_user_preference_table();
            database::get_global_db_connection().create_user_protait_table();
            spdlog::info("Init all tables successed.");
            return 0;
        }
    }

#ifdef __USE_ONNX_RUNTIME__
    neural_network::init_onnx_runtime();
#endif

    neural_network::Device device = neural_network::Device::CPU;
    if (argc > 1) {
        if (strcmp(argv[1], "--use-coreml") == 0) {
            device = neural_network::Device::CoreML;
            spdlog::info("Using Apple CoreML for neural network inference.");
        } else if (strcmp(argv[1], "--use-mps") == 0) {
            device = neural_network::Device::MPS;
            spdlog::info("Using Apple Metal Performance Shaders for neural network inference.");
        }else if (strcmp(argv[1], "--use-cuda") == 0) {
            device = neural_network::Device::CUDA;
            spdlog::info("Using CUDA for neural network inference.");
        } else if (strcmp(argv[1], "--use-tensorrt") == 0) {
            device = neural_network::Device::TensorRT;
            spdlog::info("Using TensorRT for neural network inference.");
        } else if (strcmp(argv[1], "--use-cpu") == 0) {
            device = neural_network::Device::CPU;
            spdlog::info("Using CPU for neural network inference.");
        }
    }

    neural_network::init_model_set(device);

    bot_adapter::BotAdapter adapter("ws://localhost:13378/all", Config::instance().bot_id);
    adapter.update_bot_profile();

    register_event(adapter);
    adapter.start();
    return 0;
}