
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "config.h"
#include "database.h"
#include "event.h"
#include "neural_network/nn.h"
#include <cstdlib>
#include <cstring>
#include <exception>

int main(int argc, char *argv[]) {
    const auto log_level = std::getenv("LOG_LEVEL");
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
        } catch (const std::exception& e) {
            spdlog::error("Failed to parse LOG_LEVEL: {}", e.what());
        }
    }

    Config::init();
    bot_cmd::init_command_map();
    try {

        database::init_db_connection();
        spdlog::info("Init database connection successed.");
    } catch (std::exception &e) {
        spdlog::error("Init database connection error: {}", e.what());
    }
    neural_network::init_onnx_runtime();

    if (argc > 1) {
        if (strcmp(argv[1], "init_message_table") == 0) {
            database::get_global_db_connection().create_message_record_table();
            spdlog::info("Init message record table successed.");
            return 0;
        } else if (strcmp(argv[1], "init_tools_call_record_table") == 0) {
            database::get_global_db_connection().create_tools_call_record_table();
            spdlog::info("Init tools call record table successed.");
        }
    }

    bot_adapter::BotAdapter adapter("ws://localhost:13378/all", Config::instance().bot_id);
    adapter.update_bot_profile();

    register_event(adapter);
    adapter.start();
    return 0;
}