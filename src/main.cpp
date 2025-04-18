
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "config.h"
#include "event.h"
#include <cstdlib>

int main() {
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

    init_config();
    bot_cmd::init_command_map();

    bot_adapter::BotAdapter adapter("ws://localhost:13378/all", BOT_ID);
    adapter.update_bot_profile();

    register_event(adapter);
    adapter.start();
    return 0;
}