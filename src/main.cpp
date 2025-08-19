
#include "agent/agent.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "cli_handler.h"
#include "config.h"
#include "daily_rotating_file_sink.h"
#include "database.h"
#include "event.h"
#include "global_data.h"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace {
    // Configure spdlog global level from LOG_LEVEL env variable.
    void configure_log_level_from_env() {
        const char *log_level = std::getenv("LOG_LEVEL");
        if (!log_level) {
            return; // Keep default level
        }
        try {
            std::string level_str(log_level);
            std::transform(level_str.begin(), level_str.end(), level_str.begin(),
                           [](unsigned char c) { return std::tolower(c); });

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
} // namespace

int main(int argc, char *argv[]) {

    int rotation_hour = 0;
    int rotation_minute = 0;

    // Custom daily + size rotating file sink
    // File rotation: aibot_800a_2025-07-21.txt (current) -> .1 (oldest) -> .2, .3... (newer) -> highest# (newest
    // archived)
    auto daily_size_sink = std::make_shared<spdlog::sinks::daily_rotating_file_sink_mt>(
        "logs/aibot/aibot_800a", 10 * 1024 * 1024, rotation_hour, rotation_minute);

    // Latest log file for current run (custom size-only rotation)
    // File rotation: latest.txt (current) -> .1 (oldest) -> .2, .3... (newer) -> highest# (newest archived)
    auto latest_file_sink = std::make_shared<spdlog::sinks::daily_rotating_file_sink_mt>(
        "logs/aibot/latest", 10 * 1024 * 1024, 0, 0, false); // Disable daily rotation

    // Console output
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    std::vector<spdlog::sink_ptr> sinks{console_sink, daily_size_sink, latest_file_sink};
    auto logger = std::make_shared<spdlog::logger>("aibot_800a", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);

    configure_log_level_from_env();
    spdlog::info("Set logging level to {}", spdlog::level::to_string_view(spdlog::get_level()));

    Config::init();
    bot_cmd::init_command_map();

    try {

        database::init_db_connection();
        spdlog::info("Init database connection successed.");
    } catch (std::exception &e) {
        spdlog::error("Init database connection error: {}", e.what());
    }

    if (CLIHandler::handle_table_init(argc, argv))
        return 0;

#ifdef __USE_ONNX_RUNTIME__
    neural_network::init_onnx_runtime();
#endif

    CLIHandler::process_args(argc, argv);

    neural_network::Device device = CLIHandler::determine_device(argc, argv);

    neural_network::init_model_set(device);

    const auto config = Config::instance();

    g_llm_chat_agent = std::make_shared<agent::LLMAPIAgentBase>(
        config.llm_model_name, fmt::format("{}:{}", config.llm_api_url, config.llm_api_port), config.llm_api_key);

    bot_adapter::BotAdapter adapter("ws://localhost:13378/all", CLIHandler::get_bot_id());
    adapter.update_bot_profile();

    // 初始化简单聊天动作 Agent (供原 process_llm 调用)
    g_simple_chat_action_agent = std::make_shared<agent::SimpleChatActionAgent>(
        std::shared_ptr<bot_adapter::BotAdapter>(&adapter, [](bot_adapter::BotAdapter*){}), g_llm_chat_agent);

    register_event(adapter);
    adapter.start();
    return 0;
}