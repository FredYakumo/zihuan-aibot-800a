#include "config.h"
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <MiraiCP.hpp>
#include <string_view>

std::string LLM_API_URL;
std::string LLM_API_TOKEN;
std::string LLM_MODEL_NAME;
std::string CUSTOM_SYSTEM_PROMPT;
// std::string BOT_NAME;
// MiraiCP::QQID BOT_QQID;

std::string MSG_DB_URL = "http://localhost:28080/v1";

void init_config() {
    const auto api_url = std::getenv("AIBOT_LLM_API_URL");
    LLM_API_URL = std::string(api_url != nullptr ? api_url : "");
    MiraiCP::Logger::logger.info(fmt::format("LLM_API_URL: {}", LLM_API_URL));

    const auto api_token = std::getenv("AIBOT_LLM_API_TOKEN");
    LLM_API_TOKEN = std::string(api_token != nullptr ? api_token : "");

    const auto model_name = std::getenv("AIBOT_LLM_MODEL_NAME");
    LLM_MODEL_NAME = std::string(model_name != nullptr ? model_name : "");
    MiraiCP::Logger::logger.info(fmt::format("LLM_MODEL_NAME: {}", LLM_MODEL_NAME));

    const auto custom_system_prompt = std::getenv("AIBOT_CUSTOM_SYSTEM_PROMPT");
    CUSTOM_SYSTEM_PROMPT = std::string(custom_system_prompt != nullptr ? custom_system_prompt : "");
    MiraiCP::Logger::logger.info(fmt::format("CUSTOM_SYSTEM_PROMPT: {}", CUSTOM_SYSTEM_PROMPT));

    const auto db_url = std::getenv("AIBOT_MSG_DB_URL");
    MSG_DB_URL = std::string(db_url != nullptr ? db_url : "");
    MiraiCP::Logger::logger.info(fmt::format("MSG_DB_URL: {}", MSG_DB_URL));

    // BOT_NAME = std::string(bot_name);
    // MiraiCP::Logger::logger.info(fmt::format("BOT_NAME: {}", BOT_NAME));
    // BOT_QQID = id;
    // MiraiCP::Logger::logger.info(fmt::format("BOT_QQID: {}", id));
}