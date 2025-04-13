#include "config.h"
#include "utils.h"
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <string_view>
#include <unordered_set>

std::string LLM_API_URL;
std::string LLM_API_TOKEN;
std::string LLM_MODEL_NAME;
std::string CUSTOM_SYSTEM_PROMPT;
// std::string BOT_NAME;
// MiraiCP::QQID BOT_QQID;

uint64_t BOT_ID;

std::string NET_SEARCH_API_URL;
std::string NET_SEARCH_TOKEN;

std::string URL_SEARCH_API_URL;
std::string URL_SEARCH_TOKEN;

std::string MSG_DB_URL = "http://localhost:8080/v1";

std::unordered_set<std::string> ADMIN_ID_SET;
std::unordered_set<std::string> BANNED_ID_SET;


void init_config() {
    const auto api_url = std::getenv("AIBOT_LLM_API_URL");
    LLM_API_URL = std::string(api_url != nullptr ? api_url : "");
    spdlog::info("LLM_API_URL: {}", LLM_API_URL);

    const auto api_token = std::getenv("AIBOT_LLM_API_TOKEN");
    LLM_API_TOKEN = std::string(api_token != nullptr ? api_token : "");

    const auto net_search_api_url = std::getenv("AIBOT_NET_SEARCH_API_URL");
    NET_SEARCH_API_URL = std::string(net_search_api_url != nullptr ? net_search_api_url : "");
    spdlog::info("NET_SEARCH_API_URL: {}", NET_SEARCH_API_URL);

    const auto net_search_token = std::getenv("AIBOT_NET_SEARCH_TOKEN");
    NET_SEARCH_TOKEN = std::string(net_search_token != nullptr ? net_search_token : "");
    spdlog::info("NET_SEARCH_TOKEN: {}", NET_SEARCH_TOKEN);

    const auto url_search_api_url = std::getenv("AIBOT_URL_SEARCH_API_URL");
    URL_SEARCH_API_URL = std::string(url_search_api_url != nullptr ? url_search_api_url : "");
    spdlog::info("URL_SEARCH_API_URL: {}", URL_SEARCH_API_URL);

    const auto url_search_token = std::getenv("AIBOT_URL_SEARCH_TOKEN");
    URL_SEARCH_TOKEN = std::string(url_search_token != nullptr ? url_search_token : "");
    spdlog::info("URL_SEARCH_TOKEN: {}", URL_SEARCH_TOKEN);

    const auto model_name = std::getenv("AIBOT_LLM_MODEL_NAME");
    LLM_MODEL_NAME = std::string(model_name != nullptr ? model_name : "");
    spdlog::info("LLM_MODEL_NAME: {}", LLM_MODEL_NAME);

    const auto custom_system_prompt = std::getenv("AIBOT_CUSTOM_SYSTEM_PROMPT");
    CUSTOM_SYSTEM_PROMPT = std::string(custom_system_prompt != nullptr ? custom_system_prompt : "");
    spdlog::info("CUSTOM_SYSTEM_PROMPT: {}", CUSTOM_SYSTEM_PROMPT);

    const auto db_url = std::getenv("AIBOT_MSG_DB_URL");
    MSG_DB_URL = std::string(db_url != nullptr ? db_url : "");
    spdlog::info("MSG_DB_URL: {}", MSG_DB_URL);

    const auto bot_id_str = std::getenv("AIBOT_BOT_ID");
    if (bot_id_str == nullptr) {
        throw std::runtime_error("AIBOT_BOT_ID environment variable not set");
    }
    
    try {
        BOT_ID = std::stoull(bot_id_str);
        spdlog::info("BOT_ID: {}", BOT_ID);
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("Invalid BOT_ID format: {}", e.what()));
    }

    const auto admin_id_list = std::getenv("AIBOT_ADMIN_ID_LIST");
    if (admin_id_list != nullptr) {
        spdlog::info("AIBOT_ADMIN_ID_LIST: {}", admin_id_list);
        for (auto e : SplitString(admin_id_list, ',')) {
            auto admin = std::string(ltrim(rtrim(e)));
            spdlog::info("admin id: {}", admin);
            ADMIN_ID_SET.emplace(admin);
        }
    }

    const auto banned_id_list = std::getenv("AIBOT_BANNED_ID_LIST");
    if (banned_id_list != nullptr) {
        spdlog::info("AIBOT_BANNED_ID_LIST: {}", banned_id_list);  // Fixed variable name from admin_id_list to banned_id_list
        for (auto e : SplitString(banned_id_list, ',')) {
            auto ban = std::string(ltrim(rtrim(e)));
            spdlog::info("banned id: {}", ban);
            BANNED_ID_SET.emplace(ban);
        }
    }


    // BOT_NAME = std::string(bot_name);
    // MiraiCP::Logger::logger.info(fmt::format("BOT_NAME: {}", BOT_NAME));
    // BOT_QQID = id;
    // MiraiCP::Logger::logger.info(fmt::format("BOT_QQID: {}", id));
}