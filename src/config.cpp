#include "config.h"
#include "utils.h"
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <string_view>
#include <unordered_set>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>

std::string LLM_API_URL;
std::string LLM_API_TOKEN;
std::string LLM_MODEL_NAME;
std::string LLM_DEEP_THINK_MODEL_NAME;
std::string CUSTOM_SYSTEM_PROMPT;
std::optional<std::string> CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION;
// std::string BOT_NAME;
// MiraiCP::QQID BOT_QQID;

uint64_t BOT_ID;

std::string NET_SEARCH_API_URL;
std::string NET_SEARCH_TOKEN;

std::string URL_SEARCH_API_URL;
std::string URL_SEARCH_TOKEN;

std::string VEC_DB_URL = "http://localhost:8080/v1";

std::unordered_set<std::string> ADMIN_ID_SET;
std::unordered_set<std::string> BANNED_ID_SET;

namespace fs = boost::filesystem;


void init_config() {
    // load config from yaml

    if (fs::exists("config.yaml")) {
        // throw std::runtime_error("config.yaml not found");
        spdlog::info("config.yaml found, loading...");
        const YAML::Node node = YAML::LoadFile("config.yaml");

        if (node["llm_api_url"]) {
            LLM_API_URL = node["llm_api_url"].as<std::string>();
            spdlog::info("LLM_API_URL: {}", LLM_API_URL);
        }
        if (node["llm_api_token"]) {
            LLM_API_TOKEN = node["llm_api_token"].as<std::string>();
            spdlog::info("LLM_API_TOKEN: {}", LLM_API_TOKEN);
        }

        if (node["llm_model_name"]) {
            LLM_MODEL_NAME = node["llm_model_name"].as<std::string>();
            spdlog::info("LLM_MODEL_NAME: {}", LLM_MODEL_NAME);
        }

        if (node["llm_deep_think_model_name"]) {
            LLM_DEEP_THINK_MODEL_NAME = node["llm_deep_think_model_name"].as<std::string>();
            spdlog::info("LLM_DEEP_THINK_MODEL_NAME: {}", LLM_DEEP_THINK_MODEL_NAME);
        }

        if (node["custom_system_prompt"]) {
            CUSTOM_SYSTEM_PROMPT = node["custom_system_prompt"].as<std::string>();
            spdlog::info("CUSTOM_SYSTEM_PROMPT: {}", CUSTOM_SYSTEM_PROMPT);
        }

        if (node["custom_deep_think_system_prompt"]) {
            CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION = node["custom_deep_think_system_prompt"].as<std::string>();
            spdlog::info("CUSTOM_DEEP_THINK_SYSTEM_PROMPT: {}", CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION.value());
        }

        if (node["net_search_api_url"]) {
            NET_SEARCH_API_URL = node["net_search_api_url"].as<std::string>();
            spdlog::info("NET_SEARCH_API_URL: {}", NET_SEARCH_API_URL);
        }

        if (node["net_search_token"]) {
            NET_SEARCH_TOKEN = node["net_search_token"].as<std::string>();
            spdlog::info("NET_SEARCH_TOKEN: {}", NET_SEARCH_TOKEN);
        }

        if (node["url_search_api_url"]) {
            URL_SEARCH_API_URL = node["url_search_api_url"].as<std::string>();
            spdlog::info("URL_SEARCH_API_URL: {}", URL_SEARCH_API_URL);
        }

        if (node["url_search_token"]) {
            URL_SEARCH_TOKEN = node["url_search_token"].as<std::string>();
            spdlog::info("URL_SEARCH_TOKEN: {}", URL_SEARCH_TOKEN);
        }

        if (node["vec_db_url"]) {
            VEC_DB_URL = node["vec_db_url"].as<std::string>();
            spdlog::info("VEC_DB_URL: {}", VEC_DB_URL);
        }

        if (node["bot_id"]) {
            BOT_ID = node["bot_id"].as<uint64_t>();
            spdlog::info("BOT_ID: {}", BOT_ID);
        }

        if (node["admin_id_list"] && node["admin_id_list"].IsSequence()) {
            for (const auto& id : node["admin_id_list"]) {
                ADMIN_ID_SET.emplace(id.as<std::string>());
                spdlog::info("admin id: {}", id.as<std::string>());
            }
        }

        if (node["banned_id_list"] && node["banned_id_list"].IsSequence()) {
            for (const auto& id : node["banned_id_list"]) {
                BANNED_ID_SET.emplace(id.as<std::string>());
                spdlog::info("banned id: {}", id.as<std::string>());
            }
        }
    }



    // load config from env
    if (const auto api_url = std::getenv("AIBOT_LLM_API_URL")) {
        LLM_API_URL = std::string(api_url);
        spdlog::info("LLM_API_URL: {} (from env)", LLM_API_URL);
    }

    if (const auto api_token = std::getenv("AIBOT_LLM_API_TOKEN")) {
        LLM_API_TOKEN = std::string(api_token);
    }

    if (const auto net_search_api_url = std::getenv("AIBOT_NET_SEARCH_API_URL")) {
        NET_SEARCH_API_URL = std::string(net_search_api_url);
        spdlog::info("NET_SEARCH_API_URL: {} (from env)", NET_SEARCH_API_URL);
    }

    if (const auto net_search_token = std::getenv("AIBOT_NET_SEARCH_TOKEN")) {
        NET_SEARCH_TOKEN = std::string(net_search_token);
        spdlog::info("NET_SEARCH_TOKEN: {} (from env)", NET_SEARCH_TOKEN);
    }

    if (const auto url_search_api_url = std::getenv("AIBOT_URL_SEARCH_API_URL")) {
        URL_SEARCH_API_URL = std::string(url_search_api_url);
        spdlog::info("URL_SEARCH_API_URL: {} (from env)", URL_SEARCH_API_URL);
    }

    if (const auto url_search_token = std::getenv("AIBOT_URL_SEARCH_TOKEN")) {
        URL_SEARCH_TOKEN = std::string(url_search_token);
        spdlog::info("URL_SEARCH_TOKEN: {} (from env)", URL_SEARCH_TOKEN);
    }

    if (const auto model_name = std::getenv("AIBOT_LLM_MODEL_NAME")) {
        LLM_MODEL_NAME = std::string(model_name);
        spdlog::info("LLM_MODEL_NAME: {} (from env)", LLM_MODEL_NAME);
    }

    if (const auto deep_think_model_name = std::getenv("AIBOT_LLM_DEEP_THINK_MODEL_NAME")) {
        LLM_DEEP_THINK_MODEL_NAME = std::string(deep_think_model_name);
        spdlog::info("LLM_DEEP_THINK_MODEL_NAME: {} (from env)", LLM_DEEP_THINK_MODEL_NAME);
    }

    if (const auto custom_system_prompt = std::getenv("AIBOT_CUSTOM_SYSTEM_PROMPT")) {
        CUSTOM_SYSTEM_PROMPT = std::string(custom_system_prompt);
        spdlog::info("CUSTOM_SYSTEM_PROMPT: {} (from env)", CUSTOM_SYSTEM_PROMPT);
    }

    const auto custom_deep_think_system_prompt_option = std::getenv("AIBOT_CUSTOM_DEEP_THINK_SYSTEM_PROMPT");
    if (custom_deep_think_system_prompt_option != nullptr) {
        CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION = std::string(custom_deep_think_system_prompt_option);
        spdlog::info("CUSTOM_DEEP_THINK_SYSTEM_PROMPT: {} (from env)", CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION.value());
    } else {
        CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION = std::nullopt;
    }

    const auto db_url = std::getenv("AIBOT_MSG_DB_URL");
    if (db_url != nullptr) {
        VEC_DB_URL = std::string(db_url);
        spdlog::info("MSG_DB_URL: {} (from env)", VEC_DB_URL);
    }

    const auto bot_id_str = std::getenv("AIBOT_BOT_ID");
    if (bot_id_str != nullptr) {
        // throw std::runtime_error("AIBOT_BOT_ID environment variable not set");
        try {
            BOT_ID = std::stoull(bot_id_str);
            spdlog::info("BOT_ID: {} (from env)", BOT_ID);
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format("Invalid BOT_ID format: {}", e.what()));
        }
    }
    


    const auto admin_id_list = std::getenv("AIBOT_ADMIN_ID_LIST");
    if (admin_id_list != nullptr) {
        spdlog::info("AIBOT_ADMIN_ID_LIST: {} (from env)", admin_id_list);
        for (auto e : SplitString(admin_id_list, ',')) {
            auto admin = std::string(ltrim(rtrim(e)));
            spdlog::info("admin id: {} (from env)", admin);
            ADMIN_ID_SET.emplace(admin);
        }
    }

    const auto banned_id_list = std::getenv("AIBOT_BANNED_ID_LIST");
    if (banned_id_list != nullptr) {
        spdlog::info("AIBOT_BANNED_ID_LIST: {} (from env)", banned_id_list);
        for (auto e : SplitString(banned_id_list, ',')) {
            auto ban = std::string(ltrim(rtrim(e)));
            spdlog::info("banned id: {} (from env)", ban);
            BANNED_ID_SET.emplace(ban);
        }
    }


    // BOT_NAME = std::string(bot_name);
    // MiraiCP::Logger::logger.info(fmt::format("BOT_NAME: {}", BOT_NAME));
    // BOT_QQID = id;
    // MiraiCP::Logger::logger.info(fmt::format("BOT_QQID: {}", id));
}