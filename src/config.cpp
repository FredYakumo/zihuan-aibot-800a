#include "config.h"
#include "utils.h"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

namespace fs = boost::filesystem;

void Config::init() {
    auto &config = Config::instance();

    if (fs::exists("config.yaml")) {
        spdlog::info("config.yaml found, loading...");
        const YAML::Node node = YAML::LoadFile("config.yaml");

        if (node["llm_api_url"]) {
            config.llm_api_url = node["llm_api_url"].as<std::string>();
            spdlog::info("LLM_API_URL: {}", config.llm_api_url);
        }
        if (node["llm_api_token"]) {
            config.llm_api_token = node["llm_api_token"].as<std::string>();
            spdlog::info("LLM_API_TOKEN: {}", config.llm_api_token);
        }

        if (node["llm_model_name"]) {
            config.llm_model_name = node["llm_model_name"].as<std::string>();
            spdlog::info("LLM_MODEL_NAME: {}", config.llm_model_name);
        }

        if (node["llm_deep_think_model_name"]) {
            config.llm_deep_think_model_name = node["llm_deep_think_model_name"].as<std::string>();
            spdlog::info("LLM_DEEP_THINK_MODEL_NAME: {}", config.llm_deep_think_model_name);
        }

        if (node["custom_system_prompt"]) {
            config.custom_system_prompt = node["custom_system_prompt"].as<std::string>();
            spdlog::info("CUSTOM_SYSTEM_PROMPT: {}", config.custom_system_prompt);
        }

        if (node["custom_deep_think_system_prompt"]) {
            config.custom_deep_think_system_prompt_option = node["custom_deep_think_system_prompt"].as<std::string>();
            spdlog::info("CUSTOM_DEEP_THINK_SYSTEM_PROMPT: {}", config.custom_deep_think_system_prompt_option.value());
        }

        if (node["net_search_api_url"]) {
            config.net_search_api_url = node["net_search_api_url"].as<std::string>();
            spdlog::info("NET_SEARCH_API_URL: {}", config.net_search_api_url);
        }

        if (node["net_search_token"]) {
            config.net_search_token = node["net_search_token"].as<std::string>();
            spdlog::info("NET_SEARCH_TOKEN: {}", config.net_search_token);
        }

        if (node["url_search_api_url"]) {
            config.url_search_api_url = node["url_search_api_url"].as<std::string>();
            spdlog::info("URL_SEARCH_API_URL: {}", config.url_search_api_url);
        }

        if (node["url_search_token"]) {
            config.url_search_token = node["url_search_token"].as<std::string>();
            spdlog::info("URL_SEARCH_TOKEN: {}", config.url_search_token);
        }

        if (node["vec_db_url"]) {
            config.vec_db_url = node["vec_db_url"].as<std::string>();
            spdlog::info("VEC_DB_URL: {}", config.vec_db_url);
        }

        if (node["bot_id"]) {
            config.bot_id = node["bot_id"].as<uint64_t>();
            spdlog::info("BOT_ID: {}", config.bot_id);
        }

        if (node["admin_id_list"] && node["admin_id_list"].IsSequence()) {
            for (const auto &id : node["admin_id_list"]) {
                config.admin_id_set.emplace(id.as<std::string>());
                spdlog::info("admin id: {}", id.as<std::string>());
            }
        }

        if (node["banned_id_list"] && node["banned_id_list"].IsSequence()) {
            for (const auto &id : node["banned_id_list"]) {
                config.banned_id_set.emplace(id.as<std::string>());
                spdlog::info("banned id: {}", id.as<std::string>());
            }
        }

        if (node["database_host"]) {
            config.database_host = node["database_host"].as<std::string>();
            spdlog::info("database host: {}", config.database_host);
        }
        if (node["database_port"]) {
            config.database_port = node["database_port"].as<uint16_t>();
            spdlog::info("database port: {}", config.database_port);
        }
        if (node["database_user"]) {
            config.database_user = node["database_user"].as<std::string>();
            spdlog::info("database user: {}", config.database_user);
        }
        if (node["database_password"]) {
            config.database_password = node["database_password"].as<std::string>();
        }
    }

    if (const auto api_url = std::getenv("AIBOT_LLM_API_URL")) {
        config.llm_api_url = std::string(api_url);
        spdlog::info("LLM_API_URL: {} (from env)", config.llm_api_url);
    }

    if (const auto api_token = std::getenv("AIBOT_LLM_API_TOKEN")) {
        config.llm_api_token = std::string(api_token);
    }

    if (const auto net_search_api_url = std::getenv("AIBOT_NET_SEARCH_API_URL")) {
        config.net_search_api_url = std::string(net_search_api_url);
        spdlog::info("NET_SEARCH_API_URL: {} (from env)", config.net_search_api_url);
    }

    if (const auto net_search_token = std::getenv("AIBOT_NET_SEARCH_TOKEN")) {
        config.net_search_token = std::string(net_search_token);
        spdlog::info("NET_SEARCH_TOKEN: {} (from env)", config.net_search_token);
    }

    if (const auto url_search_api_url = std::getenv("AIBOT_URL_SEARCH_API_URL")) {
        config.url_search_api_url = std::string(url_search_api_url);
        spdlog::info("URL_SEARCH_API_URL: {} (from env)", config.url_search_api_url);
    }

    if (const auto url_search_token = std::getenv("AIBOT_URL_SEARCH_TOKEN")) {
        config.url_search_token = std::string(url_search_token);
        spdlog::info("URL_SEARCH_TOKEN: {} (from env)", config.url_search_token);
    }

    if (const auto model_name = std::getenv("AIBOT_LLM_MODEL_NAME")) {
        config.llm_model_name = std::string(model_name);
        spdlog::info("LLM_MODEL_NAME: {} (from env)", config.llm_model_name);
    }

    if (const auto deep_think_model_name = std::getenv("AIBOT_LLM_DEEP_THINK_MODEL_NAME")) {
        config.llm_deep_think_model_name = std::string(deep_think_model_name);
        spdlog::info("LLM_DEEP_THINK_MODEL_NAME: {} (from env)", config.llm_deep_think_model_name);
    }

    if (const auto custom_system_prompt = std::getenv("AIBOT_CUSTOM_SYSTEM_PROMPT")) {
        config.custom_system_prompt = std::string(custom_system_prompt);
        spdlog::info("CUSTOM_SYSTEM_PROMPT: {} (from env)", config.custom_system_prompt);
    }

    const auto custom_deep_think_system_prompt_option = std::getenv("AIBOT_CUSTOM_DEEP_THINK_SYSTEM_PROMPT");
    if (custom_deep_think_system_prompt_option != nullptr) {
        config.custom_deep_think_system_prompt_option = std::string(custom_deep_think_system_prompt_option);
        spdlog::info("CUSTOM_DEEP_THINK_SYSTEM_PROMPT: {} (from env)",
                     config.custom_deep_think_system_prompt_option.value());
    } else {
        config.custom_deep_think_system_prompt_option = std::nullopt;
    }

    const auto db_url = std::getenv("AIBOT_MSG_DB_URL");
    if (db_url != nullptr) {
        config.vec_db_url = std::string(db_url);
        spdlog::info("MSG_DB_URL: {} (from env)", config.vec_db_url);
    }

    const auto database_host = std::getenv("DATABASE_HOST");
    if (database_host != nullptr) {
        config.database_host = std::string(database_host);
        spdlog::info("DATABASE_HOST: {} (from env)", config.database_host);
    }
    
    // 读取 DATABASE_PORT，并转换为无符号长整型
    const auto database_port = std::getenv("DATABASE_PORT");
    if (database_port != nullptr) {
        try {
            config.database_port = std::stoull(database_port);
            spdlog::info("DATABASE_PORT: {} (from env)", config.database_port);
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("Invalid DATABASE_PORT format: {}", e.what()));
        }
    }

    // 读取 DATABASE_USER
    const auto database_user = std::getenv("DATABASE_USER");
    if (database_user != nullptr) {
        config.database_user = std::string(database_user);
        spdlog::info("DATABASE_USER: {} (from env)", config.database_user);
    }

    // 读取 DATABASE_PASSWORD
    const auto database_password = std::getenv("DATABASE_PASSWORD");
    if (database_password != nullptr) {
        config.database_password = std::string(database_password);
        // 为了安全起见，不直接在日志中打印密码
        spdlog::info("DATABASE_PASSWORD is set (from env)");
    }

    const auto bot_id_str = std::getenv("AIBOT_BOT_ID");
    if (bot_id_str != nullptr) {
        try {
            config.bot_id = std::stoull(bot_id_str);
            spdlog::info("BOT_ID: {} (from env)", config.bot_id);
        } catch (const std::exception &e) {
            throw std::runtime_error(fmt::format("Invalid BOT_ID format: {}", e.what()));
        }
    }

    const auto admin_id_list = std::getenv("AIBOT_ADMIN_ID_LIST");
    if (admin_id_list != nullptr) {
        spdlog::info("AIBOT_ADMIN_ID_LIST: {} (from env)", admin_id_list);
        for (auto e : SplitString(admin_id_list, ',')) {
            auto admin = std::string(ltrim(rtrim(e)));
            spdlog::info("admin id: {} (from env)", admin);
            config.admin_id_set.emplace(admin);
        }
    }

    const auto banned_id_list = std::getenv("AIBOT_BANNED_ID_LIST");
    if (banned_id_list != nullptr) {
        spdlog::info("AIBOT_BANNED_ID_LIST: {} (from env)", banned_id_list);
        for (auto e : SplitString(banned_id_list, ',')) {
            auto ban = std::string(ltrim(rtrim(e)));
            spdlog::info("banned id: {} (from env)", ban);
            config.banned_id_set.emplace(ban);
        }
    }
}