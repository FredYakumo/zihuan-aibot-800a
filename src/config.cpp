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

std::optional<std::string> get_var_from_env(const std::string_view env_name) {
    if (const char *value = std::getenv(env_name.data())) {
        spdlog::info("[ENV] Found {} = {}", env_name, value); // 统一日志格式
        return std::string(value);
    }
    spdlog::debug("[ENV] Not found: {}", env_name);
    return std::nullopt;
}

template <typename T> std::optional<T> get_var_from_env_as(const std::string_view env_name) {
    if (auto str_val = get_var_from_env(env_name)) {
        try {
            if constexpr (std::is_same_v<T, uint64_t>) {
                auto value = std::stoull(*str_val);
                spdlog::info("[ENV] Converted {} to uint64_t: {}", env_name, value);
                return value;
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                auto value = static_cast<uint16_t>(std::stoul(*str_val));
                spdlog::info("[ENV] Converted {} to uint16_t: {}", env_name, value);
                return value;
            }
        } catch (const std::exception &e) {
            spdlog::error("[ENV] Convert {} failed: {}", env_name, e.what());
        }
    }
    return std::nullopt;
}

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

        if (node["think_image_url"]) {
            config.think_image_url = node["think_image_url"].as<std::string>();
            spdlog::info("think image url: {}", config.think_image_url);
        }
    }

    // 环境变量加载部分
    if (auto val = get_var_from_env("AIBOT_LLM_API_URL"))
        config.llm_api_url = *val;
    if (auto val = get_var_from_env("AIBOT_LLM_API_TOKEN"))
        config.llm_api_token = *val;
    if (auto val = get_var_from_env("AIBOT_NET_SEARCH_API_URL"))
        config.net_search_api_url = *val;
    if (auto val = get_var_from_env("AIBOT_NET_SEARCH_TOKEN"))
        config.net_search_token = *val;
    if (auto val = get_var_from_env("AIBOT_URL_SEARCH_API_URL"))
        config.url_search_api_url = *val;
    if (auto val = get_var_from_env("AIBOT_URL_SEARCH_TOKEN"))
        config.url_search_token = *val;
    if (auto val = get_var_from_env("AIBOT_LLM_MODEL_NAME"))
        config.llm_model_name = *val;
    if (auto val = get_var_from_env("AIBOT_LLM_DEEP_THINK_MODEL_NAME"))
        config.llm_deep_think_model_name = *val;
    if (auto val = get_var_from_env("AIBOT_CUSTOM_SYSTEM_PROMPT"))
        config.custom_system_prompt = *val;
    if (auto val = get_var_from_env("AIBOT_CUSTOM_DEEP_THINK_SYSTEM_PROMPT"))
        config.custom_deep_think_system_prompt_option = *val;
    if (auto val = get_var_from_env("AIBOT_MSG_DB_URL"))
        config.vec_db_url = *val;
    if (auto val = get_var_from_env("DATABASE_HOST"))
        config.database_host = *val;
    if (auto val = get_var_from_env("DATABASE_USER"))
        config.database_user = *val;
    if (auto val = get_var_from_env("DATABASE_PASSWORD"))
        config.database_password = *val;
    if (auto val = get_var_from_env("THINK_IMAGE_URL"))
        config.think_image_url = *val;

    // 数值类型转换
    if (auto val = get_var_from_env_as<uint64_t>("AIBOT_BOT_ID"))
        config.bot_id = *val;
    if (auto val = get_var_from_env_as<uint16_t>("DATABASE_PORT"))
        config.database_port = *val;

    // 列表处理
    if (auto val = get_var_from_env("AIBOT_ADMIN_ID_LIST")) {
        for (auto &&e : SplitString(*val, ',')) {
            auto admin = std::string(ltrim(rtrim(e)));
            spdlog::info("[ENV] Admin ID parsed: {}", admin);
            config.admin_id_set.emplace(admin);
        }
    }

    if (auto val = get_var_from_env("AIBOT_BANNED_ID_LIST")) {
        for (auto &&e : SplitString(*val, ',')) {
            auto ban = std::string(ltrim(rtrim(e)));
            spdlog::info("[ENV] Banned ID parsed: {}", ban);
            config.banned_id_set.emplace(ban);
        }
    }
}