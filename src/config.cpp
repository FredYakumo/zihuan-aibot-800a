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

// 通用函数：从环境变量获取字符串
std::optional<std::string> get_var_from_env(const std::string_view env_name) {
    if (const char *value = std::getenv(env_name.data())) {
        spdlog::info("[ENV] Found {} = {}", env_name, value);
        return std::string(value);
    }
    spdlog::debug("[ENV] Not found: {}", env_name);
    return std::nullopt;
}

// 通用函数：从环境变量转换特定类型
template <typename T>
std::optional<T> get_var_from_env_as(const std::string_view env_name) {
    if (auto str_val = get_var_from_env(env_name)) {
        try {
            if constexpr (std::is_same_v<T, uint64_t>) {
                return std::stoull(*str_val);
            } else if constexpr (std::is_same_v<T, uint16_t>) {
                return static_cast<uint16_t>(std::stoul(*str_val));
            }
        } catch (const std::exception &e) {
            spdlog::error("[ENV] Convert {} failed: {}", env_name, e.what());
        }
    }
    return std::nullopt;
}

// 从YAML加载配置项
template <typename T>
void load_yaml_config(const YAML::Node& node, const std::string& key, T& target, const std::string& log_prefix = "") {
    if (node[key]) {
        target = node[key].as<T>();
        if constexpr (std::is_same_v<T, std::string> || std::is_arithmetic_v<T>) {
            spdlog::info("{}{}: {}", log_prefix, key, target);
        }
    }
}

// 从环境变量加载字符串配置项
void load_env_config_str(const std::string& env_var, std::string& target) {
    if (auto val = get_var_from_env(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, target);
    }
}

// 从环境变量加载数值配置项
template <typename T>
void load_env_config_num(const std::string& env_var, T& target) {
    if (auto val = get_var_from_env_as<T>(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, target);
    }
}

// 加载列表配置项（YAML和环境变量）
void load_list_config(const YAML::Node& node, const std::string& yaml_key, 
                     const std::string& env_var, std::unordered_set<std::string>& target_set) {
    // 从YAML加载
    if (node[yaml_key] && node[yaml_key].IsSequence()) {
        for (const auto& id : node[yaml_key]) {
            std::string id_str = id.as<std::string>();
            target_set.insert(id_str);
            spdlog::info("{}: {}", yaml_key, id_str);
        }
    }
    // 从环境变量加载
    if (auto val = get_var_from_env(env_var)) {
        for (auto&& e : SplitString(*val, ',')) {
            std::string trimmed {ltrim(rtrim(e))};
            target_set.insert(trimmed);
            spdlog::info("[ENV] Parsed {}: {}", env_var, trimmed);
        }
    }
}

// 加载可选字符串配置项（如optional<string>）
void load_optional_config(const YAML::Node& node, const std::string& yaml_key,
                         const std::string& env_var, std::optional<std::string>& target) {
    if (node[yaml_key]) {
        target = node[yaml_key].as<std::string>();
        spdlog::info("{}: {}", yaml_key, *target);
    }
    if (auto val = get_var_from_env(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, *target);
    }
}

// 主配置初始化函数
void Config::init() {
    auto &config = Config::instance();
    YAML::Node node;

    // 加载YAML文件
    if (fs::exists("config.yaml")) {
        spdlog::info("config.yaml found, loading...");
        node = YAML::LoadFile("config.yaml");
    }

    // 字符串类型配置
    load_yaml_config(node, "llm_api_url", config.llm_api_url);
    load_env_config_str("AIBOT_LLM_API_URL", config.llm_api_url);

    load_yaml_config(node, "llm_model_name", config.llm_model_name);
    load_env_config_str("AIBOT_LLM_MODEL_NAME", config.llm_model_name);


    load_yaml_config(node, "custom_system_prompt", config.custom_system_prompt);
    load_env_config_str("AIBOT_CUSTOM_SYSTEM_PROMPT", config.custom_system_prompt);

    load_yaml_config(node, "search_api_url", config.search_api_url);
    load_env_config_str("AIBOT_SEARCH_API_URL", config.search_api_url);


    load_yaml_config(node, "vec_db_url", config.vec_db_url);
    load_env_config_str("AIBOT_MSG_DB_URL", config.vec_db_url);

    load_yaml_config(node, "database_host", config.database_host);
    load_env_config_str("DATABASE_HOST", config.database_host);

    load_yaml_config(node, "database_user", config.database_user);
    load_env_config_str("DATABASE_USER", config.database_user);

    load_yaml_config(node, "database_password", config.database_password);
    load_env_config_str("DATABASE_PASSWORD", config.database_password);

    load_yaml_config(node, "think_image_url", config.think_image_url);
    load_env_config_str("THINK_IMAGE_URL", config.think_image_url);

    // 数值类型配置
    load_yaml_config(node, "bot_id", config.bot_id);
    load_env_config_num<uint64_t>("AIBOT_BOT_ID", config.bot_id);

    load_yaml_config(node, "llm_api_port", config.llm_api_port);
    load_env_config_num("AIBOT_LLM_API_PORT", config.llm_api_port);

    load_yaml_config(node, "search_api_port", config.search_api_port);
    load_env_config_num("AIBOT_SEARCH_API_PORT", config.search_api_port);

    load_yaml_config(node, "database_port", config.database_port);
    load_env_config_num<uint16_t>("DATABASE_PORT", config.database_port);

    // 列表类型配置
    load_list_config(node, "admin_id_list", "AIBOT_ADMIN_ID_LIST", config.admin_id_set);
    load_list_config(node, "banned_id_list", "AIBOT_BANNED_ID_LIST", config.banned_id_set);

    // 可选类型配置
    load_optional_config(node, "custom_deep_think_system_prompt", 
                        "AIBOT_CUSTOM_DEEP_THINK_SYSTEM_PROMPT", 
                        config.custom_deep_think_system_prompt_option);
}