#include "config.h"
#include "think_image_manager.h"
#include "utils.h"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

namespace fs = boost::filesystem;

using namespace wheel;

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
template <typename T> std::optional<T> get_var_from_env_as(const std::string_view env_name) {
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
void load_yaml_config(const YAML::Node &node, const std::string &key, T &target, const std::string &log_prefix = "") {
    if (node[key]) {
        target = node[key].as<T>();
        if constexpr (std::is_same_v<T, std::string> || std::is_arithmetic_v<T>) {
            spdlog::info("{}{}: {}", log_prefix, key, target);
        }
    }
}

// 从环境变量加载字符串配置项
void load_env_config_str(const std::string &env_var, std::string &target) {
    if (auto val = get_var_from_env(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, target);
    }
}

// 从环境变量加载数值配置项
template <typename T> void load_env_config_num(const std::string &env_var, T &target) {
    if (auto val = get_var_from_env_as<T>(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, target);
    }
}

// 加载列表配置项（YAML和环境变量）
void load_list_config(const YAML::Node &node, const std::string &yaml_key, const std::string &env_var,
                      std::unordered_set<std::string> &target_set) {
    // 从YAML加载
    if (node[yaml_key] && node[yaml_key].IsSequence()) {
        for (const auto &id : node[yaml_key]) {
            std::string id_str = id.as<std::string>();
            target_set.insert(id_str);
            spdlog::info("{}: {}", yaml_key, id_str);
        }
    }
    // 从环境变量加载
    if (auto val = get_var_from_env(env_var)) {
        for (auto &&e : SplitString(*val, ',')) {
            std::string trimmed{ltrim(rtrim(e))};
            target_set.insert(trimmed);
            spdlog::info("[ENV] Parsed {}: {}", env_var, trimmed);
        }
    }
}

// 加载可选字符串配置项（如optional<string>）
void load_optional_config(const YAML::Node &node, const std::string &yaml_key, const std::string &env_var,
                          std::optional<std::string> &target) {
    if (node[yaml_key]) {
        target = node[yaml_key].as<std::string>();
        spdlog::info("{}: {}", yaml_key, *target);
    }
    if (auto val = get_var_from_env(env_var)) {
        target = *val;
        spdlog::info("[ENV] Set {}: {}", env_var, *target);
    }
}


void load_vector_config(const YAML::Node &node, const std::string &yaml_key, const std::string &env_var,
                      std::vector<std::string> &target_vector) {

    if (node[yaml_key] && node[yaml_key].IsSequence()) {

        target_vector.clear();
        
        for (const auto &item : node[yaml_key]) {
            std::string item_str = item.as<std::string>();
            target_vector.push_back(item_str);
            spdlog::info("{}: {}", yaml_key, item_str);
        }
    }

    if (auto val = get_var_from_env(env_var)) {

        target_vector.clear();
        for (auto &&e : SplitString(*val, ',')) {
            std::string trimmed{ltrim(rtrim(e))};
            target_vector.push_back(trimmed);
            spdlog::info("[ENV] Parsed {}: {}", env_var, trimmed);
        }
    }
    

    if (target_vector.empty() && yaml_key == "agent_dict_alt_paths") {
        target_vector = {"res/agent_dict.json", "../res/agent_dict.json", "../../res/agent_dict.json"};
        spdlog::info("Using default values for {}", yaml_key);
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

    load_yaml_config(node, "llm_api_key", config.llm_api_key);
    load_env_config_str("AIBOT_LLM_API_KEY", config.llm_api_key);

    load_yaml_config(node, "search_api_url", config.search_api_url);
    load_env_config_str("AIBOT_SEARCH_API_URL", config.search_api_url);

    load_yaml_config(node, "vec_db_url", config.vec_db_url);
    load_env_config_str("AIBOT_MSG_DB_URL", config.vec_db_url);

    load_yaml_config(node, "database_host", config.database_host);
    load_env_config_str("AIBOT_DATABASE_HOST", config.database_host);

    load_yaml_config(node, "database_user", config.database_user);
    load_env_config_str("AIBOT_DATABASE_USER", config.database_user);

    load_yaml_config(node, "database_password", config.database_password);
    load_env_config_str("AIBOT_DATABASE_PASSWORD", config.database_password);

    load_yaml_config(node, "think_image_url", config.think_image_url);
    load_env_config_str("AIBOT_THINK_IMAGE_URL", config.think_image_url);
    
    load_yaml_config(node, "think_pictures_dir", config.think_pictures_dir);
    load_env_config_str("AIBOT_THINK_PICTURES_DIR", config.think_pictures_dir);

    load_yaml_config(node, "temp_res_path", config.temp_res_path);
    load_env_config_str("AIBOT_TEMP_RES_PATH", config.temp_res_path);

    load_yaml_config(node, "lac_model_path", config.lac_model_path);
    load_env_config_str("AIBOT_LAC_MODEL_PATH", config.lac_model_path);

    load_yaml_config(node, "seg_model_path", config.seg_model_path);
    load_env_config_str("AIBOT_SEG_MODEL_PATH", config.seg_model_path);

    load_yaml_config(node, "rank_model_path", config.rank_model_path);
    load_env_config_str("AIBOT_RANK_MODEL_PATH", config.rank_model_path);

    // bot_id is now provided exclusively via CLI '-l <bot_id>' and is not read from YAML or env.

    load_yaml_config(node, "llm_api_port", config.llm_api_port);
    load_env_config_num("AIBOT_LLM_API_PORT", config.llm_api_port);

    load_yaml_config(node, "vec_db_port", config.vec_db_port);
    load_env_config_num("AIBOT_VEC_DB_PORT", config.vec_db_port);

    load_yaml_config(node, "search_api_port", config.search_api_port);
    load_env_config_num("AIBOT_SEARCH_API_PORT", config.search_api_port);

    load_yaml_config(node, "database_port", config.database_port);
    load_env_config_num<uint16_t>("AIBOT_DATABASE_PORT", config.database_port);

    // 列表类型配置
    load_list_config(node, "admin_id_list", "AIBOT_ADMIN_ID_LIST", config.admin_id_set);
    load_list_config(node, "banned_id_list", "AIBOT_BANNED_ID_LIST", config.banned_id_set);

    // 可选类型配置
    load_optional_config(node, "custom_system_prompt", "AIBOT_CUSTOM_SYSTEM_PROMPT", config.custom_system_prompt_option);
    load_optional_config(node, "custom_deep_think_system_prompt", "AIBOT_CUSTOM_DEEP_THINK_SYSTEM_PROMPT",
                         config.custom_deep_think_system_prompt_option);
                         
    // Initialize think image manager if directory is set
    if (!config.think_pictures_dir.empty()) {
        spdlog::info("Initializing think image manager with directory: {}", config.think_pictures_dir);
        bot_adapter::ThinkImageManager::instance().initialize();
    }
}