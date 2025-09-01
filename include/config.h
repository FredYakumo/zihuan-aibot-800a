#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

struct Config {
    Config &operator=(const Config &) = delete;

    std::string llm_api_url;
    uint32_t llm_api_port;

    /// The natural language text generator model name
    /// In 紫幻(zihuan), we use `DeepSeekR1` for natural language generation
    std::string llm_model_name;

    std::string llm_api_key;

    


    /// Use this extra system prompt when set.
    std::optional<std::string> custom_system_prompt_option;

    /// Use this extra system prompt in deep think mode when set.
    std::optional<std::string> custom_deep_think_system_prompt_option;

    /// MySQL database host
    std::string database_host;
    /// MySQL database port
    uint16_t database_port;
    /// MySQL database user (DEFAULT is root)
    std::string database_user;
    /// MySQL database password
    std::string database_password;
    /// MySQL database SCHEMA (DEFAULT is AIBot_800a)
    std::string database_schema;

    /// Vector database url. AIBot800a now use Weaviate Vector Database.
    std::string vec_db_url;

    uint32_t vec_db_port = 28080;

    /// Some bot command required administrator privile,
    /// user that QQ号/ID in this set will be treat as administrator.
    std::unordered_set<std::string> admin_id_set;

    /// Bot won't reply user which QQ号/ID in this set
    std::unordered_set<std::string> banned_id_set;


    /// Url of search API
    std::string search_api_url;

    /// Port of search API
    uint32_t search_api_port;


    uint64_t update_group_info_period_sec = 1800;

    /// A picture URL used when bot is thinking, this picture will be sent to user
    std::string think_image_url;

    /// A location to store temporary results, such as rendered html files.
    std::string temp_res_path;

    /// Path to LAC model
    std::string lac_model_path;
    /// Path to segmentation model
    std::string seg_model_path;
    /// Path to rank model
    std::string rank_model_path;

    inline static Config &instance() {
        static Config config;
        return config;
    }

    static void init();

  private:
    Config() = default;
};

// Helper functions using the singleton instance
inline bool is_admin(const std::string &id) {
    return Config::instance().admin_id_set.find(id) != std::cend(Config::instance().admin_id_set);
}

inline bool is_admin(uint64_t id) { return is_admin(std::to_string(id)); }

inline bool is_banned_id(const std::string &id) {
    return Config::instance().banned_id_set.find(id) != std::cend(Config::instance().banned_id_set);
}

inline bool is_banned_id(uint64_t id) { return is_banned_id(std::to_string(id)); }

#endif