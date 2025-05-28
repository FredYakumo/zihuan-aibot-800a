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
    std::string llm_model_name;
    std::string custom_system_prompt;
    std::optional<std::string> custom_deep_think_system_prompt_option;
    // std::string BOT_NAME;
    // MiraiCP::QQID BOT_QQID;

    std::string database_host;
    uint16_t database_port;
    std::string database_user;
    std::string database_password;
    std::string database_schema;

    std::string vec_db_url;

    std::unordered_set<std::string> admin_id_set;
    std::unordered_set<std::string> banned_id_set;

    uint64_t bot_id;

    std::string search_api_url;

    uint32_t search_api_port;


    uint64_t update_group_info_period_sec = 1800;

    std::string think_image_url;

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