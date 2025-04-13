#ifndef CONFIG_H
#define CONFIG_H

#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

extern std::string LLM_API_URL;
extern std::string LLM_API_TOKEN;
extern std::string LLM_MODEL_NAME;
extern std::string CUSTOM_SYSTEM_PROMPT;
// extern std::string BOT_NAME;
// extern MiraiCP::QQID BOT_QQID;

extern std::string MSG_DB_URL;

extern std::unordered_set<std::string> ADMIN_ID_SET;
extern std::unordered_set<std::string> BANNED_ID_SET;

extern uint64_t BOT_ID;

extern std::string NET_SEARCH_API_URL;
extern std::string NET_SEARCH_TOKEN;

extern std::string URL_SEARCH_API_URL;
extern std::string URL_SEARCH_TOKEN;


void init_config();

inline bool is_admin(const std::string &id) {
    return ADMIN_ID_SET.find(id) != std::cend(ADMIN_ID_SET);
}

inline bool is_admin(uint64_t id) {
    return is_admin(std::to_string(id));
}

inline bool is_banned_id(const std::string &id) {
    return BANNED_ID_SET.find(id) != std::cend(BANNED_ID_SET);
}

inline bool is_banned_id(uint64_t id) {
    return is_banned_id(std::to_string(id));
}

#endif