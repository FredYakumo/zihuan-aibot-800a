#ifndef LLM_H
#define LLM_H
#include "bot_cmd.h"
#include "global_data.h"

void process_llm(bot_cmd::CommandContext context);

inline bool try_begin_processing_llm(MiraiCP::QQID id) {
    std::lock_guard lock(g_chat_processing_map.first);
    if (auto it = g_chat_processing_map.second.find(id); it != std::cend(g_chat_processing_map.second)) {
        if (it->second) {
            return false;
        }
        it->second = true;
        return true;
    }
    g_chat_processing_map.second.emplace(id, true);
    return true;
}

/**
 * @brief Converts a system prompt and a message list into a JSON format.
 *
 * This function converts a system prompt and a message list into a JSON array.
 * The first element of the array is the system prompt, and the subsequent elements
 * are the messages from the message list. Each message contains two fields: "role" and "content".
 *
 * @param system_prompt The system prompt content, of type std::string_view.
 * @param msg_list The message list, of type std::deque<ChatMessage>.
 * @return nlohmann::json Returns a JSON array containing the system prompt and the message list.
 */
inline nlohmann::json msg_list_to_json(const std::string_view system_prompt, const std::deque<ChatMessage> &msg_list) {
    nlohmann::json msg_json = nlohmann::json::array();
    msg_json.push_back(nlohmann::json{{"role", "system"}, {"content", system_prompt}});
    for (const auto &msg : msg_list) {
        nlohmann::json msg_entry;
        msg_entry["role"] = msg.role;
        msg_entry["content"] = msg.content;
        MiraiCP::Logger::logger.info(msg_entry);
        msg_json.push_back(msg_entry);
    }
    return msg_json;
}

inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, const ChatMessage &msg) {
    msg_json.push_back({{"role", msg.role}, {"content", msg.content}});
    return msg_json;
}

inline nlohmann::json get_msg_json(const std::string_view system_prompt, const MiraiCP::QQID id, const std::string_view name) {
    {
        auto session = g_chat_session_map.read();
        if (auto iter = session->find(id); iter != session->cend()) {
            return msg_list_to_json(system_prompt, iter->second.message_list);
        }
    }
    auto session = g_chat_session_map.write()->insert({id, ChatSession(name)});
    return msg_list_to_json(system_prompt, session.first->second.message_list);
}

inline void release_processing_llm(MiraiCP::QQID id) {
    std::lock_guard lock(g_chat_processing_map.first);
    g_chat_processing_map.second[id] = false;
}

#endif