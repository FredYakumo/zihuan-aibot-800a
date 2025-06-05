#include "global_data.h"
#include "mutex_data.hpp"
#include <optional>
#include <set>

MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;

MutexData<std::unordered_map<uint64_t, std::set<std::string>>> g_chat_session_knowledge_list_map;

std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;

MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

MutexData<std::unordered_map<uint64_t, ChatSession>> g_group_chat_bot_send_msg;

IndividualMessageIdStorage g_group_message_storage;\

IndividualMessageIdStorage g_friend_message_storage;

/**
 * @brief Retrieves a chat session if it exists
 * @param chat_session User qq id
 * @return Optional containing the session if found
 */
std::optional<ChatSession> get_user_chat_session(uint64_t chat_session) {
    auto chat_session_map = g_chat_session_map.read();
    auto it = chat_session_map->find(chat_session);
    if (it != chat_session_map->cend()) {
        return it->second;
    }
    return std::nullopt;
}