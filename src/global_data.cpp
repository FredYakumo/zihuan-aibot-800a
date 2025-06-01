#include "global_data.h"
#include "mutex_data.hpp"
#include <optional>
#include <set>

/**
 * @brief A global data structure for managing user chat sessions.
 *
 * This is a thread-safe container that pairs a mutex with an unordered map.
 * The unordered map uses a 64-bit unsigned integer as the key (e.g., user ID)
 * and a ChatSession object as the value to represent the chat sessions of each user.
 * The mutex ensures synchronized access to the map in a multithreaded environment.
 */
MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;

/**
 * @brief A global data structure for managing knowledge items in user chat sessions.
 *
 * This is a thread-safe container that pairs a mutex with an unordered map.
 * The unordered map uses a 64-bit unsigned integer as the key (e.g., user ID)
 * and a vector of DBKnowledge objects as the value to represent the knowledge items
 * associated with each user's chat session.
 * The mutex ensures synchronized access to the map in a multithreaded environment.
 */
MutexData<std::unordered_map<uint64_t, std::set<std::string>>> g_chat_session_knowledge_list_map;

/**
 * @brief A global data structure for managing chat processing states.
 *
 * This is a thread-safe container that pairs a mutex with an unordered map.
 * The unordered map uses a 64-bit unsigned integer as the key (e.g., chat ID)
 * and a boolean as the value to indicate whether the chat is currently being processed.
 * The mutex ensures synchronized access to the map in a multithreaded environment.
 */
std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;

/**
 * @brief A global data structure for managing knowledge items to be added.
 *
 * This is a thread-safe container that pairs a mutex with a vector.
 * The vector contains DBKnowledge objects representing knowledge items
 * that are waiting to be added to the database or system.
 * The mutex ensures synchronized access to the vector in a multithreaded environment.
 */
MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

/**
 * @brief A global data structure for managing bot-sent messages in group chats.
 *
 * This is a thread-safe container that pairs a mutex with an unordered map.
 * The unordered map uses a 64-bit unsigned integer as the key (e.g., group ID)
 * and a ChatSession object as the value to represent the messages sent by the bot
 * in each group chat.
 * The mutex ensures synchronized access to the map in a multithreaded environment.
 */
MutexData<std::unordered_map<uint64_t, ChatSession>> g_group_chat_bot_send_msg;

/**
 * @brief A thread-safe nested map structure for storing group message properties.
 *
 * This is a two-level unordered_map where:
 * - The outer map uses group number (uint64_t) as keys
 * - Each group ID maps to a MutexData-protected inner map
 * - The inner map uses message IDs (uint64_t) as keys and stores MessageProperties
 *
 * The structure allows thread-safe access and modification of message properties
 * for messages within different groups.
 */
 std::unordered_map<uint64_t, MutexData<std::unordered_map<uint64_t, MessageProperties>>> group_message_storage;

 /**
  * @brief A thread-safe nested map structure for storing friend message properties.
  *
  * Similar to group_message_storage, but designed for one-to-one friend messages:
  * - The outer map uses friend qq number (uint64_t) as keys
  * - Each user ID maps to a MutexData-protected inner map
  * - The inner map uses message IDs (uint64_t) as keys and stores MessageProperties
  *
  * This structure provides thread-safe access to message properties in friend chats.
  */
 std::unordered_map<uint64_t, MutexData<std::unordered_map<uint64_t, MessageProperties>>> friend_message_storage;


std::optional<ChatSession> get_user_chat_session(uint64_t chat_session) {
    auto chat_session_map = g_chat_session_map.read();
    auto it = chat_session_map->find(chat_session);
    if (it != chat_session_map->cend()) {
        return it->second;
    }
    return std::nullopt;
}