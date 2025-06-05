#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "chat_session.hpp"
#include "db_knowledge.hpp"
#include "msg_prop.h"
#include "mutex_data.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr size_t USER_SESSION_MSG_LIMIT = 60000;
constexpr size_t MAX_KNOWLEDGE_LENGTH = 4096;

/// 用户/user chat session map
/// key = user QQ号
extern MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;
/// 用户/user chat knowledge map
/// key = user QQ号
extern MutexData<std::unordered_map<uint64_t, std::set<std::string>>> g_chat_session_knowledge_list_map;
/// 用户/user chat is processing lock map
/// key = user QQ号
extern std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;

extern MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

/**
 * @class IndividualMessageIdStorage
 * @brief Thread-safe hierarchical storage for message metadata with group-based access control
 */
class IndividualMessageIdStorage {
  private:
    mutable std::shared_mutex m_mutex; /**< Synchronization primitive for concurrent access */
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, MessageProperties>>
        message_storage; /**< Hierarchical storage structure organizing messages by groups */

  public:
    /**
     * Stores message properties with atomic write semantics
     * @note Overwrites existing messages with same identifiers
     */
    void insert_message(uint64_t group_id, uint64_t message_id, MessageProperties msg_prop) {
        std::unique_lock lock(m_mutex);
        message_storage[group_id][message_id] = std::move(msg_prop);
    }

    /**
     * Retrieves message properties if both group and message exist
     * @note Uses shared locking for concurrent read access
     */
    std::optional<MessageProperties> get_message(uint64_t individual_id, uint64_t message_id) const {
        std::shared_lock lock(m_mutex);
        auto group_it = message_storage.find(individual_id);
        if (group_it == message_storage.end()) {
            return std::nullopt;
        }
        auto msg_it = group_it->second.find(message_id);
        if (msg_it == group_it->second.end()) {
            return std::nullopt;
        }
        return msg_it->second;
    }

    /**
     * Removes specific message while maintaining atomic consistency
     * @returns Whether the message was found and removed
     */
    bool remove_message(uint64_t group_id, uint64_t message_id) {
        std::unique_lock lock(m_mutex);
        auto group_it = message_storage.find(group_id);
        if (group_it == message_storage.end()) {
            return false;
        }
        return group_it->second.erase(message_id) > 0;
    }

    /**
     * Provides efficient read-only access to all messages in a group
     * @returns Pointer to message collection if group exists
     */
    std::optional<const std::unordered_map<uint64_t, MessageProperties> *>
    get_individual_messages(uint64_t individual_id) const {
        std::shared_lock lock(m_mutex);
        auto it = message_storage.find(individual_id);
        if (it == message_storage.cend()) {
            return std::nullopt;
        }
        return &(it->second);
    }

    /**
     * Clears all messages associated with a specific group
     * @returns Whether the group existed and was cleared
     */
    bool clear_individual_messages(uint64_t individual_id) {
        std::unique_lock lock(m_mutex);
        return message_storage.erase(individual_id) > 0;
    }
};

/// 群聊/group chat msg storage
extern IndividualMessageIdStorage g_group_message_storage;
/// 私聊/friend chat msg storage
extern IndividualMessageIdStorage g_friend_message_storage;

#endif