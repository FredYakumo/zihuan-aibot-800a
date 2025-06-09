#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "chat_session.hpp"
#include "concurrent_unordered_map.hpp"
#include "concurrent_vector.hpp"
#include "db_knowledge.hpp"
#include "msg_prop.h"
#include "mutex_data.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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
    mutable std::shared_mutex m_mutex; /**< add individual mutex */
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, MessageProperties>>
        message_storage; /**< Hierarchical storage structure organizing messages by groups */

  public:
    /**
     * Stores message properties with atomic write semantics
     * @note Overwrites existing messages with same identifiers
     */
    void insert_message(uint64_t individual_id, uint64_t message_id, MessageProperties msg_prop) {
        std::unique_lock lock(m_mutex);
        message_storage[individual_id][message_id] = std::move(msg_prop);
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
    bool remove_message(uint64_t individual_id, uint64_t message_id) {
        std::unique_lock lock(m_mutex);
        auto group_it = message_storage.find(individual_id);
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

struct MessageStorageEntry {
    uint64_t message_id;
    std::string sender_name;
    uint64_t sender_id;
    std::chrono::system_clock::time_point send_time;
    std::shared_ptr<MessageProperties> msg_prop;
};

/**
 * @brief Concurrent map for storing message entries indexed by message ID
 *
 * Thread-safe unordered map that maintains message storage entries using a lock-free
 * implementation. Provides O(1) average case lookup complexity.
 *
 * @see MessageStorageEntry
 * @note Shares message objects through
 *       shared_ptr to ensure data consistency. The same physical message
 *       may be accessed through either structure.
 */
using MessageIdView = concurrent_unordered_map<uint64_t, std::shared_ptr<MessageStorageEntry>>;

class IndividualMessageStorage {
  public:
    void add_message(uint64_t individual_id, uint64_t message_id,
                     std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        add_message_to_individual_view(individual_id, message_id, msg_entry_ptr);
        add_message_to_individual_time_sequence_view(individual_id, msg_entry_ptr);
    }

    void add_message(uint64_t individual_id, uint64_t message_id, MessageStorageEntry msg_entry) {
        add_message(individual_id, message_id, std::make_shared<MessageStorageEntry>(std::move(msg_entry)));
    }

    std::optional<std::reference_wrapper<const MessageStorageEntry>> find_message_id(uint64_t individual_id,
                                                                                     uint64_t message_id) const {
        auto group = message_id_view_map.find(individual_id);
        if (!group.has_value()) {
            return std::nullopt;
        }
        auto msg = group->get().find(message_id);
        if (msg.has_value()) {
            return std::cref(*msg->get());
        }
        return std::nullopt;
    }

    std::vector<std::reference_wrapper<const MessageStorageEntry>> get_individual_last_msg_list(uint64_t individual_id,
                                                                                              size_t limit = 5) {
        std::vector<std::reference_wrapper<const MessageStorageEntry>> ret;
        auto time_sequence_group = time_sequence_view.find(individual_id);
        if (time_sequence_group.has_value()) {
            auto &messages = time_sequence_group->get();
            size_t size = messages.size();
            size_t start = size > limit ? size - limit : 0;
            for (size_t i = start; i < size; i++) {
                ret.push_back(std::cref(*messages[i]->get()));
            }
        }
        return ret;
    }

  private:
    inline void add_message_to_individual_view(uint64_t individual_id, uint64_t message_id,
                                                 std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        // Group dimension
        auto individual = message_id_view_map.find(individual_id);
        if (!individual.has_value()) {
            individual = message_id_view_map.insert_or_assign(individual_id, MessageIdView());
        }

        individual->get().insert_or_assign(message_id, msg_entry_ptr);
    }

    inline void add_message_to_individual_time_sequence_view(uint64_t individual_id,
                                                        std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        concurrent_vector<std::shared_ptr<MessageStorageEntry>> *time_sequence_group_ptr;
        if (auto time_sequence_group = time_sequence_view.find(individual_id); time_sequence_group.has_value()) {
            time_sequence_group_ptr = &time_sequence_group->get();
        } else {
            time_sequence_group_ptr =
                &time_sequence_view
                     .insert_or_assign(individual_id, concurrent_vector<std::shared_ptr<MessageStorageEntry>>())
                     .get();
        }

        time_sequence_group_ptr->push_back(msg_entry_ptr);
    }

    /**
     * @brief Two-dimensional message index organized by individual_id
     * @ingroup MessageStorage
     *
     * Provides O(1) access to messages using individual_id(QQ号/群号) -> message_id lookup.
     *
     * Structure:
     * @code
     * {
     *   individual_id1: {
     *     message_id1: shared_ptr<MessageStorageEntry>,
     *     message_id1: shared_ptr<MessageStorageEntry>,
     *     ...
     *   },
     *   ...
     * }
     * @endcode
     *
     * @note Uses shared_ptr to maintain memory consistency with
     *       @ref time_sequence_view (same message objects)
     */
    concurrent_unordered_map<uint64_t, MessageIdView> message_id_view_map;

    /**
     * @brief Temporal message sequence organized by group ID
     * @ingroup MessageStorage
     *
     * Maintains messages in chronological order within each group.
     *
     * Structure:
     * @code
     * {
     *   individual_id1: [msg_ptr1, msg_ptr2, ...], // sorted by time
     *   individual_id2: [msg_ptr1, msg_ptr2, ...], // sorted by time
     *   ...
     * }
     * @endcode
     *
     * @note Shares message objects with @ref group_member_view_map through
     *       shared_ptr to ensure data consistency. The same physical message
     *       may be accessed through either structure.
     */
    concurrent_unordered_map<uint64_t, concurrent_vector<std::shared_ptr<MessageStorageEntry>>>
        time_sequence_view;
};

/// 群聊/group chat msg storage
extern IndividualMessageIdStorage g_group_message_storage;
/// 私聊/friend chat msg storage
extern IndividualMessageIdStorage g_friend_message_storage;
/// bot send message to group storage.
extern IndividualMessageIdStorage g_bot_send_group_message_storage;

#endif