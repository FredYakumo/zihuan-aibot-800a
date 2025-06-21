#pragma once

#include "adapter_message.h"
#include <cstdint>
#include <general-wheel-cpp/collection/concurrent_unordered_map.hpp>
#include <general-wheel-cpp/collection/concurrent_vector.hpp>
#include <memory>

using bot_adapter::MessageStorageEntry;
using MessageIdView = wheel::concurrent_unordered_map<uint64_t, std::shared_ptr<MessageStorageEntry>>;

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
class IndividualMessageStorage {
  public:
    void add_message(uint64_t individual_id, uint64_t message_id, std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
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
        auto individual = message_id_view_map.get_or_emplace_value(individual_id);

        individual->insert_or_assign(message_id, msg_entry_ptr);
    }

    inline void add_message_to_individual_time_sequence_view(uint64_t individual_id,
                                                             std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        auto time_sequence_group = time_sequence_view.get_or_emplace_value(
            individual_id);

        time_sequence_group->push_back(msg_entry_ptr);
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
    wheel::concurrent_unordered_map<uint64_t, MessageIdView> message_id_view_map;

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
    wheel::concurrent_unordered_map<uint64_t, wheel::concurrent_vector<std::shared_ptr<MessageStorageEntry>>>
        time_sequence_view;
};