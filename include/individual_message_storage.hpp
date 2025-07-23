#pragma once

#include "adapter_message.h"
#include "constant_types.hpp"
#include <algorithm>
#include <general-wheel-cpp/collection/concurrent_unordered_map.hpp>
#include <general-wheel-cpp/collection/concurrent_vector.hpp>
#include <memory>
#include <unordered_set>
#include <vector>

using bot_adapter::MessageStorageEntry;
using MessageIdView = wheel::concurrent_unordered_map<qq_id_t, std::shared_ptr<MessageStorageEntry>>;
// using MessageIdContentEmbeddingView = wheel::concurrent_unordered_map<qq_id_t, std::shared_ptr<std::vector<float>>>;

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
    void add_message(qq_id_t individual_id, qq_id_t message_id, std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        add_message_to_individual_view(individual_id, message_id, msg_entry_ptr);
        add_message_to_individual_time_sequence_view(individual_id, msg_entry_ptr);
    }

    void add_message(qq_id_t individual_id, qq_id_t message_id, MessageStorageEntry msg_entry) {
        add_message(individual_id, message_id, std::make_shared<MessageStorageEntry>(std::move(msg_entry)));
    }

    void batch_add_message(const std::vector<qq_id_t> &individual_ids, const std::vector<qq_id_t> &message_ids,
                           const std::vector<std::shared_ptr<MessageStorageEntry>> &msg_entry_ptrs) {
        batch_add_message_to_individual_view(individual_ids, message_ids, msg_entry_ptrs);
        batch_add_message_to_individual_time_sequence_view(individual_ids, msg_entry_ptrs);
    }

    std::optional<std::reference_wrapper<const MessageStorageEntry>> find_message_id(qq_id_t individual_id,
                                                                                     qq_id_t message_id) const {
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

    std::vector<MessageStorageEntry> get_individual_last_msg_list(qq_id_t individual_id, size_t limit = 5) {
        std::vector<MessageStorageEntry> ret;
        auto time_sequence_group = time_sequence_view.find(individual_id);
        if (time_sequence_group.has_value()) {
            auto &messages = time_sequence_group->get();
            size_t size = messages.size();
            size_t start = size > limit ? size - limit : 0;
            ret.reserve(size - start);  // Reserve space for performance
            // Iterate backwards, but insert into the vector from the beginning to maintain order
            for (size_t i = size; i > start; --i) {
                ret.insert(ret.begin(), *messages[i-1]->get());
            }
        }
        return ret;
    }

  private:
    inline void add_message_to_individual_view(qq_id_t individual_id, qq_id_t message_id,
                                               std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        auto individual = message_id_view_map.get_or_emplace_value(individual_id);

        individual->insert_or_assign(message_id, msg_entry_ptr);
    }

    inline void
    batch_add_message_to_individual_view(const std::vector<qq_id_t> &individual_ids,
                                         const std::vector<qq_id_t> &message_ids,
                                         const std::vector<std::shared_ptr<MessageStorageEntry>> &msg_entry_ptrs) {
        if (individual_ids.size() != message_ids.size() || individual_ids.size() != msg_entry_ptrs.size()) {
            throw std::invalid_argument(
                "IndividualMessageStorage::batch_add_message_to_individual_view(): mismatched vector sizes");
        }
        message_id_view_map.modify_map([&](auto &map) {
            for (size_t i = 0; i < individual_ids.size(); ++i) {
                MessageIdView empty_view;
                auto [it, inserted] = map.try_emplace(individual_ids[i], empty_view);
                it->second.insert_or_assign(message_ids[i], msg_entry_ptrs[i]);
            }
        });
    }

    inline void add_message_to_individual_time_sequence_view(qq_id_t individual_id,
                                                             std::shared_ptr<MessageStorageEntry> msg_entry_ptr) {
        auto time_sequence_group = time_sequence_view.get_or_emplace_value(individual_id);

        time_sequence_group->push_back(msg_entry_ptr);
    }

    inline void batch_add_message_to_individual_time_sequence_view(
        const std::vector<qq_id_t> &individual_ids,
        const std::vector<std::shared_ptr<MessageStorageEntry>> &msg_entry_ptrs) {
        if (individual_ids.size() != msg_entry_ptrs.size()) {
            throw std::invalid_argument("IndividualMessageStorage::batch_add_message_to_individual_time_sequence_view()"
                                        ": mismatched vector sizes");
        }
        
        // Track individual_ids that need sorting
        std::unordered_set<qq_id_t> modified_individuals;
        
        // Step 1: Add messages first
        time_sequence_view.modify_map([&](auto &map) {
            for (size_t i = 0; i < individual_ids.size(); ++i) {
                auto [it, inserted] = map.try_emplace(individual_ids[i],
                                                      wheel::concurrent_vector<std::shared_ptr<MessageStorageEntry>>());
                it->second.push_back(msg_entry_ptrs[i]);
                modified_individuals.insert(individual_ids[i]);
            }
        });
        
        // Step 2: Sort each modified individual's messages separately
        // This avoids holding the main map lock during sorting operations
        for (const auto& individual_id : modified_individuals) {
            auto individual_vector = time_sequence_view.find(individual_id);
            if (individual_vector.has_value()) {
                auto& msg_vector = individual_vector->get();
                // Thread-safe access to underlying vector for sorting
                msg_vector.modify_vector([](std::vector<std::shared_ptr<MessageStorageEntry>>& vec) {
                    // Sort by send_time from oldest to newest
                    std::sort(vec.begin(), vec.end(), 
                        [](const std::shared_ptr<MessageStorageEntry>& a, 
                           const std::shared_ptr<MessageStorageEntry>& b) {
                            return a->send_time < b->send_time;
                        });
                });
            }
        }
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
    wheel::concurrent_unordered_map<qq_id_t, MessageIdView> message_id_view_map;

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
    wheel::concurrent_unordered_map<qq_id_t, wheel::concurrent_vector<std::shared_ptr<MessageStorageEntry>>>
        time_sequence_view;
};
