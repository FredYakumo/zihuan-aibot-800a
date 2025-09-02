#ifndef EVENT_H
#define EVENT_H

#include "bot_adapter.h"
#include "global_data.h"
#include "constant_types.hpp"

void register_event(bot_adapter::BotAdapter &adapter);

/**
 * @brief Tries to mark a user as being processed for a chat reply.
 * 
 * This function checks if a user is currently being processed for a chat reply.
 * If not, it marks the user as being processed and returns true.
 * If the user is already being processed, it returns false.
 * This prevents multiple concurrent processing of the same user's messages.
 *
 * @param target_id The QQ ID of the target user to check and mark.
 * @return bool Returns true if the user can be processed (was not being processed or was released),
 *         false if the user is already being processed.
 */
inline bool try_to_replay_person(uint64_t target_id) {
    if (auto v = g_chat_processing_map.find(target_id); v.has_value()) {
        if (*v) {
            return false;
        } else {
            v->get() = true;
            return true;
        }
    }
    g_chat_processing_map.insert_or_assign(target_id, true);
    return true;
}

/**
 * @brief Releases a user from being marked as processed for chat replies.
 * 
 * This function marks a user as no longer being processed, allowing
 * future chat messages from this user to be processed again.
 * Should be called after processing of a user's message is complete.
 *
 * @param id The QQ ID of the user to release from processing.
 */
inline void release_processing_replay_person(qq_id_t id) {
    if (auto v = g_chat_processing_map.find(id); v.has_value()) {
        v->get() = false;
    }
}

#endif