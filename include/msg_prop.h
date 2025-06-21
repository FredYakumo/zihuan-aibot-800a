#ifndef MSG_PROP_H
#define MSG_PROP_H

#include "adapter_event.h"
#include "adapter_model.h"
#include "constants.hpp"
#include <chrono>
#include <fmt/format.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>

struct MessageProperties {
    bool is_at_me = false;
    std::shared_ptr<std::string> ref_msg_content = nullptr;
    std::shared_ptr<std::string> plain_content = nullptr;
    std::set<uint64_t> at_id_set;

    MessageProperties() = default;
    MessageProperties(bool is_at_me, std::shared_ptr<std::string> ref_msg_content,
                      std::shared_ptr<std::string> ref_plain_content, std::set<uint64_t> at_id_set)
        : is_at_me(is_at_me), ref_msg_content(ref_msg_content), plain_content(ref_plain_content),
          at_id_set(std::move(at_id_set)) {}
};

/**
 * @brief Extracts and processes message properties from a bot adapter message event for LLM consumption
 *
 * This function parses a message event and extracts key properties that are relevant for
 * LLM (Language Learning Model) processing. It handles various message types including
 * plain text, quote messages, and @mentions, creating a structured representation that
 * the LLM can use to understand the context and intent of the user's message.
 *
 * The function processes the following message components:
 * - Plain text content (cleaned and trimmed)
 * - Quote/reference messages (formatted as "引用了一段消息文本: \"...\"")
 * - @mentions (both by name and QQ ID/QQ号)
 * - Bot-specific @mentions (detects when the bot is being addressed)
 *
 * For @mentions, the function:
 * - Removes @mentions from plain text content
 * - Tracks all mentioned user IDs/用户ID in at_id_set
 * - Sets is_at_me flag when the bot is mentioned
 * - Handles both name-based (@botname) and ID-based (@123456) mentions
 *
 * @param e The message event containing the raw message data
 * @param bot_name The name of the bot (used to detect @mentions)
 * @param bot_id The QQ ID/QQ号 of the bot (used to detect @mentions)
 * @return MessageProperties A structured representation of the message suitable for LLM processing
 *
 * @note Empty messages are tagged with EMPTY_MSG_TAG for consistent processing
 * @note The function ensures thread safety and handles null message components gracefully
 * @see MessageProperties
 * @see bot_adapter::MessageEvent
 */


MessageProperties get_msg_prop_from_event(const bot_adapter::MessageEvent &event, const std::string_view bot_name,
                                          uint64_t bot_id);

void store_msg_prop_to_db(const MessageProperties &msg_prop, const bot_adapter::Sender &sender,
                 const std::chrono::system_clock::time_point &send_time,
                 const std::optional<std::set<uint64_t>> specify_at_target_set = std::nullopt);

/**
 * @brief Retrieves the list of message records associated with a specific user from the global chat session.
 *
 * The function fetches all message records for the given user and formats them as strings
 * in the pattern: "Username: \"Message content\"", with the quotation marks in the user information
 * converted to escaped quotation marks (i.e., `\"`).
 *
 * @param sender_name The name of the sender (user).
 * @param sender_id The unique identifier for the sender.
 * @return std::vector<std::string> A vector containing formatted message records for the user.
 */
std::vector<std::string> get_message_list_from_chat_session(const std::string_view sender_name, qq_id_t sender_id);

#endif