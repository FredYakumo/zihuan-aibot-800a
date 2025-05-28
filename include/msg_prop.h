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

MessageProperties get_msg_prop_from_event(const bot_adapter::MessageEvent &e, const std::string_view bot_name,
                                          uint64_t bot_id);

void msg_storage(const MessageProperties &msg_prop, const bot_adapter::Sender &sender,
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