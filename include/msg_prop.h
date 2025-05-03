#ifndef MSG_PROP_H
#define MSG_PROP_H

#include "adapter_event.h"
#include "adapter_model.h"
#include <chrono>
#include <memory>
#include <string>
#include <fmt/format.h>
#include <string_view>

struct MessageProperties {
    bool is_at_me = false;
    std::shared_ptr<std::string> ref_msg_content = nullptr;
    std::shared_ptr<std::string> plain_content = nullptr;

    MessageProperties() = default;
    MessageProperties(bool is_at_me, std::shared_ptr<std::string> ref_msg_content,
                      std::shared_ptr<std::string> ref_plain_content)
        : is_at_me(is_at_me), ref_msg_content(ref_msg_content), plain_content(ref_plain_content) {}
};

MessageProperties get_msg_prop_from_event(const bot_adapter::MessageEvent &e, const std::string_view bot_name, uint64_t bot_id);

void msg_storage(const MessageProperties &msg_prop, const bot_adapter::Sender &sender, const std::chrono::system_clock::time_point &send_time);

#endif