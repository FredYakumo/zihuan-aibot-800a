#ifndef MSG_PROP_H
#define MSG_PROP_H

#include <string>
#include <MiraiCP.hpp>
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

MessageProperties get_msg_prop_from_event(const MiraiCP::GroupMessageEvent &e, const std::string_view bot_name);

void msg_storage(const MessageProperties &msg_prop, MiraiCP::QQID group_id, MiraiCP::QQID sender_id,
                 const std::string_view sender_name, const std::string_view group_name);

#endif