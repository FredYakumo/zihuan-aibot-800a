#include "msg_prop.h"
#include "constants.hpp"
#include "rag.h"
#include "utils.h"

MessageProperties get_msg_prop_from_event(const MiraiCP::GroupMessageEvent &e, const std::string_view bot_name) {
    MessageProperties ret{};

    for (auto msg : e.message) {
        MiraiCP::Logger::logger.info(std::string("Message Type: ") + std::to_string(msg.getType()) +
                                     ", Content: " + msg->content);

        if (msg.getType() == MiraiCP::SingleMessageType::At_t && msg->content == std::to_string(e.bot.id())) {
            ret.is_at_me = true;
        } else if (msg.getType() == MiraiCP::SingleMessageType::QuoteReply_t) {
            std::string s = msg.get()->toJson()["source"]["originalMessage"].dump();
            MiraiCP::Logger::logger.info(msg.get()->toJson());
            if (ret.ref_msg_content == nullptr) {
                ret.ref_msg_content = std::make_unique<std::string>(s);
            } else {
                *ret.ref_msg_content += s;
            }
        } else if (msg.getType() == MiraiCP::SingleMessageType::PlainText_t) {
            std::string s = msg->content;
            if (ret.plain_content == nullptr) {
                ret.plain_content = std::make_unique<std::string>(s);
            } else {
                *ret.plain_content += s;
            }
        } else if (msg.getType() == MiraiCP::SingleMessageType::OnlineForwardedMessage_t) {
            MiraiCP::Logger::logger.logger.info(msg->toJson());
        }
    }
    if (ret.plain_content != nullptr) {
        *ret.plain_content = std::string{rtrim(ltrim(*ret.plain_content))};
        if (ret.plain_content->empty()) {
            *ret.plain_content = EMPTY_MSG_TAG;
        } else if (const auto at_me_str = fmt::format("@{}", bot_name); ret.plain_content -> find(at_me_str) != std::string::npos) {
            ret.is_at_me = true;
            size_t pos = 0;
            while ((pos = ret.plain_content->find(at_me_str, pos)) != std::string::npos) {
                ret.plain_content->replace(pos, at_me_str.size(), "");
                pos += at_me_str.size();
            }
        }
    } else {
        ret.ref_msg_content = std::make_unique<std::string>(EMPTY_MSG_TAG);
    }

    return ret;
}

void msg_storage(const MessageProperties &msg_prop, MiraiCP::QQID group_id, MiraiCP::QQID sender_id,
                 const std::string_view sender_name) {
    if ((msg_prop.plain_content == nullptr || *msg_prop.plain_content == EMPTY_MSG_TAG) &&
        (msg_prop.ref_msg_content == nullptr || *msg_prop.ref_msg_content == EMPTY_MSG_TAG)) {
        return;
    }

    std::string msg_content =
        msg_prop.ref_msg_content == nullptr
            ? *msg_prop.plain_content
            : fmt::format("引用了消息: {}\n{}", *msg_prop.ref_msg_content, *msg_prop.plain_content);

    rag::insert_group_msg(group_id, sender_name, sender_id, sender_name, msg_content);
}