#include "utils.h"
#include "config.h"
#include "constants.hpp"
#include "global_data.h"
#include <cstdint>
#include <fmt/format.h>
#include <string_view>

std::string gen_common_prompt(const std::string_view bot_name, const MiraiCP::QQID bot_id,
                              const std::string_view user_name, const uint64_t user_id) {
    return fmt::format("你的名字叫{}(qq号{})，{}。当前时间是: {}，当前跟你聊天的群友的名字叫\"{}\"(qq号{})，", bot_name,
                       bot_id, CUSTOM_SYSTEM_PROMPT, get_current_time_formatted(), user_name, user_id);
}

MessageProperties get_msg_prop_from_event(const MiraiCP::GroupMessageEvent &e) {
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
        *ret.plain_content = std::string{ltrim(*ret.plain_content)};
        if (ret.plain_content->empty()) {
            *ret.plain_content = EMPTY_MSG_TAG;
        }
    } else {
        ret.ref_msg_content = std::make_unique<std::string>(EMPTY_MSG_TAG);
    }

    return ret;
}