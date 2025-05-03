#include "msg_prop.h"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "database.h"
#include "rag.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <spdlog/spdlog.h>

MessageProperties get_msg_prop_from_event(const bot_adapter::MessageEvent &e, const std::string_view bot_name,
                                          uint64_t bot_id) {
    MessageProperties ret{};
    for (auto msg : e.message_chain) {
        if (msg == nullptr) {
            continue;
        }
        spdlog::info("Message type: {}, json: {}", msg->get_type(), msg->to_json().dump());

        if (const auto at_me_msg = bot_adapter::try_at_me_message(*msg)) {
            if (at_me_msg->get().target == bot_id) {
                ret.is_at_me = true;
            }
        } else if (auto quote_msg = bot_adapter::try_quote_message(*msg)) {
            spdlog::info("引用信息: {}", quote_msg->get().to_json().dump());
            std::string s = fmt::format("引用了一段消息文本: \"{}\"", quote_msg->get().get_quote_text());
            spdlog::debug("文本: {}", quote_msg->get().get_quote_text());
            if (ret.ref_msg_content == nullptr) {
                ret.ref_msg_content = std::make_unique<std::string>(s);
            } else {
                *ret.ref_msg_content += s;
            }
        } else if (auto plain = bot_adapter::try_plain_text_message(*msg)) {
            if (ret.plain_content == nullptr) {
                ret.plain_content = std::make_unique<std::string>(plain->get().text);
            } else {
                *ret.plain_content += plain->get().text;
            }
        }
        // else if (msg.getType() == MiraiCP::SingleMessageType::OnlineForwardedMessage_t) {
        //     MiraiCP::Logger::logger.logger.info(msg->toJson());
        // }
    }
    if (ret.plain_content != nullptr) {
        *ret.plain_content = std::string{rtrim(ltrim(*ret.plain_content))};
        if (ret.plain_content->empty()) {
            *ret.plain_content = EMPTY_MSG_TAG;
        } else if (const auto at_me_str = fmt::format("@{}", bot_name);
                   !bot_name.empty() && ret.plain_content->find(at_me_str) != std::string::npos) {
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

void msg_storage(const MessageProperties &msg_prop, const bot_adapter::Sender &sender, const std::chrono::system_clock::time_point &send_time) {
    if ((msg_prop.plain_content == nullptr || *msg_prop.plain_content == EMPTY_MSG_TAG) &&
        (msg_prop.ref_msg_content == nullptr || *msg_prop.ref_msg_content == EMPTY_MSG_TAG)) {
        return;
    }

    std::string msg_content =
        msg_prop.ref_msg_content == nullptr
            ? *msg_prop.plain_content
            : fmt::format("引用了消息: {}\n{}", *msg_prop.ref_msg_content, *msg_prop.plain_content);
    
    
    database::get_global_db_connection().insert_message(msg_content, sender, send_time);
}