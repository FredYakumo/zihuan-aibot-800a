#include "msg_prop.h"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "database.h"
#include "global_data.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <regex>
#include <spdlog/spdlog.h>
#include <string>

std::regex at_target_pattern("@(\\d+)");

MessageProperties get_msg_prop_from_event(const bot_adapter::MessageEvent &e, const std::string_view bot_name,
                                          uint64_t bot_id) {
    MessageProperties ret{};
    for (auto msg : e.message_chain) {
        if (msg == nullptr) {
            continue;
        }
        spdlog::info("Message type: {}, json: {}", msg->get_type(), msg->to_json().dump());

        if (const auto at_msg = bot_adapter::try_at_target_message(*msg)) {
            ret.at_id_set.insert(at_msg->get().target);
            if (at_msg->get().target == bot_id) {
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
        if (rtrim(ltrim(*ret.plain_content)).empty()) {
            // Empty msg process

            *ret.plain_content = EMPTY_MSG_TAG;
        } else if (const auto at_me_str = fmt::format("@{}", bot_name);
                   !bot_name.empty() && ret.plain_content->find(at_me_str) != std::string::npos) {
            ret.is_at_me = true;
            size_t pos = 0;
            while ((pos = ret.plain_content->find(at_me_str, pos)) != std::string::npos) {
                ret.plain_content->replace(pos, at_me_str.size(), "");
                pos += at_me_str.size();
            }

            // Regex match @[qqid(uint64_t)]
            auto begin =
                std::sregex_iterator(std::cbegin(*ret.plain_content), std::cend(*ret.plain_content), at_target_pattern);
            auto end = std::sregex_iterator();

            // 遍历所有匹配结果，提取数字并转换为 uint64_t 后存入 at_list
            for (std::sregex_iterator i = begin; i != end; ++i) {
                std::smatch match = *i;
                uint64_t value = std::stoull(match[1].str());
                ret.at_id_set.insert(value);
                if (value == bot_id) {
                    ret.is_at_me = true;
                }
            }

            // 将所有匹配部分替换为空字符串
            *ret.plain_content = std::regex_replace(*ret.plain_content, at_target_pattern, "");
        }
    } else {
        // Empty msg process
        ret.plain_content = std::make_unique<std::string>(EMPTY_MSG_TAG);
    }

    return ret;
}

void msg_storage(const MessageProperties &msg_prop, const bot_adapter::Sender &sender,
                 const std::chrono::system_clock::time_point &send_time,
                 const std::optional<std::set<uint64_t>> specify_at_target_set) {
    if ((msg_prop.plain_content == nullptr || *msg_prop.plain_content == EMPTY_MSG_TAG) &&
        (msg_prop.ref_msg_content == nullptr || *msg_prop.ref_msg_content == EMPTY_MSG_TAG)) {
        return;
    }

    std::string msg_content =
        msg_prop.ref_msg_content == nullptr
            ? *msg_prop.plain_content
            : fmt::format("引用了消息: {}\n{}", *msg_prop.ref_msg_content, *msg_prop.plain_content);

    database::get_global_db_connection().insert_message(
        msg_content, sender, send_time, specify_at_target_set ? specify_at_target_set : msg_prop.at_id_set);
}

std::vector<std::string> get_message_list_from_chat_session(const std::string_view sender_name, qq_id_t sender_id) {
    std::vector<std::string> ret;

    const auto &chat_session_map = g_chat_session_map.read();
    if (const auto &chat_session = chat_session_map->find(sender_id); chat_session != chat_session_map->cend()) {
        for (const auto &msg : chat_session->second.message_list) {
            ret.push_back(fmt::format("{}: {}", sender_name, msg.content));
        }
    }
    return std::move(ret);
}