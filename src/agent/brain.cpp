#include "agent/brain.h"
#include "adapter_model.h"
#include "agent/llm.h"
#include "bot_cmd.h"
#include "config.h"
#include "constant_types.hpp"
#include "msg_prop.h"
#include <iterator>
#include <string_utils.hpp>

namespace agent {

    Config &conf = Config::instance();

    using std::string_view;
    using wheel::join_str;
    using wheel::ltrim;
    using wheel::rtrim;

    std::string get_mixed_user_input_message_text(string_view sender_name, qq_id_t bot_id,
                                                  const MessageProperties &msg_prop) {
        std::string llm_content{};
        if (msg_prop.ref_msg_content != nullptr && !msg_prop.ref_msg_content->empty()) {
            llm_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", sender_name, *msg_prop.ref_msg_content));
        }
        if (!msg_prop.at_id_set.empty() && !(msg_prop.at_id_set.size() == 1 && *msg_prop.at_id_set.begin() == bot_id)) {
            if (*msg_prop.at_id_set.begin() == bot_id) {
                llm_content.append("本消息提到了你");
            } else {
                llm_content.append("本消息提到了: ");
                llm_content.append(join_str(std::cbegin(msg_prop.at_id_set), std::cend(msg_prop.at_id_set), "、"));
                llm_content.append("。");
            }
        }
        std::string speak_content = EMPTY_MSG_TAG;
        if (msg_prop.plain_content != nullptr &&
            !wheel::ltrim(wheel::rtrim(*msg_prop.plain_content)).empty()) {
            speak_content = *msg_prop.plain_content;
        }

        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            if (auto group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr);
                group_sender.has_value()) {
                llm_content.append(fmt::format("{}\"{}\"({})[{}]对你说: \"{}\"",
                                               bot_adapter::get_permission_chs(group_sender->get().permission),
                                               group_sender->get().name, context.event->sender_ptr->id,
                                               get_current_time_formatted(), speak_content));
            } else {
                llm_content.append(fmt::format("\"{}\"({})[{}]对你说: \"{}\"", context.event->sender_ptr->name,
                                               context.event->sender_ptr->id, get_current_time_formatted(),
                                               speak_content));
            }
        }
        std::string mixed_input_content;
        if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
            mixed_input_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                                   *context.msg_prop.ref_msg_content));
        }
        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            mixed_input_content.append(fmt::format("{}", *context.msg_prop.plain_content));
        }
    }

    void Brain::process_friend_message_event(const bot_adapter::FriendMessageEvent &event) {
        spdlog::info("Brain agent Processing friend message event from user ID: {}", event.sender_ptr->id);
        // Implement friend message processing logic here
    }

    void Brain::process_group_message_event(const bot_cmd::CommandContext &context) {
        const auto &gs = bot_adapter::try_group_sender(*context.event->sender_ptr);
        if (!gs.has_value()) {
            spdlog::error("Brain agent: Oops failed processing: Failed to get group sender info for user ID: {}",
                          context.event->sender_ptr->id);
            return;
        }
        const auto &sender = gs->get();
        spdlog::info("Brain agent Processing group message event from group ID: {}, user ID: {}", sender.group.id,
                     sender.id);

        gen_common_prompt(m_adapter->get_bot_profile(), *m_adapter, sender, false, "向这条消息做出反应",
                          conf.custom_system_prompt_option);
    }

} // namespace agent
