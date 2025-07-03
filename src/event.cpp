#include "adapter_event.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "config.h"
#include "constants.hpp"
#include "database.h"
#include "individual_message_storage.hpp"
#include "llm.h"
#include "msg_prop.h"
#include "time_utils.h"
#include "utils.h"
#include <chrono>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <functional>
#include <general-wheel-cpp/string_utils.hpp>
#include <global_data.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

using namespace wheel;

struct ParseRunCmdRes {
    bool skip_default_llm = false;
    bool is_deep_think = false;
};

/**
 * @brief Parse command in user chat text
 */
ParseRunCmdRes parse_and_run_chat_command(bot_adapter::BotAdapter &adapter,
                                          std::shared_ptr<bot_adapter::MessageEvent> event, MessageProperties &msg_prop,
                                          qq_id_t sender_id) {
    bool is_deep_think = false;

    spdlog::info("开始处理指令信息");
    const auto admin = is_admin(sender_id);

    /// Commands to run.
    /// pair.first: command function,
    /// pair.second: arguments raw string.
    std::vector<std::pair<std::function<bot_cmd::CommandRes(bot_cmd::CommandContext)>, std::string>> run_cmd_list;
    // Loop all command properties.
    ParseRunCmdRes ret;
    for (auto &cmd : bot_cmd::keyword_command_map) {

        if (cmd.second.is_need_admin && !admin) {
            continue;
        }

        // Skip definitely no command situations.
        if (msg_prop.plain_content == nullptr || msg_prop.plain_content->empty()) {
            continue;
        }

        if (cmd.second.is_need_param) { ///< process need param
            if (auto param = extract_parentheses_content_after_keyword(*msg_prop.plain_content, cmd.first);
                !param.empty()) {
                // Remove the str which used to be command(param)
                auto p = std::string(param);
                *msg_prop.plain_content =
                    replace_keyword_and_parentheses_content(*msg_prop.plain_content, cmd.first, "");
                // No any param
                if (is_strict_format(*msg_prop.plain_content, cmd.first)) {
                    param = "";
                }
                run_cmd_list.push_back(std::make_pair(cmd.second.runer, std::string(param)));

            } else if (msg_prop.plain_content->find(cmd.first) != std::string::npos) {
                adapter.send_replay_msg(*event->sender_ptr,
                                        bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                            fmt::format(" 错误。请指定参数, 用法 {} (...)", cmd.first))));
                ret.skip_default_llm = true;
                return ret;
            }
        } else { ///< process don't need param
            if (msg_prop.plain_content->find(cmd.first) != std::string::npos) {
                *msg_prop.plain_content = replace_str(*msg_prop.plain_content, cmd.first, "");
                run_cmd_list.push_back(std::make_pair(cmd.second.runer, ""));
            } else {
                continue;
            }
        }
    }

    for (const auto &cmd : run_cmd_list) {

        const auto res = cmd.first(bot_cmd::CommandContext{adapter, event, cmd.second, is_deep_think, msg_prop});
        if (res.is_deep_think) {
            is_deep_think = true;
            ret.is_deep_think = true;
        }
        if (res.skip_default_process_llm) {
            ret.skip_default_llm = true;
        }

        if (res.is_modify_msg) {
            res.is_modify_msg.value()(msg_prop);
        }

        if (res.interrupt_following_commands) {
            return ret;
        }
    }
    return ret;
}

ParseRunCmdRes message_preprocessing(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::MessageEvent> event,
                                     MessageProperties &msg_prop, qq_id_t sender_id) {
    auto ret = parse_and_run_chat_command(adapter, event, msg_prop, sender_id);
    // Processing empty messagem
    if ((msg_prop.plain_content == nullptr || ltrim(rtrim(*msg_prop.plain_content)).empty()) &&
        (msg_prop.ref_msg_content == nullptr || ltrim(rtrim(*msg_prop.ref_msg_content)).empty()))
        *msg_prop.plain_content = EMPTY_MSG_TAG;

    return ret;
}

void store_msg(const MessageProperties &msg_prop, const std::shared_ptr<bot_adapter::MessageEvent> event,
               const std::chrono::system_clock::time_point &send_time) {
    auto msg_storage_thread = std::thread([msg_prop, event, send_time] {
        set_thread_name("AIBot msg storage");
        spdlog::info("Start message storage thread.");

        store_msg_prop_to_db(msg_prop, *event->sender_ptr, send_time);
    });
    msg_storage_thread.detach();
}

void on_group_msg_event(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::GroupMessageEvent> event) {
    const auto sender_id = event->sender_ptr->id;

    auto bot_profile = adapter.get_bot_profile();
    std::string_view bot_name = bot_profile.name;
    auto bot_id = bot_profile.id;
    auto msg_prop = get_msg_prop_from_event(*event, bot_name, bot_id);

    spdlog::debug("Event: {}", event->to_json().dump());
    spdlog::debug("Sender: {}", event->sender_ptr->to_json().dump());

    g_group_message_storage.add_message(event->get_group_sender().group.id, event->message_id,
                                        MessageStorageEntry{event->message_id, event->sender_ptr->name,
                                                            event->sender_ptr->id, event->send_time,
                                                            event->message_chain_ptr});

    store_msg(msg_prop, event, event->send_time);

    if (is_banned_id(sender_id)) {
        return;
    }

    if (!msg_prop.is_at_me) {
        return;
    }

    // Get User preference
    auto user_preference = database::get_global_db_connection().get_user_preference(event->sender_ptr->id);
    if (!user_preference.has_value()) {
        spdlog::warn("用户'{}'没有偏好设置，创建默认偏好设置", event->sender_ptr->id);
        user_preference = database::UserPreference{};
        database::get_global_db_connection().insert_or_update_user_preferences(
            std::vector<std::pair<qq_id_t, database::UserPreference>>{
                std::make_pair(event->sender_ptr->id, database::UserPreference{})});
    }
    spdlog::info("用户'{}'的偏好设置: {}", event->sender_ptr->id, user_preference.value().to_string());


    if (user_preference->auto_new_chat_session_sec.has_value() && user_preference->auto_new_chat_session_sec.value() > 0) {
        auto last_msg_list = g_group_message_storage.get_individual_last_msg_list(event->get_group_sender().group.id, 1);
        if (last_msg_list.size() > 0) {
            auto last_msg_time = last_msg_list[0].get().send_time;
            auto event_time = event->send_time;
            if (event_time - last_msg_time > std::chrono::seconds(user_preference->auto_new_chat_session_sec.value())) {
                spdlog::info("用户对话超过{}秒，创建新的对话", user_preference->auto_new_chat_session_sec.value());
                g_chat_session_map.erase(event->get_group_sender().id);
                g_chat_session_knowledge_list_map.erase(event->get_group_sender().id);
            }
        }
    }

    spdlog::debug("At list: {}", join_str(std::cbegin(msg_prop.at_id_set), std::cend(msg_prop.at_id_set), ",",
                                          [](const auto i) { return std::to_string(i); }));

    // process at id set
    const auto &group_member_info =
        adapter.fetch_group_member_info(event->get_group_sender().group.id)->get().member_info_list;
    if (!msg_prop.at_id_set.empty() && msg_prop.at_id_set.size() > 1) {
        std::string mention_str = "提到了群友:";

        for (const auto &at_id : msg_prop.at_id_set) {
            if (at_id == bot_id) {
                continue;
            }
            if (auto member_info_opt = group_member_info->find(at_id); member_info_opt.has_value()) {
                auto &member_info = member_info_opt->get();

                mention_str += fmt::format("\n- {}", member_info.member_name);

                if (member_info.special_title.has_value() && !member_info.special_title->empty()) {
                    mention_str += fmt::format("\n  头衔: {}", *member_info.special_title);
                }

                mention_str += fmt::format("\n  身份: {}", bot_adapter::get_permission_chs(member_info.permission));

                if (member_info.join_time.has_value()) {
                    mention_str +=
                        fmt::format("\n  入群时间: {}", system_clock_to_string(member_info.join_time.value()));
                }

                if (member_info.mute_time_remaining > 0) {
                    mention_str += fmt::format("\n  禁言剩余时间: {:.2f}分钟", member_info.mute_time_remaining / 60);
                }
            }
        }

        if (msg_prop.plain_content == nullptr) {
            msg_prop.plain_content = std::make_shared<std::string>(mention_str);
        } else {
            *msg_prop.plain_content += "\n" + mention_str;
        }
    }

    auto process_res = message_preprocessing(adapter, event, msg_prop, sender_id);
    if (!process_res.skip_default_llm) {
        auto context = bot_cmd::CommandContext(adapter, event, "", process_res.is_deep_think, msg_prop);
        process_llm(context, std::nullopt);
    }
}

void on_friend_msg_event(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::FriendMessageEvent> event) {
    const auto sender_id = event->sender_ptr->id;

    auto bot_profile = adapter.get_bot_profile();
    std::string_view bot_name = bot_profile.name;
    auto bot_id = bot_profile.id;
    auto msg_prop = get_msg_prop_from_event(*event, bot_name, bot_id);

    spdlog::debug("Event: {}", event->to_json().dump());
    spdlog::debug("Sender: {}", event->sender_ptr->to_json().dump());

    g_friend_message_storage.add_message(event->sender_ptr->id, event->message_id,
                                         MessageStorageEntry{event->message_id, event->sender_ptr->name,
                                                             event->sender_ptr->id, event->send_time,
                                                             event->message_chain_ptr});

    store_msg(msg_prop, event, event->send_time);

    if (is_banned_id(sender_id)) {
        return;
    }

    auto process_res = message_preprocessing(adapter, event, msg_prop, sender_id);

    if (!process_res.skip_default_llm) {
        auto context = bot_cmd::CommandContext(adapter, event, "", process_res.is_deep_think, msg_prop);
        process_llm(context, std::nullopt);
    }
}

void register_event(bot_adapter::BotAdapter &adapter) {
    adapter.register_event<bot_adapter::GroupMessageEvent>(
        std::bind(on_group_msg_event, std::ref(adapter), std::placeholders::_1));
    adapter.register_event<bot_adapter::FriendMessageEvent>(
        std::bind(on_friend_msg_event, std::ref(adapter), std::placeholders::_1));
}