#include "adapter_event.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "config.h"
#include "llm.h"
#include "msg_prop.h"
#include "rag.h"
#include "utils.h"
#include <chrono>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <functional>
#include <global_data.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

void on_group_msg_event(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::GroupMessageEvent> event) {
    const auto sender_id = event->sender_ptr->id;

    if (is_banned_id(sender_id)) {
        return;
    }

    auto bot_profile = adapter.get_bot_profile();
    std::string_view bot_name = bot_profile.name;
    auto bot_id = bot_profile.id;
    const auto msg_prop = get_msg_prop_from_event(*event, bot_name, bot_id);
    spdlog::debug("At list: {}", join_str(std::cbegin(msg_prop.at_id_set), std::cend(msg_prop.at_id_set), ",",
                                          [](const auto i) { return std::to_string(i); }));

    spdlog::debug("Event: {}", event->to_json().dump());

    spdlog::debug("Sender: {}", event->sender_ptr->to_json().dump());

    const bot_adapter::GroupSender &sender = event->get_group_sender();
    const auto group_id = sender.group.id;
    const auto group_name = sender.group.name;
    const auto sender_name = event->sender_ptr->name;
    const auto send_time = std::chrono::system_clock::now();

    const auto admin = is_admin(sender_id);

    bool is_deep_think = false;

    // @TODO: need optim

    std::vector<std::pair<std::function<bot_cmd::CommandRes(bot_cmd::CommandContext)>, bot_cmd::CommandContext>>
        run_cmd_list;
    if (msg_prop.is_at_me) {
        spdlog::info("开始处理指令信息");
        for (auto &cmd : bot_cmd::keyword_command_map) {
            if (cmd.second.is_need_admin && !admin) {
                continue;
            }
            if (msg_prop.plain_content == nullptr || msg_prop.plain_content->empty()) {
                continue;
            }
            if (cmd.second.is_need_param) {
                if (auto param = extract_parentheses_content_after_keyword(*msg_prop.plain_content, cmd.first);
                    !param.empty()) {
                    // Remove the str which used to be command(param)
                    *msg_prop.plain_content = replace_keyword_and_parentheses_content(*msg_prop.plain_content, cmd.first, "");
                    
                    run_cmd_list.push_back(std::make_pair(
                        cmd.second.runer, bot_cmd::CommandContext(adapter, event, param,
                                                                  is_strict_format(*msg_prop.plain_content, cmd.first),
                                                                  is_deep_think, msg_prop)));

                } else {
                    adapter.send_replay_msg(*event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                fmt::format(" 错误。请指定参数, 用法 {} (...)", cmd.first))));
                    return;
                }
            } else {
                if (msg_prop.plain_content->find(cmd.first) != std::string::npos) {
                    *msg_prop.plain_content = replace_str(*msg_prop.plain_content, cmd.first, "");
                    run_cmd_list.push_back(std::make_pair(
                        cmd.second.runer, bot_cmd::CommandContext(adapter, event, "", true, is_deep_think, msg_prop)));
                } else {
                    continue;
                }
            }
        }
    }

    rag::optimize_message_query(event->get_group_sender().id);

    for (const auto &cmd : run_cmd_list) {
        const auto res = cmd.first(cmd.second);
        if (res.is_break_cmd_process) {
            return;
        }
        if (res.is_deep_think) {
            is_deep_think = true;
        }

        if (res.is_modify_msg) {
            res.is_modify_msg.value()(msg_prop);
        }
    }

    auto msg_storage_thread = std::thread([msg_prop, event, send_time] {
        set_thread_name("AIBot msg storage");
        spdlog::info("Start message storage thread.");

        msg_storage(msg_prop, *event->sender_ptr, send_time);
    });
    msg_storage_thread.detach();

    if (!msg_prop.is_at_me) {
        return;
    }

    auto context = bot_cmd::CommandContext(adapter, event, "", false, is_deep_think, msg_prop);
    process_llm(context, std::nullopt);
}

void on_friend_msg_event(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::FriendMessageEvent> event) {
    const auto sender_id = event->sender_ptr->id;

    if (is_banned_id(sender_id)) {
        return;
    }

    auto bot_profile = adapter.get_bot_profile();
    std::string_view bot_name = bot_profile.name;
    auto bot_id = bot_profile.id;
    const auto msg_prop = get_msg_prop_from_event(*event, bot_name, bot_id);

    spdlog::debug("Event: {}", event->to_json().dump());

    spdlog::debug("Sender: {}", event->sender_ptr->to_json().dump());

    const bot_adapter::Sender &sender = *event->sender_ptr;
    const auto sender_name = event->sender_ptr->name;

    const auto admin = is_admin(sender_id);

    const auto send_time = std::chrono::system_clock::now();

    bool is_deep_think = false;

    // @TODO: need optim
    spdlog::info("开始处理指令信息");

    for (auto &cmd : bot_cmd::keyword_command_map) {
        if (cmd.second.is_need_admin && !admin) {
            continue;
        }
        if (msg_prop.plain_content == nullptr || msg_prop.plain_content->empty()) {
            continue;
        }
        bot_cmd::CommandRes res;
        if (cmd.second.is_need_param) {
            if (auto param = extract_parentheses_content_after_keyword(*msg_prop.plain_content, cmd.first);
                !param.empty()) {
                res = cmd.second.runer(bot_cmd::CommandContext(adapter, event, param,
                                                               is_strict_format(*msg_prop.plain_content, cmd.first),
                                                               is_deep_think, msg_prop));
            } else {
                adapter.send_replay_msg(*event->sender_ptr,
                                        bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                            fmt::format(" 错误。请指定参数, 用法 {} (...)", cmd.first))));
                return;
            }
        } else {
            if (msg_prop.plain_content->find(cmd.first) != std::string::npos) {
                res = cmd.second.runer(bot_cmd::CommandContext(adapter, event, "", true, is_deep_think, msg_prop));
            } else {
                continue;
            }
        }
        if (res.is_break_cmd_process) {
            return;
        }
        if (res.is_deep_think) {
            is_deep_think = true;
        }

        if (res.is_modify_msg) {
            res.is_modify_msg.value()(msg_prop);
        }
    }

    auto msg_storage_thread = std::thread([msg_prop, event, send_time, bot_id] {
        set_thread_name("AIBot msg storage");
        spdlog::info("Start message storage thread.");
        msg_storage(msg_prop, *event->sender_ptr, send_time, std::set<uint64_t>{bot_id});
    });
    msg_storage_thread.detach();

    // if (!msg_prop.is_at_me) {
    //     return;
    // }

    auto context = bot_cmd::CommandContext(adapter, event, "", false, is_deep_think, msg_prop);
    process_llm(context, std::nullopt);
}

void register_event(bot_adapter::BotAdapter &adapter) {
    adapter.register_event<bot_adapter::GroupMessageEvent>(
        std::bind(on_group_msg_event, std::ref(adapter), std::placeholders::_1));
    adapter.register_event<bot_adapter::FriendMessageEvent>(
        std::bind(on_friend_msg_event, std::ref(adapter), std::placeholders::_1));
}