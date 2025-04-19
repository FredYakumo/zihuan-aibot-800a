#include "bot_cmd.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "llm.h"
#include "msg_prop.h"
#include "rag.h"
#include "utils.h"
#include <charconv>
#include <chrono>
#include <optional>
#include <utility>

namespace bot_cmd {
    std::vector<std::pair<std::string, bot_cmd::CommandProperty>> keyword_command_map;

    // CommandRes queto_command(bot_adapter::BotAdapter &adapter, CommandContext context) {
    //     std::string res{};
    //     const auto group_msg = rag::query_group_msg(context.param, context.group_id);
    //     if (group_msg.empty()) {
    //         if (const auto group_sender = bot_adapter::try_group_sender(*context.e.sender_ptr)) {
    //             adapter.send_message(
    //                 group_sender->get().group,
    //                 bot_adapter::make_message_chain_list(
    //                     bot_adapter::AtTargetMessage(group_sender->get().id),
    //                     bot_adapter::PlainTextMessage(fmt::format(" 在本群中未找到关于\"{}\"的语录",
    //                     context.param))));
    //         }

    //         return CommandRes{true};
    //     }
    //     for (const auto &e : group_msg) {
    //         res.append(fmt::format("\n{}: {}。时间: {}, 关联度: {:.4f}", e.first.sender_name, e.first.content,
    //                                e.first.send_time, e.second));
    //     }

    //     if (context.is_command_only) {
    //         output_in_split_string(adapter, bot_adapter::Group(context.group_id, "", ""),
    //                                bot_adapter::GroupSender(*context.e.sender_ptr_id, *context.e.sender_ptr_name,
    //                                std::nullopt, "",
    //                                                         std::nullopt, std::chrono::system_clock::now()),
    //                                res);
    //         return CommandRes{true};
    //     }

    //     return CommandRes{false, [res](const MessageProperties &msg_prop) {
    //                           *msg_prop.plain_content = replace_keyword_and_parentheses_content(
    //                               *msg_prop.plain_content, "#语录", fmt::format("在本群中关于: {}的消息", res));
    //                       }};
    // }

    CommandRes clear_chat_session_command(CommandContext context) {
        spdlog::info("开始清除聊天记录");
        auto chat_session_map = g_chat_session_map.write();
        if (auto it = chat_session_map->find(context.e->sender_ptr->id); it != chat_session_map->cend()) {
            chat_session_map->erase(it);
        }
        if ((context.msg_prop.plain_content != nullptr || !ltrim(rtrim(*context.msg_prop.plain_content)).empty()) &&
            (context.msg_prop.ref_msg_content != nullptr || !ltrim(rtrim(*context.msg_prop.ref_msg_content)).empty())) {
            return CommandRes{false,
                              [](const MessageProperties &msg_prop) {
                                  *msg_prop.plain_content = replace_str(*msg_prop.plain_content, "#新对话", "");
                              },
                              false};
        }
        context.adapter.send_replay_msg(
            *context.e->sender_ptr,
            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("成功清除了对话上下文，请继续跟我聊天吧。")));
        return CommandRes{true};
    }

    CommandRes deep_think_command(CommandContext context) {
        spdlog::info("开始深度思考");

        return CommandRes{false,
                          [](const MessageProperties &msg_prop) {
                              *msg_prop.plain_content = replace_str(*msg_prop.plain_content, "#思考", "");
                          },
                          true};
    }

    CommandRes query_knowledge_command(CommandContext context) {
        std::string res{};
        const auto query_msg = rag::query_knowledge(context.param);
        if (query_msg.empty()) {
            context.adapter.send_replay_msg(*context.e->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                fmt::format(" 未找到关于\"{}\"的数据", context.param))));
            return CommandRes{true};
        }
        for (const auto &e : query_msg) {
            res.append(fmt::format("\n{}。创建者: {}, 时间: {}, 关联度: {:.4f}", e.first.content, e.first.creator_name,
                                   e.first.create_dt, e.second));
        }
        context.adapter.send_long_plain_text_replay(*context.e->sender_ptr, res);
        return CommandRes{true};
    }

    CommandRes add_knowledge_command(CommandContext context) {
        spdlog::info("{} 添加了知识到待添加列表中: {}", context.e->sender_ptr->name, context.param);
        std::thread([context] {
            std::string content{context.param};
            auto wait_add_list = g_wait_add_knowledge_list.write();
            wait_add_list->emplace_back(DBKnowledge{content, context.e->sender_ptr->name});
            context.adapter.send_replay_msg(*context.e->sender_ptr,
                                            bot_adapter::make_message_chain_list(
                                                bot_adapter::PlainTextMessage("成功, 添加1条知识到待添加列表中。")));
        }).detach();

        return CommandRes{true};
    }

    CommandRes checkin_knowledge_command(CommandContext context) {
        size_t index = 0;
        auto param = context.param;
        auto [ptr, ec] = std::from_chars(param.data(), param.data() + param.size(), index);
        if (ec != std::errc() && ptr != param.data() + param.size()) {
            context.adapter.send_replay_msg(*context.e->sender_ptr,
                                            bot_adapter::make_message_chain_list(
                                                bot_adapter::PlainTextMessage("错误。用法: #入库知识 (id: number)")));
            return CommandRes{true};
        }
        spdlog::info("Index = {}", index);
        std::thread([context, index] {
            spdlog::info("Start add knowledge thread.");
            auto wait_add_list = g_wait_add_knowledge_list.write();

            if (index >= wait_add_list->size()) {
                context.adapter.send_replay_msg(*context.e->sender_ptr,
                                                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                    fmt::format(" 错误。id {} 不存在于待添加列表中", index))));
                return;
            }
            rag::insert_knowledge(wait_add_list->at(index));
            wait_add_list->erase(wait_add_list->cbegin() + index);
            context.adapter.send_replay_msg(*context.e->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                fmt::format(" 入库知识成功。列表剩余{}条。", wait_add_list->size()))));
        }).detach();
        return CommandRes{true};
    }

    CommandRes query_memory_command(CommandContext context) {
        const auto bot_name = context.adapter.get_bot_profile().name;
        std::string memory_str = fmt::format(" '{}'当前记忆列表:\n", bot_name);
        auto chat_session_map = g_chat_session_map.read();
        if (chat_session_map->empty()) {
            memory_str += "空的";
        } else {
            for (auto entry : *chat_session_map) {
                memory_str += fmt::format("QQ号: {}, 昵称: {}, 记忆数: {}\n", entry.first, entry.second.nick_name,
                                          entry.second.user_msg_count);
                for (auto m : entry.second.message_list) {
                    auto role = m.role;
                    if (role == "user") {
                        role = entry.second.nick_name;
                    } else {
                        role = bot_name;
                    }
                    memory_str += fmt::format("\t- [{}] {}: {}\n", m.get_formatted_timestamp(), role, m.content);
                }
            }
        }
        context.adapter.send_long_plain_text_replay(*context.e->sender_ptr, memory_str);
        return CommandRes{true};
    }

    CommandRes query_add_knowledge_list_command(CommandContext context) {
        auto wait_add_list = g_wait_add_knowledge_list.read();
        if (wait_add_list->empty()) {
            context.adapter.send_replay_msg(
                *context.e->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("暂无待添加知识。")));
            return CommandRes{true};
        }
        std::string wait_add_list_str{" 待添加知识列表:"};
        size_t index = 0;
        for (; index < 4 && index < wait_add_list->size(); ++index) {
            const auto &knowledge = wait_add_list->at(index);
            wait_add_list_str.append(fmt::format("\n{} - {} - {}: {}", index, knowledge.creator_name,
                                                 knowledge.create_dt, knowledge.content));
        }
        if (index < wait_add_list->size()) {
            wait_add_list_str.append(fmt::format("\n...(剩余{}条)...", wait_add_list->size() - index));
        }
        context.adapter.send_long_plain_text_replay(*context.e->sender_ptr, wait_add_list_str);
        return CommandRes{true};
    }

    bot_cmd::CommandRes net_search_command(bot_cmd::CommandContext context) {
        std::string search{};
        if (context.msg_prop.ref_msg_content != nullptr && !rtrim(ltrim(*context.msg_prop.ref_msg_content)).empty()) {
            search += *context.msg_prop.ref_msg_content;
        }
        if (context.msg_prop.plain_content != nullptr && !rtrim(ltrim(*context.msg_prop.plain_content)).empty()) {
            search += '\n' + *context.msg_prop.plain_content;
        }

        // If no search query is provided, prompt the user to enter one
        if (search.empty() || search == "#联网") {
            context.adapter.send_replay_msg(*context.e->sender_ptr, bot_adapter::make_message_chain_list(
                                                                        bot_adapter::PlainTextMessage("请输入查询。")));
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            spdlog::info("Start net search thread");
            auto search_text = replace_str(search, "#联网", "");
            auto net_search_res = rag::net_search_content(search_text);
            std::string net_search_str;
            if (net_search_res.empty()) {
                net_search_str = fmt::format("联网搜索了{}, 但是没有搜到任何东西。", search_text);
            } else {
                net_search_str += "\n以下是联网查询的结果, "
                                  "由于这个输入用户看不到，所以请在回答中列出概要或者详细的结果(根据用户的指示):\n";
                for (const auto res : net_search_res) {
                    net_search_str.append(fmt::format("{},{}:{}\n", res.url, res.title, res.content));
                }
            }
            spdlog::info(net_search_str);
            *context.msg_prop.plain_content = replace_str(search, "#联网", net_search_str);
            process_llm(context, net_search_str);
        }).detach();

        return bot_cmd::CommandRes{true};
    }

    bot_cmd::CommandRes url_search_command(bot_cmd::CommandContext context) {
        std::string search{};
        if (context.msg_prop.ref_msg_content != nullptr && !rtrim(ltrim(*context.msg_prop.ref_msg_content)).empty()) {
            search += *context.msg_prop.ref_msg_content;
        }
        if (context.msg_prop.plain_content != nullptr && !rtrim(ltrim(*context.msg_prop.plain_content)).empty()) {
            search += '\n' + *context.msg_prop.plain_content;
        }

        // If no search query is provided, prompt the user to enter one
        if (search.empty() || search == "#url") {
            context.adapter.send_replay_msg(*context.e->sender_ptr, bot_adapter::make_message_chain_list(
                                                                        bot_adapter::PlainTextMessage("请输入查询。")));
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            spdlog::info("Start url search thread");
            auto search_text = extract_parentheses_content_after_keyword(search, "#url");

            std::vector<std::string> url_list;
            for (const auto url : SplitString(search_text, ',')) {
                auto u = std::string{ltrim(rtrim(url))};
                spdlog::info("URL: {}", u);
                url_list.emplace_back(u);
            }

            auto net_search_res = rag::url_search_content(url_list);
            spdlog::info(net_search_res);
            *context.msg_prop.plain_content = replace_keyword_and_parentheses_content(search, "#url", net_search_res);
            process_llm(context, net_search_res);
        }).detach();

        return bot_cmd::CommandRes{true};
    }
} // namespace bot_cmd
