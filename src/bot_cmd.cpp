#include "bot_cmd.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "config.h"
#include "global_data.h"
#include "llm.h"
#include "msg_prop.h"
#include "rag.h"
#include "utils.h"
#include <charconv>
#include <chrono>
#include <general-wheel-cpp/string_utils.hpp>
#include <optional>
#include <utility>

namespace bot_cmd {
    using namespace wheel;
    std::vector<std::pair<std::string, bot_cmd::CommandProperty>> keyword_command_map;

    CommandRes clear_chat_session_command(CommandContext context) {
        spdlog::info("开始清除聊天记录");
        g_chat_session_map.erase(context.event->sender_ptr->id);
        g_chat_session_knowledge_list_map.erase(context.event->sender_ptr->id);

        if ((context.msg_prop.plain_content == nullptr ||
             ltrim(rtrim(replace_str(*context.msg_prop.plain_content, "#新对话", ""))).empty()) &&
            (context.msg_prop.ref_msg_content == nullptr || ltrim(rtrim(*context.msg_prop.ref_msg_content)).empty())) {
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                "成功清除了对话上下文，请继续跟我聊天吧。")));
            return CommandRes{true, true};
        }
        return CommandRes{false, false, false};
    }

    CommandRes deep_think_command(CommandContext context) {
        spdlog::info("开始深度思考");

        return CommandRes{false, false, true};
    }

    CommandRes query_knowledge_command(CommandContext context) {
        std::string res{};
        const auto query_msg = rag::query_knowledge(context.param);
        if (query_msg.empty()) {
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                fmt::format(" 未找到关于\"{}\"的数据", context.param))));
            return CommandRes{true};
        }
        for (const auto &e : query_msg) {
            res.append(fmt::format("\nkey: \"{}\", value: \"{}\"。创建者: {}, 时间: {},   置信度: {:.4f}", e.key,
                                   e.value, e.creator_name, e.create_dt, e.certainty));
        }
        context.adapter.send_long_plain_text_reply(*context.event->sender_ptr, res);
        return CommandRes{true};
    }

    CommandRes add_knowledge_command(CommandContext context) {
        // spdlog::info("{} 添加了知识到待添加列表中: {}", context.e->sender_ptr->name, context.param);
        // std::thread([context] {
        //     std::string content{context.param};
        //     auto wait_add_list = g_wait_add_knowledge_list.write();
        //     wait_add_list->emplace_back(DBKnowledge{content, context.e->sender_ptr->name});
        //     context.adapter.send_replay_msg(*context.e->sender_ptr,
        //                                     bot_adapter::make_message_chain_list(
        //                                         bot_adapter::PlainTextMessage("成功, 添加1条知识到待添加列表中。")));
        // }).detach();

        return CommandRes{true};
    }

    CommandRes checkin_knowledge_command(CommandContext context) {
        size_t index = 0;
        auto param = context.param;
        auto [ptr, ec] = std::from_chars(param.data(), param.data() + param.size(), index);
        if (ec != std::errc() && ptr != param.data() + param.size()) {
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(
                                                bot_adapter::PlainTextMessage("错误。用法: #入库知识 (id: number)")));
            return CommandRes{true, true};
        }
        spdlog::info("Index = {}", index);
        std::thread([context, index] {
            spdlog::info("Start add knowledge thread.");

            g_wait_add_knowledge_list.modify_vector([context, index](std::vector<DBKnowledge> &wait_add_list) {
                if (index >= g_wait_add_knowledge_list.size()) {
                    context.adapter.send_replay_msg(*context.event->sender_ptr,
                                                    bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                        fmt::format(" 错误。id {} 不存在于待添加列表中", index))));
                    return;
                }
                rag::insert_knowledge(wait_add_list.at(index));
                wait_add_list.erase(wait_add_list.cbegin() + index);
                context.adapter.send_replay_msg(
                    *context.event->sender_ptr,
                    bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                        fmt::format(" 入库知识成功。列表剩余{}条。", wait_add_list.size()))));
            });
        }).detach();
        return CommandRes{true, true};
    }

    CommandRes query_memory_command(CommandContext context) {
        const auto bot_name = context.adapter.get_bot_profile().name;
        std::string memory_str = fmt::format(" '{}'当前记忆列表:\n", bot_name);

        if (g_chat_session_map.empty()) {
            memory_str += "空的";
        } else {
            g_chat_session_map.for_each([&memory_str, &bot_name](const auto &key, const auto &value) {
                memory_str +=
                    fmt::format("QQ号: {}, 昵称: {}, 记忆数: {}\n", key, value.nick_name, value.message_list.size());
                for (auto m : value.message_list) {
                    auto role = m.role;
                    if (role == "user") {
                        role = value.nick_name;
                    } else {
                        role = bot_name;
                    }
                    memory_str += fmt::format("\t- [{}] {}: {}\n", m.get_formatted_timestamp(), role, m.content);
                }
            });
        }
        context.adapter.send_long_plain_text_reply(*context.event->sender_ptr, memory_str);
        return CommandRes{true};
    }

    CommandRes query_add_knowledge_list_command(CommandContext context) {
        if (g_wait_add_knowledge_list.empty()) {
            context.adapter.send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("暂无待添加知识。")));
            return CommandRes{true};
        }
        std::string wait_add_list_str{" 待添加知识列表:"};
        size_t index = 0;
        for (; index < 4 && index < g_wait_add_knowledge_list.size(); ++index) {
            if (const auto &knowledge = g_wait_add_knowledge_list[index]; knowledge.has_value()) {
                const auto &k = knowledge->get();
                wait_add_list_str.append(
                    fmt::format("\n{}: {} - {} - {}: {}", index, k.creator_name, k.create_dt, k.key, k.value));
            }
        }
        auto size = g_wait_add_knowledge_list.size();
        if (index < size) {
            wait_add_list_str.append(fmt::format("\n...(剩余{}条)...", size - index));
        }
        context.adapter.send_long_plain_text_reply(*context.event->sender_ptr, wait_add_list_str);
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
            context.adapter.send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("请输入查询。")));
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            spdlog::info("Start net search thread");
            auto search_text = replace_str(search, "#联网", "");
            auto net_search_res = rag::net_search_content(search_text);
            std::string net_search_str;
            std::vector<bot_adapter::ForwardMessageNode> first_replay;

            if (net_search_res.empty()) {
                net_search_str = fmt::format("联网搜索了{}, 但是没有搜到任何东西。", search_text);
            } else {
                net_search_str += "\n以下是联网查询的结果:\n";
                first_replay.emplace_back(
                    context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                    context.adapter.get_bot_profile().name,
                    bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("参考资料")));
                for (const auto res : net_search_res) {
                    net_search_str.append(fmt::format("{}( {} ):{}\n", res.url, res.title, res.content));
                    first_replay.emplace_back(
                        context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                        context.adapter.get_bot_profile().name,
                        bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                            fmt::format("关联度: {:.2f}%\n{}( {} )", res.score * 100.0f, res.title, res.url))));
                }
            }
            spdlog::info(net_search_str);
            if (!first_replay.empty()) {
                context.adapter.send_replay_msg(
                    *context.event->sender_ptr,
                    bot_adapter::make_message_chain_list(bot_adapter::ForwardMessage(
                        first_replay, bot_adapter::DisplayNode(std::string("联网搜索结果")))));
                // context.adapter.send_replay_msg(
                //     *context.event->sender_ptr,
                //     bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                //         "PS: 紫幻现在自己会思考要不要去网上找数据啦, 你可以不用每次都用#联网.")));
            }
            process_llm(context, net_search_str);
        }).detach();

        return bot_cmd::CommandRes{true, true};
    }

    bot_cmd::CommandRes url_search_command(bot_cmd::CommandContext context) {
        std::string search{context.param};

        // If no search query is provided, prompt the user to enter one
        if (search.empty() || search == "#url") {
            context.adapter.send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("请输入查询。")));
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            spdlog::info("Start url search thread");

            std::vector<std::string> url_list;
            for (const auto url : SplitString(search, ',')) {
                auto u = std::string{ltrim(rtrim(url))};
                spdlog::info("URL: {}", u);
                url_list.emplace_back(u);
            }

            auto net_search_res = rag::url_search_content(url_list);
            std::string content;

            // Process successful results
            for (const auto &[url, raw_content] : net_search_res.results) {
                content += fmt::format("链接[{}]内容:\n{}\n\n", url, raw_content);
            }

            // Process failed results
            if (!net_search_res.failed_reason.empty()) {
                content += "以下链接获取失败:\n";
                for (const auto &[url, error] : net_search_res.failed_reason) {
                    content += fmt::format("链接[{}]失败原因: {}\n", url, error);
                }
            }

            if (net_search_res.results.empty()) {
                context.adapter.send_replay_msg(
                    *context.event->sender_ptr,
                    bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage{
                        fmt::format("{}打开url: {}失败, 请重试.", context.adapter.get_bot_profile().name, search)}));
            } else {
                *context.msg_prop.plain_content = replace_keyword_and_parentheses_content(search, "#url", content);
                process_llm(context, content);
            }
        }).detach();

        return bot_cmd::CommandRes{true, true};
    }
} // namespace bot_cmd
