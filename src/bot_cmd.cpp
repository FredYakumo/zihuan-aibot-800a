#include "bot_cmd.h"
#include "llm.h"
#include "rag.h"
#include "utils.h"
#include <charconv>

namespace bot_cmd {

    std::map<std::string, bot_cmd::CommandProperty> keyword_command_map;

    CommandRes queto_command(CommandContext context) {
        auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
        std::string res{};
        const auto group_msg = rag::query_group_msg(context.param, context.group_id);
        if (group_msg.empty()) {
            msg_chain.add(MiraiCP::PlainText{fmt::format(" 在本群中未找到关于\"{}\"的语录", context.param)});
            context.e.group.sendMessage(msg_chain);
            return CommandRes{true};
        }
        for (const auto &e : group_msg) {
            res.append(fmt::format("\n{}: {}。时间: {}, 关联度: {:.4f}", e.first.sender_name, e.first.content,
                                   e.first.send_time, e.second));
        }

        if (context.is_command_only) {
            output_in_split_string(context.e, context.sender_id, res);
            return CommandRes{true};
        }

        return CommandRes{false, [res](const MessageProperties &msg_prop) {
                              *msg_prop.plain_content = replace_keyword_and_parentheses_content(
                                  *msg_prop.plain_content, "#语录", fmt::format("在本群中关于: {}的消息", res));
                          }};
    }

    CommandRes query_knowledge_command(CommandContext context) {
        std::string res{};
        const auto query_msg = rag::query_knowledge(context.param);
        if (query_msg.empty()) {
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
            msg_chain.add(MiraiCP::PlainText{fmt::format(" 未找到关于\"{}\"的数据", context.param)});
            context.e.group.sendMessage(msg_chain);
            return CommandRes{true};
        }
        for (const auto &e : query_msg) {
            res.append(fmt::format("\n{}。创建者: {}, 时间: {}, 关联度: {:.4f}", e.first.content, e.first.creator_name,
                                   e.first.create_dt, e.second));
        }

        output_in_split_string(context.e, context.sender_id, res);
        return CommandRes{true};
    }

    CommandRes add_knowledge_command(CommandContext context) {
        MiraiCP::Logger::logger.info(
            fmt::format("{} 添加了知识到待添加列表中: {}", context.sender_name, context.param));
        std::thread([context] {
            std::string content{context.param};
            auto wait_add_list = g_wait_add_knowledge_list.write();
            wait_add_list->emplace_back(DBKnowledge{content, context.sender_name});
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
            msg_chain.add(MiraiCP::PlainText{" 成功, 添加1条知识到待添加列表中。"});
            context.e.group.sendMessage(msg_chain);
        }).detach();
        return CommandRes{true};
    }

    CommandRes checkin_knowledge_command(CommandContext context) {
        size_t index = 0;
        auto param = context.param;
        auto [ptr, ec] = std::from_chars(param.data(), param.data() + param.size(), index);
        if (ec != std::errc() && ptr != param.data() + param.size()) {
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};

            msg_chain.add(MiraiCP::PlainText{" 错误。用法: #入库知识 (id: number)"});
            context.e.group.sendMessage(msg_chain);
            return CommandRes{true};
        }
        MiraiCP::Logger::logger.info(fmt::format("Index = {}", index));

        std::thread([context, index] {
            MiraiCP::Logger::logger.info("Start add knowledge thread.");
            auto wait_add_list = g_wait_add_knowledge_list.write();
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};

            if (index >= wait_add_list->size()) {
                msg_chain.add(MiraiCP::PlainText{fmt::format(" 错误。id {} 不存在于待添加列表中", index)});
                context.e.group.sendMessage(msg_chain);
                return;
            }
            rag::insert_knowledge(wait_add_list->at(index));
            wait_add_list->erase(wait_add_list->cbegin() + index);
            msg_chain.add(MiraiCP::PlainText{fmt::format(" 入库知识成功。列表剩余{}条。", wait_add_list->size())});
            context.e.group.sendMessage(msg_chain);
        }).detach();
        return CommandRes{true};
    }

    CommandRes query_memory_command(CommandContext context) {
        std::string memory_str = fmt::format(" '{}'当前记忆列表:\n", context.bot_name);
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
                        role = context.bot_name;
                    }
                    memory_str += fmt::format("\t- [{}] {}: {}\n", m.get_formatted_timestamp(), role, m.content);
                }
            }
        }
        output_in_split_string(context.e, context.sender_id, memory_str);
        return CommandRes{true};
    }

    CommandRes query_add_knowledge_list_command(CommandContext context) {
        auto wait_add_list = g_wait_add_knowledge_list.read();
        auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
        if (wait_add_list->empty()) {
            msg_chain.add(MiraiCP::PlainText{"暂无待添加知识。"});
            context.e.group.sendMessage(msg_chain);
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
        msg_chain.add(MiraiCP::PlainText{wait_add_list_str});
        context.e.group.sendMessage(msg_chain);
        return CommandRes{true};
    }

    /**
     * @brief Handles a network search command based on the provided context.
     *
     * This function processes a network search command by extracting the search query from the message context.
     * If no valid search query is found, it prompts the user to provide one. Otherwise, it initiates a
     * background thread to perform the network search and process the results.
     *
     * @param context The command context containing message properties and sender information.
     * @return bot_cmd::CommandRes Returns a bot_cmd::CommandRes object indicating the success of the command handling.
     */
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
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
            msg_chain.add(MiraiCP::PlainText{" 请输入查询。"});
            context.e.group.sendMessage();
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            MiraiCP::Logger::logger.info("Start net search thread");
            auto search_text = get_current_time_formatted() + replace_str(search, "#联网", "");
            auto net_search_res = rag::net_search_content(search_text);
            std::string net_search_str;
            if (net_search_res.empty()) {
                net_search_str = fmt::format("你联网搜索了{}, 但是没有搜到任何东西。", search_text);
            } else {
                net_search_str += "以下是联网查询的结果，请在回答的时候附上相关链接\n";
                for (const auto res : net_search_res) {
                    net_search_str.append(fmt::format("{},{}:{}\n", res.url, res.title, res.content));
                }
            }
            MiraiCP::Logger::logger.info(net_search_str);
            *context.msg_prop.plain_content = replace_str(search, "#联网", net_search_str);
            process_llm(context);
        }).detach();

        return bot_cmd::CommandRes{true};
    }

        /**
     * @brief Handles a network search command based on the provided context.
     *
     * This function processes a network search command by extracting the search query from the message context.
     * If no valid search query is found, it prompts the user to provide one. Otherwise, it initiates a
     * background thread to perform the network search and process the results.
     *
     * @param context The command context containing message properties and sender information.
     * @return bot_cmd::CommandRes Returns a bot_cmd::CommandRes object indicating the success of the command handling.
     */
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
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};
            msg_chain.add(MiraiCP::PlainText{" 请输入查询。"});
            context.e.group.sendMessage();
            return bot_cmd::CommandRes{true};
        }
        std::thread([context, search]() {
            MiraiCP::Logger::logger.info("Start url search thread");
            auto search_text = extract_parentheses_content_after_keyword(search, "#url");

            std::vector<std::string> url_list;
            for (const auto url : SplitString(search_text, ',')) {
                auto u = std::string{url};
                MiraiCP::Logger::logger.info("URL: " + u);
                url_list.emplace_back(u);
            }

            auto net_search_res = rag::url_search_content(url_list);
            MiraiCP::Logger::logger.info(net_search_res);
            *context.msg_prop.plain_content = replace_keyword_and_parentheses_content(search, "#url", net_search_res);
            process_llm(context);
        }).detach();

        return bot_cmd::CommandRes{true};
    }
} // namespace bot_cmd