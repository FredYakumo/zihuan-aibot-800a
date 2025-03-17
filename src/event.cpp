#include "MiraiCP.hpp"
#include "config.h"
#include "constants.hpp"
#include "msg_db.h"
#include "nlohmann/json_fwd.hpp"
#include "plugin.h"
#include "utils.h"
#include <charconv>
#include <cpr/cpr.h>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fmt/format.h>
#include <global_data.h>
#include <iterator>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

void remove_text_between_markers(std::string &str, const std::string &start_marker, const std::string &end_marker) {
    size_t start_pos = str.find(start_marker);
    size_t end_pos = str.find(end_marker);

    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
        str.erase(start_pos, end_pos - start_pos + end_marker.length());
    }
}

nlohmann::json msg_list_to_json(const std::string_view system_prompt, const std::deque<ChatMessage> &msg_list) {
    nlohmann::json msg_json = nlohmann::json::array();
    // 将消息队列中的消息添加到 JSON 数组中
    msg_json.push_back(nlohmann::json{{"role", "system"}, {"content", system_prompt}});
    for (const auto &msg : msg_list) {
        nlohmann::json msg_entry;
        msg_entry["role"] = msg.role;
        msg_entry["content"] = msg.content;
        MiraiCP::Logger::logger.info(msg_entry);
        msg_json.push_back(msg_entry);
    }
    return msg_json;
}

inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, const ChatMessage &msg) {
    msg_json.push_back({{"role", msg.role}, {"content", msg.content}});
    return msg_json;
}

ChatMessage get_llm_response(const nlohmann::json &msg_json) {

    nlohmann::json body = {{"model", LLM_MODEL_NAME}, {"messages", msg_json}, {"stream", false}};
    const auto json_str = body.dump();
    MiraiCP::Logger::logger.info("llm body: " + json_str);
    cpr::Response response =
        cpr::Post(cpr::Url{LLM_API_URL}, cpr::Body{json_str},
                  cpr::Header{{"Content-Type", "application/json"}, {"Authorization", LLM_API_TOKEN}});
    MiraiCP::Logger::logger.info(response.error.message);
    MiraiCP::Logger::logger.info(response.status_code);

    try {
        MiraiCP::Logger::logger.info(response.text);
        auto json = nlohmann::json::parse(response.text);
        std::string result = std::string(ltrim(json["choices"][0]["message"]["content"].get<std::string_view>()));
        remove_text_between_markers(result, "<think>", "</think>");

        return ChatMessage(ROLE_ASSISTANT, result);
    } catch (const std::exception &e) {
        MiraiCP::Logger::logger.info(std::string{"JSON 解析失败: "} + e.what());
        throw e;
    }
}

nlohmann::json get_msg_json(const std::string_view system_prompt, const MiraiCP::QQID id, const std::string_view name) {
    {
        auto session = g_chat_session_map.read();
        if (auto iter = session->find(id); iter != session->cend()) {
            return msg_list_to_json(system_prompt, iter->second.message_list);
        }
    }
    auto session = g_chat_session_map.write()->insert({id, ChatSession(name)});
    return msg_list_to_json(system_prompt, session.first->second.message_list);
}

inline bool try_begin_processing_llm(MiraiCP::QQID id) {
    std::lock_guard lock(g_chat_processing_map.first);
    if (auto it = g_chat_processing_map.second.find(id); it != std::cend(g_chat_processing_map.second)) {
        if (it->second) {
            return false;
        }
        it->second = true;
        return true;
    }
    g_chat_processing_map.second.emplace(id, true);
    return true;
}

inline void release_processing_llm(MiraiCP::QQID id) {
    std::lock_guard lock(g_chat_processing_map.first);
    g_chat_processing_map.second[id] = false;
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

    insert_group_msg(group_id, sender_name, sender_id, sender_name, msg_content);
}

void output_in_split_string(const MiraiCP::GroupMessageEvent &e, const MiraiCP::QQID target,
                            const std::string_view content) {
    auto split_output = Utf8Splitter(content, MAX_OUTPUT_LENGTH);
    size_t output_number = 0;
    for (auto chunk : split_output) {
        MiraiCP::Logger::logger.info(fmt::format("正在输出块: {}", output_number));
        auto msg_chain = MiraiCP::MessageChain{};
        if (output_number == 0) {
            msg_chain.add(MiraiCP::At(target));
            msg_chain.add(MiraiCP::PlainText(" "));
        }
        ++output_number;
        msg_chain.add(MiraiCP::PlainText{std::string(chunk)});
        e.group.sendMessage(msg_chain);
    }
}

void AIBot::onEnable() {
    // MiraiCP::Event::registerEvent<MiraiCP::BotOnlineEvent>([](MiraiCP::BotOnlineEvent e) {
    //     init_config();
    // });
    init_config();

    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([](MiraiCP::GroupMessageEvent e) {
        // MiraiCP::Logger::logger.info("Recv message: " + e.message.toString());
        auto sender_id = e.sender.id();

        if (sender_id == 2113328) {
            return;
        }
        const auto msg_prop = get_msg_prop_from_event(e);
        const auto group_id = e.group.id();
        const auto sender_name = e.sender.nickOrNameCard();
        const auto admin = is_admin(sender_id);

        if (!msg_prop.is_at_me) {
            return;
        }

        MiraiCP::Logger::logger.info("开始处理指令信息");

        auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};
        /*
        语录功能暂时不考虑递归
         */
        if (msg_prop.plain_content != nullptr && !msg_prop.plain_content->empty()) {
            if (auto quoted_content = extract_quoted_content(*msg_prop.plain_content, "#语录");
                !quoted_content.empty()) {
                std::string res{};
                const auto group_msg = query_group_msg(quoted_content, group_id);
                if (group_msg.empty()) {
                    msg_chain.add(MiraiCP::PlainText{fmt::format(" 在本群中未找到关于\"{}\"的语录", quoted_content)});
                    e.group.sendMessage(msg_chain);
                    return;
                }
                for (const auto &e : group_msg) {
                    res.append(fmt::format("\n{}: {}。时间: {}, 关联度: {:.4f}", e.first.sender_name, e.first.content,
                                           e.first.send_time, e.second));
                }

                if (is_strict_format(*msg_prop.plain_content, "#语录")) {
                    output_in_split_string(e, sender_id, res);
                    return;
                } else {
                    *msg_prop.plain_content = replace_quoted_content(*msg_prop.plain_content, "#语录",
                                                                     fmt::format("在本群中关于: {}的消息", res));
                }
            } else if (auto quoted_content = extract_quoted_content(*msg_prop.plain_content, "#知识搜索");
                !quoted_content.empty()) {
                std::string res{};
                const auto query_msg = query_knowledge(quoted_content);
                if (query_msg.empty()) {
                    msg_chain.add(MiraiCP::PlainText{fmt::format(" 未找到关于\"{}\"的数据", quoted_content)});
                    e.group.sendMessage(msg_chain);
                    return;
                }
                for (const auto &e : query_msg) {
                    res.append(fmt::format("\n{}。创建者: {}, 时间: {}, 关联度: {:.4f}",  e.first.content, e.first.creator_name,
                                           e.first.create_dt, e.second));
                }

                output_in_split_string(e, sender_id, res);
                return;

            } else if (auto quoted_content = extract_quoted_content(*msg_prop.plain_content, "#添加知识");
                       !quoted_content.empty()) {
                MiraiCP::Logger::logger.info(
                    fmt::format("{} 添加了知识到待添加列表中: {}", sender_name, quoted_content));
                std::thread([sender_name, sender_id, quoted_content, e] {
                    std::string content{quoted_content};
                    auto wait_add_list = g_wait_add_knowledge_list.write();
                    wait_add_list->emplace_back(DBKnowledge{content, sender_name});
                    auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};
                    msg_chain.add(MiraiCP::PlainText{" 成功, 添加1条知识到待添加列表中。"});
                    e.group.sendMessage(msg_chain);
                }).detach();
                return;
            } else if (auto quoted_content = extract_quoted_content(*msg_prop.plain_content, "#CheckIn知识");
                       !quoted_content.empty() && admin) {
                size_t index = 0;
                auto [ptr, ec] =
                    std::from_chars(quoted_content.data(), quoted_content.data() + quoted_content.size(), index);
                if (ec != std::errc() && ptr != quoted_content.data() + quoted_content.size()) {
                    msg_chain.add(MiraiCP::PlainText{" 错误。用法: #CheckIn知识 (id: number)"});
                    e.group.sendMessage(msg_chain);
                    return;
                }
                MiraiCP::Logger::logger.info(fmt::format("Index = {}", index));

                std::thread([e, index, sender_id] {
                    MiraiCP::Logger::logger.info("Start add knowledge thread.");
                    auto wait_add_list = g_wait_add_knowledge_list.write();
                    auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};

                    if (index >= wait_add_list->size()) {
                        msg_chain.add(MiraiCP::PlainText{fmt::format(" 错误。id {} 不存在于待添加列表中", index)});
                        e.group.sendMessage(msg_chain);
                        return;
                    }
                    insert_knowledge(wait_add_list->at(index));
                    wait_add_list->erase(wait_add_list->cbegin() + index);
                    msg_chain.add(MiraiCP::PlainText{fmt::format(" CheckIn知识成功。列表剩余{}条。", wait_add_list->size())});
                    e.group.sendMessage(msg_chain);
                }).detach();
                return;
            } else {
                auto msg_storage_thread = std::thread([msg_prop, group_id, sender_id, sender_name] {
                    set_thread_name("AIBot msg storage");
                    MiraiCP::Logger::logger.info("Start message storage thread.");
                    msg_storage(msg_prop, group_id, sender_id, sender_name);
                });

                msg_storage_thread.detach();
            }
        }

        if (msg_prop.plain_content != nullptr && admin) {
            if (*msg_prop.plain_content == "#记忆查看") {
                std::string memory_str = fmt::format(" '{}'当前记忆列表:\n", e.bot.nick());
                auto chat_session_map = g_chat_session_map.read();
                if (chat_session_map->empty()) {
                    memory_str += "空的";
                } else {
                    for (auto entry : *chat_session_map) {
                        memory_str += fmt::format("QQ号: {}, 昵称: {}, 记忆数: {}\n", entry.first,
                                                  entry.second.nick_name, entry.second.user_msg_count);
                        for (auto m : entry.second.message_list) {
                            auto role = m.role;
                            if (role == "user") {
                                role = entry.second.nick_name;
                            } else {
                                role = e.bot.nick();
                            }
                            memory_str +=
                                fmt::format("\t- [{}] {}: {}\n", m.get_formatted_timestamp(), role, m.content);
                        }
                    }
                }
                output_in_split_string(e, sender_id, memory_str);
                return;
            } else if (*msg_prop.plain_content == "#待添加知识列表") {
                auto wait_add_list = g_wait_add_knowledge_list.read();
                if (wait_add_list->empty()) {
                    msg_chain.add(MiraiCP::PlainText{"暂无待添加知识。"});
                    e.group.sendMessage(msg_chain);
                    return;
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
                e.group.sendMessage(msg_chain);
                return;
            }
        }

        MiraiCP::Logger::logger.info("开始处理LLM信息");

        if (!try_begin_processing_llm(sender_id)) {
            MiraiCP::Logger::logger.warning(
                fmt::format("User {} try to let bot answer, but bot is still thiking", e.sender.id()));
            msg_chain.add((MiraiCP::PlainText{" 我还在思考中...你别急"}));
            e.group.sendMessage(msg_chain);
            return;
        }

        std::string msg_content_str{};
        if (msg_prop.ref_msg_content != nullptr) {
            msg_content_str.append(fmt::format("我引用了一个消息: {}\n", *msg_prop.ref_msg_content));
        }
        msg_content_str.append(*msg_prop.plain_content);

        auto bot_name = e.bot.nick();
        auto bot_id = e.bot.id();

        auto llm_thread = std::thread([e, msg_content_str, sender_id, bot_name, bot_id] {
            set_thread_name("AIBot LLM process");
            MiraiCP::Logger::logger.info("Start llm thread.");

            auto system_prompt = gen_common_prompt(bot_name, bot_id, e.sender.nickOrNameCard(), sender_id);

            MiraiCP::Logger::logger.info("Try query knowledge");
            auto knowledge_list = query_knowledge(msg_content_str);
            if (!knowledge_list.empty()) {
            std::string query_result_str = "\n以下是相关知识:";
                for (const auto &knowledge : knowledge_list) {
                    query_result_str.append(fmt::format("\n{},关联度:{:.4f}", 
                    knowledge.first.content, knowledge.second));
                }
                MiraiCP::Logger::logger.info(query_result_str);
                system_prompt += query_result_str;
            } else {
                MiraiCP::Logger::logger.info("未查询到关联的知识");
            }

            auto msg_json = get_msg_json(system_prompt,
                                         e.sender.id(), e.sender.nickOrNameCard());

            auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
            add_to_msg_json(msg_json, user_chat_msg);

            auto llm_chat_msg = get_llm_response(msg_json);
            auto session_map = g_chat_session_map.write();
            auto &session = session_map->find(sender_id)->second;
            if (session.user_msg_count + 1 >= USER_SESSION_MSG_LIMIT) {
                session.message_list.pop_front();
                session.message_list.pop_front();
            } else {
                ++session.user_msg_count;
            }

            session.message_list.push_back(user_chat_msg);
            session.message_list.emplace_back(llm_chat_msg.role, llm_chat_msg.content);

            MiraiCP::Logger::logger.info("Prepare to send msg response");

            output_in_split_string(e, sender_id, llm_chat_msg.content);

            release_processing_llm(sender_id);
        });

        llm_thread.detach();
    });
}