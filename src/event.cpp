#include "MiraiCP.hpp"
#include "config.h"
#include "constants.hpp"
#include "msg_db.h"
#include "nlohmann/json_fwd.hpp"
#include "plugin.h"
#include "utils.h"
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

void msg_storage(const MessageProperties &msg_prop, MiraiCP::QQID group_id, MiraiCP::QQID sender_id, const std::string_view sender_name) {
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
        
        auto msg_storage_thread = std::thread([msg_prop, group_id, sender_id, sender_name] {
            set_thread_name("AIBot msg storage");
            MiraiCP::Logger::logger.info("Start message storage thread.");
            msg_storage(msg_prop, group_id, sender_id, sender_name);
        });

        msg_storage_thread.detach();

        if (!msg_prop.is_at_me) {
            return;
        }

        auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};

        if (msg_prop.plain_content != nullptr && *msg_prop.plain_content == "#记忆查看" && sender_id == 3507578481) {
            std::string memory_str = fmt::format(" '{}'当前记忆列表:\n", e.bot.nick());
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
                            role = e.bot.nick();
                        }
                        memory_str += fmt::format("\t- [{}] {}: {}\n", m.get_formatted_timestamp(), role, m.content);
                    }
                }
            }
            msg_chain.add(MiraiCP::PlainText{memory_str});
            e.group.sendMessage(msg_chain);
            return;
        }

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
            auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};
            auto msg_json =
                get_msg_json(gen_common_prompt(bot_name, bot_id, e.sender.nickOrNameCard(), sender_id),
                             e.sender.id(), e.sender.nickOrNameCard());

            auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
            add_to_msg_json(msg_json, user_chat_msg);

            auto llm_chat_msg = get_llm_response(msg_json);
            msg_chain.add((MiraiCP::PlainText{" " + llm_chat_msg.content}));
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
            e.group.sendMessage(msg_chain);

            release_processing_llm(sender_id);
        });

        llm_thread.detach();
    });
}