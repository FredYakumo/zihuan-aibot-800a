#include "llm.h"
#include "adapter_message.h"
#include "bot_cmd.h"
#include "config.h"
#include "constants.hpp"
#include "global_data.h"
#include "rag.h"
#include "utils.h"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <string>

std::string gen_common_prompt(const bot_adapter::Profile &bot_profile, const bot_adapter::Sender &sender,
                              bool is_deep_think) {
    return fmt::format(
        "你的名字叫{}(qq号{}),性别是: {}，{}。当前时间是: {}，当前跟你聊天的群友的名字叫\"{}\"(qq号{})，",
        bot_profile.name, bot_profile.id, bot_adapter::to_chs_string(bot_profile.sex),
        (is_deep_think && CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION) ? *CUSTOM_DEEP_THINK_SYSTEM_PROMPT_OPTION
                                                                  : CUSTOM_SYSTEM_PROMPT,
        get_current_time_formatted(), sender.name, sender.id);
}

ChatMessage get_llm_response(const nlohmann::json &msg_json, bool is_deep_think = false) {
    nlohmann::json body = {{"model", is_deep_think ? LLM_DEEP_THINK_MODEL_NAME : LLM_MODEL_NAME},
                           {"messages", msg_json},
                           {"stream", false}};
    const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    spdlog::info("llm body: {}", json_str);
    cpr::Response response =
        cpr::Post(cpr::Url{LLM_API_URL}, cpr::Body{json_str},
                  cpr::Header{{"Content-Type", "application/json"}, {"Authorization", LLM_API_TOKEN}});
    spdlog::info("Error msg: {}, status code: {}", response.error.message, response.status_code);

    try {
        spdlog::info(response.text);
        auto json = nlohmann::json::parse(response.text);
        std::string result = std::string(ltrim(json["choices"][0]["message"]["content"].get<std::string_view>()));
        remove_text_between_markers(result, "<think>", "</think>");

        return ChatMessage(ROLE_ASSISTANT, result);
    } catch (const std::exception &e) {
        spdlog::error("JSON 解析失败: {}", e.what());
    }
    return ChatMessage();
}

std::string query_chat_session_knowledge(const bot_cmd::CommandContext &context,
                                         const std::string_view &msg_content_str) {
    // Search knowledge for this chat message
    spdlog::info("Search knowledge for this chat message");
    auto msg_knowledge_list = rag::query_knowledge(msg_content_str);
    std::string chat_use_knowledge_str;
    {
        auto map = g_chat_session_knowledge_list_map.write();
        auto user_set_iter = map->find(context.e->sender_ptr->id);
        if (user_set_iter == map->cend()) {
            user_set_iter = map->insert(std::make_pair(context.e->sender_ptr->id, std::set<std::string>())).first;
        }
        for (const auto &knowledge : msg_knowledge_list) {
            if (knowledge.content.empty()) {
                continue;
            }
            if (user_set_iter->second.size() >= MAX_KNOWLEDGE_COUNT) {
                spdlog::info("{}({})的对话session知识数量超过限制, 删除最旧的知识", context.e->sender_ptr->name,
                             context.e->sender_ptr->id);
                user_set_iter->second.erase(user_set_iter->second.cbegin());
            }
            user_set_iter->second.insert(knowledge.content);
        }

        const auto user_chat_knowledge_list = map->find(context.e->sender_ptr->id);
        if (user_chat_knowledge_list == map->cend() || user_chat_knowledge_list->second.empty()) {
            spdlog::info("未查询到对话关联的知识");
        } else {
            chat_use_knowledge_str.append("当前对话相关的知识:\"");
            chat_use_knowledge_str.append(join_str(std::cbegin(user_chat_knowledge_list->second),
                                                   std::cend(user_chat_knowledge_list->second), "\n"));
            chat_use_knowledge_str.append("\"");
        }
    }

    // Search knowledge for username
    spdlog::info("Search knowledge for username");
    auto sender_name_knowledge_list = rag::query_knowledge(context.e->sender_ptr->name, true);
    std::string sender_name_knowledge_str;
    if (sender_name_knowledge_list.empty()) {
        spdlog::info("未查询到用户关联的知识");
    } else {
        sender_name_knowledge_str = "与你聊天用户相关的信息:\"";
        sender_name_knowledge_str.append(join_str(std::cbegin(sender_name_knowledge_list),
                                                  std::cend(sender_name_knowledge_list), "\n",
                                                  [](const auto &knowledge) { return knowledge.content; }));
        sender_name_knowledge_str.append("\"");
    }

    return chat_use_knowledge_str + sender_name_knowledge_str;
}

void process_llm(const bot_cmd::CommandContext &context,
                 const std::optional<std::string> &additional_system_prompt_option) {
    spdlog::info("开始处理LLM信息");

    if (!try_begin_processing_llm(context.e->sender_ptr->id)) {
        spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.e->sender_ptr->id);
        context.adapter.send_replay_msg(
            *context.e->sender_ptr,
            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("我还在思考中...你别急")));
        return;
    }

    spdlog::debug("Event type: {}, Sender json: {}", context.e->get_typename(),
                  context.e->sender_ptr->to_json().dump());

    std::string msg_content_str{};
    if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
        msg_content_str.append(fmt::format("我引用了一个消息: {}\n", *context.msg_prop.ref_msg_content));
    }
    if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
        msg_content_str.append(*context.msg_prop.plain_content);
    }
    spdlog::info(msg_content_str);

    auto llm_thread = std::thread([context, msg_content_str, additional_system_prompt_option] {
        spdlog::debug("Event type: {}, Sender json: {}", context.e->get_typename(),
                      context.e->sender_ptr->to_json().dump());

        spdlog::info("Start llm thread.");
        set_thread_name("AIBot LLM process");
        if (context.is_deep_think) {
            spdlog::info("开始深度思考");
        }
        const auto bot_profile = context.adapter.get_bot_profile();
        auto system_prompt = gen_common_prompt(bot_profile, *context.e->sender_ptr, context.is_deep_think);

        system_prompt += "\n" + query_chat_session_knowledge(context, msg_content_str);

        if (additional_system_prompt_option.has_value()) {
            system_prompt += additional_system_prompt_option.value();
        }

        auto msg_json = get_msg_json(system_prompt, context.e->sender_ptr->id, context.e->sender_ptr->name);

        auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
        add_to_msg_json(msg_json, user_chat_msg);

        auto llm_chat_msg = get_llm_response(msg_json, context.is_deep_think);

        auto session_map = g_chat_session_map.write();
        auto &session = session_map->find(context.e->sender_ptr->id)->second;

        if (session.user_msg_count + 1 >= USER_SESSION_MSG_LIMIT) {
            session.message_list.pop_front();
            session.message_list.pop_front();
        } else {
            ++session.user_msg_count;
        }

        session.message_list.push_back(user_chat_msg);
        session.message_list.emplace_back(llm_chat_msg.role, llm_chat_msg.content);

        spdlog::info("Prepare to send msg response");

        context.adapter.send_long_plain_text_replay(*context.e->sender_ptr, llm_chat_msg.content);

        release_processing_llm(context.e->sender_ptr->id);
    });

    llm_thread.detach();
}