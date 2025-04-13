#include "llm.h"
#include "adapter_message.h"
#include "bot_cmd.h"
#include "config.h"
#include "constants.hpp"
#include "rag.h"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <string>

ChatMessage get_llm_response(const nlohmann::json &msg_json) {
    nlohmann::json body = {{"model", LLM_MODEL_NAME}, {"messages", msg_json}, {"stream", false}};
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

void process_llm(const bot_cmd::CommandContext &context, const std::optional<std::string> &additional_system_prompt_option) {
    spdlog::info("开始处理LLM信息");

    if (!try_begin_processing_llm(context.e->sender_ptr->id)) {
        spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.e->sender_ptr->id);
        context.adapter.send_replay_msg(*context.e->sender_ptr, bot_adapter::make_message_chain_list(
                                                            bot_adapter::PlainTextMessage("我还在思考中...你别急")));
        return;
    }

    spdlog::debug("Event type: {}, Sender json: {}", context.e->get_typename(), context.e->sender_ptr->to_json().dump());

    std::string msg_content_str{};
    if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
        msg_content_str.append(fmt::format("我引用了一个消息: {}\n", *context.msg_prop.ref_msg_content));
    }
    if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
        msg_content_str.append(*context.msg_prop.plain_content);
    }
    spdlog::info(msg_content_str);
    auto llm_thread = std::thread([context, msg_content_str, additional_system_prompt_option] {
        spdlog::debug("Event type: {}, Sender json: {}", context.e->get_typename(), context.e->sender_ptr->to_json().dump());

        spdlog::info("Start llm thread.");
        set_thread_name("AIBot LLM process");
        const auto bot_profile = context.adapter.get_bot_profile();
        auto system_prompt =
            gen_common_prompt(bot_profile.name, bot_profile.id, context.e->sender_ptr->name, context.e->sender_ptr->id);

        spdlog::info("Try query knowledge");
        std::string query_result_str{""};
        auto msg_knowledge_list = rag::query_knowledge(msg_content_str);
        auto sender_name_knowledge_list = rag::query_knowledge(context.e->sender_ptr->name);
        if (!msg_knowledge_list.empty()) {
            query_result_str += "\n以下是相关知识:";
            for (const auto &knowledge : msg_knowledge_list) {
                query_result_str.append(fmt::format("\n{}", knowledge.first.content));
            }
        }
        if (!sender_name_knowledge_list.empty()) {
            query_result_str += "\n以下是跟你聊天的用户的相关知识:";
            for (const auto &knowledge : sender_name_knowledge_list) {
                query_result_str.append(fmt::format("\n{}", knowledge.first.content));
            }
        }
        if (!query_result_str.empty()) {
            spdlog::info(query_result_str);
            system_prompt += query_result_str;
        } else {
            spdlog::info("未查询到关联的知识");
        }
        if (additional_system_prompt_option.has_value()) {
            system_prompt += additional_system_prompt_option.value();
        }
        auto msg_json = get_msg_json(system_prompt, context.e->sender_ptr->id, context.e->sender_ptr->name);

        auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
        add_to_msg_json(msg_json, user_chat_msg);

        auto llm_chat_msg = get_llm_response(msg_json);

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