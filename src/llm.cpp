#include "llm.h"
#include "constants.hpp"
#include "bot_cmd.h"
#include "rag.h"
#include <cpr/cpr.h>
#include "config.h"
#include <string>
#include <nlohmann/json.hpp>
#include <MiraiCP.hpp>

ChatMessage get_llm_response(const nlohmann::json &msg_json) {
    nlohmann::json body = {{"model", LLM_MODEL_NAME}, {"messages", msg_json}, {"stream", false}};
    const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
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
        MiraiCP::Logger::logger.error(std::string{"JSON 解析失败: "} + e.what());
    }
    return ChatMessage();
}

void process_llm(bot_cmd::CommandContext context) {
    MiraiCP::Logger::logger.info("开始处理LLM信息");

    if (!try_begin_processing_llm(context.sender_id)) {
        MiraiCP::Logger::logger.warning(
            fmt::format("User {} try to let bot answer, but bot is still thiking", context.sender_id));
        auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(context.sender_id)};

        msg_chain.add((MiraiCP::PlainText{" 我还在思考中...你别急"}));
        context.e.group.sendMessage(msg_chain);
        return;
    }

    std::string msg_content_str{};
    if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
        msg_content_str.append(fmt::format("我引用了一个消息: {}\n", *context.msg_prop.ref_msg_content));
    }
    if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
        msg_content_str.append(*context.msg_prop.plain_content);
    }
    MiraiCP::Logger::logger.info(msg_content_str);
    auto llm_thread = std::thread([context, msg_content_str] {
        MiraiCP::Logger::logger.info("Start llm thread.");
        set_thread_name("AIBot LLM process");
        auto system_prompt =
            gen_common_prompt(context.bot_name, context.bot_id, context.sender_name, context.sender_id);

        MiraiCP::Logger::logger.info("Try query knowledge");
        std::string query_result_str {""};
        auto msg_knowledge_list = rag::query_knowledge(msg_content_str);
        auto sender_name_knowledge_list = rag::query_knowledge(context.sender_name);
        if (!msg_knowledge_list.empty()) {
            query_result_str += "\n以下是相关知识:";
            for (const auto &knowledge : msg_knowledge_list) {
                query_result_str.append(fmt::format("\n{},关联度:{:.4f}", knowledge.first.content, knowledge.second));
            }
        }
        if (!sender_name_knowledge_list.empty())  {
            query_result_str += "\n以下是跟你聊天的用户的相关知识:";
            for (const auto &knowledge : sender_name_knowledge_list) {
                query_result_str.append(fmt::format("\n{},关联度:{:.4f}", knowledge.first.content, knowledge.second));
            }
        }
        if (!query_result_str.empty()) {
            MiraiCP::Logger::logger.info(query_result_str);
            system_prompt += query_result_str;
        } else {
            MiraiCP::Logger::logger.info("未查询到关联的知识");
        }
        auto msg_json = get_msg_json(system_prompt, context.sender_id, context.sender_name);

        auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
        add_to_msg_json(msg_json, user_chat_msg);

        auto llm_chat_msg = get_llm_response(msg_json);

        auto session_map = g_chat_session_map.write();
        auto &session = session_map->find(context.sender_id)->second;
        if (session.user_msg_count + 1 >= USER_SESSION_MSG_LIMIT) {
            session.message_list.pop_front();
            session.message_list.pop_front();
        } else {
            ++session.user_msg_count;
        }

        session.message_list.push_back(user_chat_msg);
        session.message_list.emplace_back(llm_chat_msg.role, llm_chat_msg.content);

        MiraiCP::Logger::logger.info("Prepare to send msg response");

        output_in_split_string(context.e, context.sender_id, llm_chat_msg.content);

        release_processing_llm(context.sender_id);
    });

    llm_thread.detach();
}