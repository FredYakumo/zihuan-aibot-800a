#include "agent/simple_chat_action_agent.h"
#include "agent/llm.h" // for gen_common_prompt
#include "event.h"     // try_to_replay_person/release_processing_replay_person
#include "global_data.h"
#include "rag.h"
#include "think_image_manager.h"
#include "tool_impl/common.hpp"
#include "tool_impl/fetch_url_content.hpp"
#include "tool_impl/get_function_list.hpp"
#include "tool_impl/query_group.hpp"
#include "tool_impl/search_info.hpp"
#include "tool_impl/view_chat_history.hpp"
#include "tool_impl/view_model_info.hpp"
#include "utils.h"
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>
#include <thread>

// using wheel::join_str; // unused here
// using wheel::ltrim;    // unused here
// using wheel::rtrim;    // unused here

namespace agent {

    // get_permission_chs moved to tool_impl/common.hpp

    // helper conversions (duplicated from llm.cpp; consider refactor later)
    inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, ChatMessage msg) {
        msg_json.push_back(msg.to_json());
        return msg_json;
    }

    inline nlohmann::json msg_list_to_json(const std::deque<ChatMessage> &msg_list,
                                           const std::optional<std::string_view> system_prompt_option = std::nullopt) {
        nlohmann::json msg_json = nlohmann::json::array();
        if (system_prompt_option.has_value()) {
            msg_json.push_back(nlohmann::json{{"role", "system"}, {"content", system_prompt_option.value()}});
        }
        for (const auto &msg : msg_list) {
            nlohmann::json msg_entry = msg.to_json();
            spdlog::info("msg_entry: {}", msg_entry.dump());
            msg_json.push_back(std::move(msg_entry));
        }
        return msg_json;
    }

    inline nlohmann::json get_msg_json(const qq_id_t id, std::string name,
                                       const std::optional<std::string> &system_prompt_option = std::nullopt) {
        auto session = g_chat_session_map.get_or_create_value(
            id, [name = std::move(name)]() mutable { return ChatSession(std::move(name)); });
        return msg_list_to_json(session->message_list, system_prompt_option);
    }

    void insert_tool_call_record_async(const std::string &sender_name, qq_id_t sender_id,
                                       const nlohmann::json &msg_json, const std::string &func_name,
                                       const std::string &func_arguments, const std::string &tool_content) {
        std::thread([=] {
            set_thread_name("insert tool call record");
            spdlog::info("Start insert tool call record thread.");
            database::get_global_db_connection().insert_tool_calls_record(
                sender_name, sender_id, msg_json.dump(), std::chrono::system_clock::now(),
                fmt::format("{}({})", func_name, func_arguments), tool_content);
        }).detach();
    }

    void SimpleChatActionAgent::process_llm(const bot_cmd::CommandContext &context,
                                            const std::optional<std::string> &additional_system_prompt_option,
                                            const std::optional<database::UserPreference> &user_preference_option,
                                            const std::optional<nlohmann::json> &function_tools_opt) {
        spdlog::info("[SimpleChatActionAgent] 开始处理LLM信息");

        if (!try_to_replay_person(context.event->sender_ptr->id)) {
            spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.event->sender_ptr->id);
            adapter->send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("我还在思考中...你别急")));
            return;
        }

        spdlog::debug("Event type: {}, Sender json: {}", context.event->get_typename(),
                      context.event->sender_ptr->to_json().dump());

        std::string llm_content{};
        if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
            llm_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                           *context.msg_prop.ref_msg_content));
        }
        if (!context.msg_prop.at_id_set.empty() &&
            !(context.msg_prop.at_id_set.size() == 1 &&
              *context.msg_prop.at_id_set.begin() == adapter->get_bot_profile().id)) {
            llm_content.append("本消息提到了：");
            bool first = true;
            for (const auto &at_id : context.msg_prop.at_id_set) {
                if (!first) {
                    llm_content.append("、");
                }
                llm_content.append(fmt::format("{}", at_id));
                first = false;
            }
            llm_content.append("。");
        }
        std::string speak_content = EMPTY_MSG_TAG;
        if (context.msg_prop.plain_content != nullptr &&
            !wheel::ltrim(wheel::rtrim(*context.msg_prop.plain_content)).empty()) {
            speak_content = *context.msg_prop.plain_content;
        }

        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            if (auto group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr);
                group_sender.has_value()) {
                llm_content.append(fmt::format("{}\"{}\"({})[{}]对你说: \"{}\"",
                                               tool_impl::get_permission_chs(group_sender->get().permission),
                                               group_sender->get().name, context.event->sender_ptr->id,
                                               get_current_time_formatted(), speak_content));
            } else {
                llm_content.append(fmt::format("\"{}\"({})[{}]对你说: \"{}\"", context.event->sender_ptr->name,
                                               context.event->sender_ptr->id, get_current_time_formatted(),
                                               speak_content));
            }
        }
        std::string mixed_input_content;
        if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
            mixed_input_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                                   *context.msg_prop.ref_msg_content));
        }
        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            mixed_input_content.append(fmt::format("{}", *context.msg_prop.plain_content));
        }
        auto session_knowledge_opt = rag::query_knowledge(mixed_input_content, false, context.event->sender_ptr->id,
                                                          context.event->sender_ptr->name);
        std::string system_prompt = gen_common_prompt(adapter->get_bot_profile(), *adapter, *context.event->sender_ptr,
                                                      context.is_deep_think, additional_system_prompt_option);

        // Add session knowledge to system prompt if available
        if (session_knowledge_opt.has_value()) {
            system_prompt += "\n" + session_knowledge_opt.value();
        }

        spdlog::info("作为用户输入给llm的content: {}", llm_content);

        auto llm_thread =
            std::thread([this, context, llm_content, system_prompt, user_preference_option, function_tools_opt] {
                on_llm_thread(context, llm_content, system_prompt, user_preference_option, function_tools_opt);
            });

        llm_thread.detach();
    }

    // get_target_group_chat_history was moved to tool_impl/view_chat_history.hpp

    void SimpleChatActionAgent::process_tool_calls(const bot_cmd::CommandContext &context, nlohmann::json &msg_json,
                                                   std::vector<ChatMessage> &one_chat_session,
                                                   const std::optional<nlohmann::json> &function_tools_opt) {

        std::vector<bot_adapter::ForwardMessageNode> first_replay;

        // loop check if have function call
        while (one_chat_session.rbegin()->tool_calls) {
            const auto &llm_res = *one_chat_session.rbegin();

            // llm request tool call
            add_to_msg_json(msg_json, llm_res);
            std::vector<ChatMessage> append_tool_calls;
            for (const auto &func_calls : *llm_res.tool_calls) {
                spdlog::info("Tool calls: {}()", func_calls.name, func_calls.arguments);
                std::optional<ChatMessage> tool_call_msg = std::nullopt;
                if (func_calls.name == "search_info") {
                    tool_call_msg = tool_impl::search_info(context, func_calls, first_replay);
                } else if (func_calls.name == "fetch_url_content") {
                    tool_call_msg = tool_impl::fetch_url_content(context, func_calls, llm_res);
                } else if (func_calls.name == "view_model_info") {
                    tool_call_msg = tool_impl::view_model_info(func_calls);
                } else if (func_calls.name == "view_chat_history") {
                    tool_call_msg = tool_impl::view_chat_history(context, func_calls);
                } else if (func_calls.name == "query_group") {
                    tool_call_msg = tool_impl::query_group(context, func_calls);
                } else if (func_calls.name == "get_function_list") {
                    tool_call_msg = tool_impl::get_function_list(func_calls);
                } else {
                    spdlog::error("Function {} is not impl.", func_calls.name);
                    tool_call_msg = ChatMessage(ROLE_TOOL,
                                                "主人还没有实现这个功能,快去github页面( "
                                                "https://github.com/FredYakumo/zihuan-aibot-800a )提issues吧",
                                                func_calls.id);
                }

                if (tool_call_msg.has_value()) {
                    insert_tool_call_record_async(context.event->sender_ptr->name, context.event->sender_ptr->id,
                                                  msg_json, func_calls.name, func_calls.arguments,
                                                  tool_call_msg->content);
                    append_tool_calls.emplace_back(std::move(*tool_call_msg));
                }
            }
            if (!append_tool_calls.empty()) {
                for (auto &append : append_tool_calls) {
                    add_to_msg_json(msg_json, append);
                    one_chat_session.emplace_back(std::move(append));
                }
                // one_chat_session.insert(std::end(one_chat_session),
                //                             std::make_move_iterator(std::begin(append_tool_calls)),
                //                             std::make_move_iterator(std::end(append_tool_calls)));
            }

            AgentInferenceParam inference_param;
            inference_param.think_mode = context.is_deep_think;
            inference_param.messages_json = msg_json;
            inference_param.function_tools_opt = function_tools_opt;
            if (auto llm_res_new = bind_output_llm_agent->inference(inference_param); llm_res_new.has_value()) {
                one_chat_session.push_back(std::move(*llm_res_new));
            } else {
                spdlog::warn("LLM did not response any chat message...");
                // Clear the user's chat session and knowledge when LLM doesn't respond
                g_chat_session_map.erase(context.event->sender_ptr->id);
                g_chat_session_knowledge_list_map.erase(context.event->sender_ptr->id);
                spdlog::info("Cleared context and knowledge for user {} due to LLM failure",
                             context.event->sender_ptr->id);
                context.adapter.send_replay_msg(*context.event->sender_ptr, bot_adapter::make_message_chain_list(
                                                                                bot_adapter::PlainTextMessage("?")));
                release_processing_replay_person(context.event->sender_ptr->id);

                return;
            }
        }

        if (!first_replay.empty()) {
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::ForwardMessage(
                                                first_replay, bot_adapter::DisplayNode(std::string("联网搜索结果")))));
        }
    }

    void SimpleChatActionAgent::on_llm_thread(const bot_cmd::CommandContext &context, const std::string &llm_content,
                                              const std::string &system_prompt,
                                              const std::optional<database::UserPreference> &user_preference_option,
                                              const std::optional<nlohmann::json> &function_tools_opt) {
        set_thread_name(fmt::format("llm thread for {}", context.event->sender_ptr->id).c_str());
        spdlog::info("llm thread for {} started", context.event->sender_ptr->id);

        if (context.is_deep_think) {
            // get a random thinking image from the ThinkImageManager
            std::string image_path = bot_adapter::ThinkImageManager::instance().get_random_image_path();
            
            // If image_path is empty, only send text without image
            if (image_path.empty()) {
                context.adapter.send_replay_msg(
                    *context.event->sender_ptr,
                    bot_adapter::make_message_chain_list(
                        bot_adapter::PlainTextMessage{"正在思思考中..."}),
                    false);
            } else {
                // Check if the image is a URL or a local file path
                bool is_url = (image_path.find("http://") == 0 || image_path.find("https://") == 0);
                
                context.adapter.send_replay_msg(
                    *context.event->sender_ptr,
                    bot_adapter::make_message_chain_list(
                        bot_adapter::PlainTextMessage{"正在思思考中..."},
                        is_url ? static_cast<std::shared_ptr<bot_adapter::MessageBase>>(
                                    std::make_shared<bot_adapter::ImageMessage>(image_path))
                               : static_cast<std::shared_ptr<bot_adapter::MessageBase>>(
                                    std::make_shared<bot_adapter::LocalImageMessage>(image_path))),
                    false);
            }
        }
        auto session = g_chat_session_map.get_or_create_value(
            context.event->sender_ptr->id,
            [name = context.event->sender_ptr->name]() mutable { return ChatSession(std::move(name)); });

        session->message_list.emplace_back(ROLE_USER, llm_content);
        auto msg_json = msg_list_to_json(session->message_list, system_prompt);

        AgentInferenceParam inference_param;
        inference_param.think_mode = context.is_deep_think;
        inference_param.messages_json = msg_json;
        inference_param.function_tools_opt = function_tools_opt;

        std::vector<ChatMessage> one_chat_session;
        one_chat_session.emplace_back(ROLE_USER, llm_content);

        if (auto llm_res = bind_output_llm_agent->inference(inference_param); llm_res.has_value()) {
            one_chat_session.push_back(std::move(*llm_res));
        } else {
            spdlog::warn("LLM did not response any chat message...");
            // Clear the user's chat session and knowledge when LLM doesn't respond
            g_chat_session_map.erase(context.event->sender_ptr->id);
            g_chat_session_knowledge_list_map.erase(context.event->sender_ptr->id);
            spdlog::info("Cleared context and knowledge for user {} due to LLM failure", context.event->sender_ptr->id);
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("?")));
            release_processing_replay_person(context.event->sender_ptr->id);
            return;
        }

        process_tool_calls(context, msg_json, one_chat_session, function_tools_opt);

        const auto &final_res = one_chat_session.back();
        if (final_res.content.empty()) {
            spdlog::warn("LLM response content is empty");
        } else {
            spdlog::info("LLM response content: {}", final_res.content);
            context.adapter.send_long_plain_text_reply(*context.event->sender_ptr, final_res.content, true);
        }

        session->message_list.insert(session->message_list.end(), std::make_move_iterator(one_chat_session.begin()),
                                     std::make_move_iterator(one_chat_session.end()));

        release_processing_replay_person(context.event->sender_ptr->id);
        g_last_chat_message_time_map.insert_or_assign(context.event->sender_ptr->id, std::chrono::system_clock::now());
        spdlog::info("llm thread for {} finished", context.event->sender_ptr->id);
    }

} // namespace agent