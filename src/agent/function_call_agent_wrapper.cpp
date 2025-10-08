#include "agent/function_call_agent_wrapper.h"
#include "agent/agent.h"
#include "get_optional.hpp"

namespace agent {

    using std::optional;

    bool FunctionCallAgentWrapper::validate_function_tool_impled(const nlohmann::json &func_call_ary) {
        for (const auto &item : func_call_ary) {
            const optional<std::string> &name = get_optional(item, "name");
            if (!name.has_value()) {
                spdlog::error("FunctionCallAgentWrapper: Function call item missing 'name' field: {}", item.dump());
                return false;
            }
            if (!function_map.contains(*name)) {
                spdlog::error("FunctionCallAgentWrapper: No implementation found for function '{}'", *name);
                return false;
            }
        }
        return true;
    }

    std::vector<ChatMessage> FunctionCallAgentWrapper::inference(const AgentInferenceParam &param) {
        if (!param.function_tools_opt.has_value() && !validate_function_tool_impled(*param.function_tools_opt)) {
            spdlog::error("FunctionCallAgentWrapper: Validation of function tools failed.");
            return {};
        }

        auto response_opt = llm_agent->inference(param);
        if (!response_opt.has_value()) {
            spdlog::error("FunctionCallAgentWrapper: LLM agent inference returned no response.");
            return {};
        }
        std::vector<ChatMessage> responses;

        responses.push_back(*response_opt);
        while (responses.rbegin()->tool_calls) {
            const auto &llm_res = *responses.rbegin();
            std::vector<ChatMessage> append_tool_calls;

            for (const auto &func_calls : *llm_res.tool_calls) {
                spdlog::info("Tool calls: {}()", func_calls.name, func_calls.arguments);
                std::optional<ChatMessage> tool_call_msg = std::nullopt;
                
                const auto &func = function_map[func_calls.name];
                
                auto res = func(func_calls.arguments);
                if (res.has_value()) {
                    
                }

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
        }
    }

    return responses;
}

} // namespace agent