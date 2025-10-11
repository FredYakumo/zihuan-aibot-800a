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
                    res->tool_call_id = func_calls.id;
                    append_tool_calls.push_back(std::move(*res));

                    // @TODO: Need fix
                    // insert_tool_call_record_async(param.)
                }
            }

            if (!append_tool_calls.empty()) {
                for (auto &append : append_tool_calls) {
                    responses.push_back(std::move(append));
                }
                responses.insert(std::cend(responses), std::make_move_iterator(std::begin(append_tool_calls)),
                                 std::make_move_iterator(std::end(append_tool_calls)));
            }
        }

        return responses;
    }

} // namespace agent