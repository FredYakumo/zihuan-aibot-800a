#include "agent/function_call_agent_wrapper.h"

namespace agent {

    FunctionCallAgentWrapper::~FunctionCallAgentWrapper() = default;

    std::optional<ChatMessage> FunctionCallAgentWrapper::inference(const AgentInferenceParam &param) {
        auto response_opt = llm_agent->inference(param);
        if (!response_opt.has_value()) {
            spdlog::error("FunctionCallAgentWrapper: LLM agent inference returned no response.");
            return std::nullopt;
        }

        const auto &response = response_opt.value();
        if (response.function_call.has_value()) {
            const auto &func_call = response.function_call.value();
            const auto &func_name = func_call.name;
            const auto &func_args = func_call.arguments;

            auto it = function_map.find(func_name);
            if (it != function_map.end()) {
                try {
                    it->second(func_args);
                } catch (const std::exception &e) {
                    spdlog::error("FunctionCallAgentWrapper: Exception while executing function '{}': {}",
                                  func_name, e.what());
                }
            } else {
                spdlog::warn("FunctionCallAgentWrapper: No implementation found for function '{}'", func_name);
            }
        }

        return response;
    }

} // namespace agent