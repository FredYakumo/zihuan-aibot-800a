#pragma once

// Need to impl a class ,wrapping function calling logic in llm agent class

#include "agent/agent.h"
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

namespace agent {

    class FunctionCallAgentWrapper {
      public:
        FunctionCallAgentWrapper(
            std::shared_ptr<LLMAgentBase> llm_agent,
            std::unordered_map<std::string, std::function<std::optional<ChatMessage>(const nlohmann::json &)>>
                function_map)
            : llm_agent(std::move(llm_agent)), function_map(std::move(function_map)) {

              };
        ~FunctionCallAgentWrapper();

        std::vector<ChatMessage> inference(const AgentInferenceParam &param);

        void register_function(const std::string &function_name,
                               std::function<std::optional<ChatMessage>(const nlohmann::json &)> function_impl) {
            function_map[function_name] = function_impl;
        }

      private:
        /**
         * @brief Check if function call json's tools are all implemented in this->function_map
         *
         * @param func_call_ary
         * @return true all implemented
         * @return false missing anyone
         */
        bool validate_function_tool_impled(const nlohmann::json &func_call_ary);

        std::shared_ptr<LLMAgentBase> llm_agent;
        std::unordered_map<std::string, std::function<std::optional<ChatMessage>(const nlohmann::json &)>> function_map;
    };

} // namespace agent