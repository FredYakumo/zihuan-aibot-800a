#pragma once

#include "agent/agent.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "database.h"
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "nlohmann/json_fwd.hpp"

namespace agent {
    class SimpleChatActionAgent {
      public:
        SimpleChatActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter,
                              std::shared_ptr<LLMAgentBase> bind_output_llm_agent)
            : adapter(std::move(adapter)), bind_output_llm_agent(std::move(bind_output_llm_agent)) {}

        void process_llm(const bot_cmd::CommandContext &context,
                         const std::optional<std::string> &additional_system_prompt_option,
                         const std::optional<database::UserPreference> &user_preference_option,
                         const std::optional<nlohmann::json> &function_tools_opt = std::nullopt);

      private:
        void on_llm_thread(const bot_cmd::CommandContext &context, const std::string &llm_content,
                           const std::string &system_prompt,
                           const std::optional<database::UserPreference> &user_preference_option,
                           const std::optional<nlohmann::json> &function_tools_opt);
        void process_tool_calls(const bot_cmd::CommandContext &context, nlohmann::json &msg_json,
                                  std::vector<ChatMessage> &one_chat_session,
                                  const std::optional<nlohmann::json> &function_tools_opt);

        std::shared_ptr<bot_adapter::BotAdapter> adapter; // (reserved for future non-static usage)
        std::shared_ptr<LLMAgentBase> bind_output_llm_agent;
    };
} // namespace agent