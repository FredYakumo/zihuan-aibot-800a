#pragma once

#include "agent/agent.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "database.h"
#include <memory>
#include <optional>
#include <utility>

namespace agent {
    class SimpleChatActionAgent {
      public:
        SimpleChatActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter,
                              std::shared_ptr<LLMAgentBase> bind_output_llm_agent)
            : adapter(std::move(adapter)), bind_output_llm_agent(std::move(bind_output_llm_agent)) {}

        // 处理聊天 (原 process_llm 逻辑)
        void process_llm(const bot_cmd::CommandContext &context,
                         const std::optional<std::string> &additional_system_prompt_option,
                         const std::optional<database::UserPreference> &user_preference_option);

      private:
        void on_llm_thread(const bot_cmd::CommandContext &context, const std::string &llm_content,
                           const std::string &system_prompt,
                           const std::optional<database::UserPreference> &user_preference_option);

        std::shared_ptr<bot_adapter::BotAdapter> adapter; // (reserved for future non-static usage)
        std::shared_ptr<LLMAgentBase> bind_output_llm_agent;
    };
} // namespace agent