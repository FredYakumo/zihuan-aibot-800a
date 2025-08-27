#pragma once

#include "agent/agent.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "database.h"
#include "nlohmann/json_fwd.hpp"
#include <memory>
#include <optional>
#include <utility>
#include <vector>

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

    ChatMessage handle_search_info(const bot_cmd::CommandContext &context, const ToolCall &tool_call);
    ChatMessage handle_fetch_url_content(const bot_cmd::CommandContext &context, const ToolCall &tool_call,
                                         const ChatMessage &llm_res);
    ChatMessage handle_view_model_info(const ToolCall &tool_call);
    ChatMessage handle_view_chat_history(const bot_cmd::CommandContext &context, const ToolCall &tool_call);
    ChatMessage handle_query_group(const bot_cmd::CommandContext &context, const ToolCall &tool_call);
    ChatMessage handle_get_function_list(const ToolCall &tool_call);

    std::shared_ptr<bot_adapter::BotAdapter> adapter;
    std::shared_ptr<LLMAgentBase> bind_output_llm_agent;
};
} // namespace agent
