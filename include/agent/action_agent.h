#pragma once

#include "adapter_model.h"
#include "bot_adapter.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace agent {
    enum class AgentActionType {
        NONE = 0,
        TEXT_REPLY = 1,
        SYSTEM_INFO_REPLY = 1<<1,
        MODEL_INFO_REPLY = 1<<2,
        CODE_REPLY = 1<<3,
        PICTURES_REPLY = 1<<4
    };

    struct ActionAgentResult {
        AgentActionType action_type = AgentActionType::NONE;
        std::string content_text;
    };

    // 导出用于检测系统提示词相关概念的函数，供单元测试使用
    bool contains_system_prompt_terms(const std::string &text);
    
    // 系统提示词的目标类型枚举
    enum class SystemPromptTarget {
        SELF,         // The bot itself ("你的system_prompt", "紫幻的系统提示词")
        GENERAL_LLM,  // General large language models ("大模型的system_prompt")
        SPECIFIC_LLM, // Specific named models ("ChatGPT的系统提示词", "Claude的提示词")
        GENERIC,      // Generic or unspecified ("系统提示词是什么")
        NONE          // Not a system prompt request
    };
    
    std::optional<SystemPromptTarget>
    identify_system_prompt_target(const std::vector<neural_network::lac::OutputItem> &segmented_words);

    class ActionAgent {
      public:
        ActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter) : adapter(std::move(adapter)) {}
        ActionAgentResult process_user_input(const bot_adapter::Sender &sender, const std::string &user_input);

      private:
        std::shared_ptr<bot_adapter::BotAdapter> adapter;
    };
} // namespace agent