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

    // Forward declaration for ReferenceTerms
    struct ReferenceTerms;
    
    // 导出用于检测系统提示词相关概念的函数，供单元测试使用
    bool contains_system_prompt_terms(const std::string &text, const ReferenceTerms* cachedTerms);
    
    // 系统提示词的目标类型枚举
    enum class SystemPromptTarget {
        SELF,         // The bot itself
        GENERAL_LLM,  // General large language models
        SPECIFIC_LLM, // Specific named models
        GENERIC,      // Generic or unspecified
        NONE          // Not a system prompt request
    };
    
    std::optional<SystemPromptTarget>
    identify_system_prompt_target(const std::vector<neural_network::lac::OutputItem> &segmented_words, 
                                 const ReferenceTerms* cachedTerms);

    // Forward declaration of ReferenceTerms struct
struct ReferenceTerms;

class ActionAgent {
      public:
        ActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter);
        ~ActionAgent();
        ActionAgentResult process_user_input(const bot_adapter::Sender &sender, const std::string &user_input);

      private:
        std::shared_ptr<bot_adapter::BotAdapter> adapter;
        std::unique_ptr<ReferenceTerms> reference_terms; // Cached reference terms
        void load_reference_terms();
    };
} // namespace agent