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

class ActionAgent {
      public:
        ActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter);
        ~ActionAgent();
        ActionAgentResult process_user_input(const bot_adapter::Sender &sender, const std::string &user_input);

      private:
        std::shared_ptr<bot_adapter::BotAdapter> adapter;
    };
} // namespace agent