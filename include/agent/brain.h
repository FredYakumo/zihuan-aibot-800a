#pragma once
#include "adapter_event.h"
#include "agent.h"
#include "bot_adapter.h"
#include "constant_types.hpp"
#include <unordered_set>

#include <memory>
namespace agent {
    enum class EmotionType { Happy, Sad, Angry, Normal };

    struct EmotionState {
        EmotionType type;
        std::unordered_set<qq_id_t> related_target;
        std::unordered_set<std::string> related_entities;
    };

    class Brain {
      public:
        Brain(std::shared_ptr<bot_adapter::BotAdapter> adapter, std::shared_ptr<LLMAgentBase> decision_llm)
            : m_adapter(std::move(adapter)), m_decision_llm(std::move(decision_llm)) {}

        void process_friend_message_event(const bot_adapter::FriendMessageEvent &event);
        void process_group_message_event(const bot_adapter::GroupMessageEvent &event);

        virtual ~Brain() = default;

      private:
        

      
        std::shared_ptr<bot_adapter::BotAdapter> m_adapter;
        std::shared_ptr<LLMAgentBase> m_decision_llm;
    };
} // namespace agent