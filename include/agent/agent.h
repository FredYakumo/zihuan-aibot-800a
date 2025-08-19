#pragma once

#include "adapter_model.h"
#include "agent/agent.h"
#include "chat_session.hpp"
#include "config.h"
#include "constants.hpp"
#include "get_optional.hpp"
#include "agent/llm.h"
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <optional>
#include <spdlog/spdlog.h>

namespace agent {
    struct AgentInferenceParam {
        bool think_mode = false;                                 // whether to enable deep thinking / chain-of-thought
        nlohmann::json messages_json = nlohmann::json::array();  // pre-serialized messages array
        std::optional<nlohmann::json> function_tools_opt;        // optional tool/function definitions
    };

    class LLMAgentBase {
      public:
        explicit LLMAgentBase(std::string model_name) : model_name(std::move(model_name)) {}
        // Now inference receives all dynamic data via AgentInferenceParam (stateless agent        virtual std::optional<ChatMessage> inference(const AgentInferenceParam &param) = 0;

      protected:
        std::string model_name;
    };

    class LLMAPIAgentBase : public LLMAgentBase {
      public:
        LLMAPIAgentBase(std::string model_name, std::string api_url, std::optional<std::string> api_key_option)
            : LLMAgentBase(model_name), model_name(std::move(model_name)), api_url(std::move(api_url)),
              api_key_option(std::move(api_key_option)) {}
        std::optional<ChatMessage> inference(const AgentInferenceParam &param) override;

      private:
        std::string model_name;
        std::string api_url;
        std::optional<std::string> api_key_option;
    };

} // namespace agent