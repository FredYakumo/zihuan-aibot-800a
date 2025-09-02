// Implementation of LLMAPIAgentBase moved out of header to reduce rebuild scope.
#include "agent/agent.h"
#include "config.h"
#include "constants.hpp"
#include "get_optional.hpp"
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>

namespace agent {

    std::optional<ChatMessage> LLMAPIAgentBase::inference(const AgentInferenceParam &param) {
        nlohmann::json body = {{"model", model_name},
                               {"messages", param.messages_json},
                               {"stream", false},
                               {"is_deep_think", param.think_mode},
                               {"temperature", param.think_mode ? 0.0 : 1.3}};

        if (param.function_tools_opt.has_value()) {
            body["tools"] = *param.function_tools_opt;
        }

        const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        spdlog::info("llm body: {}", json_str);

        // Build target URL. If a base URL (host:port) was passed in, append the endpoint suffix.
        std::string target_url;
        if (!api_url.empty()) {
            target_url = api_url;
            // Heuristic: if caller provided only base without endpoint path, append suffix.
            // We append when the url does not already contain a known completions endpoint keyword.
            const bool has_completions = (target_url.find("chat/completions") != std::string::npos) ||
                                         (target_url.find("/completions") != std::string::npos) ||
                                         (target_url.find("/responses") != std::string::npos);
            if (!has_completions) {
                if (!target_url.empty() && target_url.back() == '/')
                    target_url += LLM_API_SUFFIX;
                else
                    target_url += std::string("/") + LLM_API_SUFFIX;
            }
        } else {
            const auto &cfg = Config::instance();
            target_url = fmt::format("{}:{}/{}", cfg.llm_api_url, cfg.llm_api_port, LLM_API_SUFFIX);
        }

        cpr::Header header{{"Content-Type", "application/json"}};
        if (api_key_option.has_value()) {
            header["Authorization"] = fmt::format("Bearer {}", *api_key_option);
        } else if (!Config::instance().llm_api_key.empty()) {
            header["Authorization"] = fmt::format("{}", Config::instance().llm_api_key);
        }

        cpr::Response response = cpr::Post(cpr::Url{target_url}, cpr::Body{json_str}, header);
        spdlog::info("llm response: {}, status code: {}", response.error.message, response.status_code);

        // Short-circuit on HTTP error
        if (response.status_code != 200) {
            spdlog::error("LLM API request failed. status: {}, url: {}, response: {}", response.status_code,
                          target_url, response.text);
            return std::nullopt;
        }

        try {
            spdlog::info("LLM response: {}", response.text);
            auto json = nlohmann::json::parse(response.text);
            // choices
            auto choices_opt = get_optional<nlohmann::json>(json, "choices");
            if (!choices_opt || !choices_opt->is_array() || choices_opt->empty()) {
                spdlog::error("LLM response missing choices: {}", response.text);
                return std::nullopt;
            }
            const nlohmann::json &first_choice = (*choices_opt)[0];
            // message (fallback to delta if provider uses that field)
            auto message_opt = get_optional<nlohmann::json>(first_choice, "message");
            if (!message_opt) {
                message_opt = get_optional<nlohmann::json>(first_choice, "delta");
            }
            if (!message_opt) {
                spdlog::error("LLM response missing 'message' (or 'delta') field: {}", first_choice.dump());
                return std::nullopt;
            }
            const nlohmann::json &message = *message_opt;

            // content may be null when the assistant only returns tool_calls
            std::string result;
            if (auto content_opt = get_optional<std::string>(message, "content"); content_opt.has_value()) {
                result = std::string(wheel::ltrim(std::string_view(*content_opt)));
            } else {
                result = ""; // keep empty and rely on tool_calls if present
            }
            std::string role = get_optional<std::string>(message, "role").value_or(ROLE_ASSISTANT);
            wheel::remove_text_between_markers(result, "<think>", "</think>");
            ChatMessage ret{role, result};
            if (auto tool_calls = get_optional<nlohmann::json>(message, "tool_calls"); tool_calls.has_value()) {
                spdlog::info("LLM response TOOL CALLS request");
                std::vector<ToolCall> function_calls;
                for (const nlohmann::json &tool_call : *tool_calls) {
                    if (auto tc = try_get_chat_completeion_from_messag_tool_call(tool_call); tc.has_value())
                        function_calls.emplace_back(*tc);
                    else
                        spdlog::error("解析tool call失败, 原始json为: {}", tool_call.dump());
                }
                ret.tool_calls = std::move(function_calls);
            }
            return ret;
        } catch (const std::exception &e) {
            spdlog::error("LLMAPIAgentBase::inference(): JSON 解析失败, {}", e.what());
        }
        return std::nullopt;
    }

} // namespace agent
