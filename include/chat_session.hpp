#ifndef CHAT_SESSION_HPP
#define CHAT_SESSION_HPP

#include "get_optional.hpp"
#include <chrono>
#include <cstdint>
#include <deque>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

struct ToolCall {
    std::string id;
    std::string arguments;
    std::string name;
    std::string type;
    int64_t index;
};

inline std::optional<ToolCall> try_get_chat_completeion_from_messag_tool_call(const nlohmann::json &tool_call) {
    const auto func = get_optional(tool_call, "function");
    if (!func.has_value()) {
        spdlog::error("JSON 解析失败: tool_call里没有func");
        return std::nullopt;
    }
    const auto id = get_optional<std::string>(tool_call, "id");
    if (!id.has_value()) {
        spdlog::error("JSON 解析失败: tool_call里没有id");
        return std::nullopt;
    }
    const auto name = get_optional<std::string>(func, "name");
    if (!name.has_value()) {
        spdlog::error("JSON 解析失败: tool_call.function里没有name");
        return std::nullopt;
    }
    const auto arguments = get_optional(func, "arguments");
    if (!name.has_value()) {
        spdlog::error("JSON 解析失败: tool_call.function里没有arguments");
        return std::nullopt;
    }
    const auto type = get_optional(tool_call, "type");
    if (!name.has_value()) {
        spdlog::error("JSON 解析失败: tool_call里没有type");
        return std::nullopt;
    }
    const auto index = get_optional<int64_t>(tool_call, "index");
    if (!name.has_value()) {
        spdlog::error("JSON 解析失败: tool_call里没有index");
        return std::nullopt;
    }
    return ToolCall{.id = *std::move(id),
                    .arguments = std::move(*arguments),
                    .name = std::move(*name),
                    .type = std::move(*type),
                    .index = std::move(*index)};
}
struct ChatMessage {
    std::string role{};
    std::string content{};
    std::optional<std::string> tool_call_id = std::nullopt;
    std::chrono::system_clock::time_point timestamp{std::chrono::system_clock::now()};
    std::optional<std::vector<ToolCall>> tool_calls;

    /// Specify a tool call's context
    /// Use to judge if this tool call request message and tool call role content
    /// is send to llm in current chat context env(at present, Group).
    /// This will not send to llm.
    std::optional<std::string> tool_call_context_env;

    ChatMessage() = default;
    ChatMessage(const std::string_view role, const std::string_view content)
        : role(role), content(content), timestamp(std::chrono::system_clock::now()) {}
    ChatMessage(const std::string_view role, const std::string_view content, const std::string_view tool_id)
        : role(role), content(content), tool_call_id(std::string(tool_id)),
          timestamp(std::chrono::system_clock::now()) {}

    std::string get_formatted_timestamp() const {
        auto in_time_t = std::chrono::system_clock::to_time_t(timestamp);
        std::tm bt{};
        localtime_r(&in_time_t, &bt);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &bt);
        return std::string(buffer);
    }
    nlohmann::json to_json() const {
        nlohmann::json j{{"role", role}, {"content", content}, {"timestamp", get_formatted_timestamp()}};
        if (tool_call_id) {
            j["tool_call_id"] = *tool_call_id;
        }
        if (tool_calls && !tool_calls->empty()) {
            nlohmann::json tool_calls_array = nlohmann::json::array();
            for (const auto &tool_call : *tool_calls) {
                tool_calls_array.push_back(
                    {{"id", tool_call.id},
                     {"type", tool_call.type},
                     {"function",
                      {{"arguments", tool_call.arguments}, {"name", tool_call.name}, {"index", tool_call.index}}}});
            }
            j["tool_calls"] = tool_calls_array;
        }
        return j;
    }
};

/**
 * @brief A user chat session with bot
 */
struct ChatSession {
    /// @brief How bot call user
    std::string nick_name;
    std::deque<ChatMessage> message_list{};

    ChatSession(std::string name) : nick_name(std::move(name)) {}

    ChatSession(std::string name, ChatMessage initial_message)
        : nick_name(std::move(name)), message_list{std::move(initial_message)} {}

    ChatSession(ChatSession &&) = default;
    ChatSession &operator=(ChatSession &&) = default;

    // Prevent copy
    ChatSession(const ChatSession &) = delete;
    ChatSession &operator=(const ChatSession &) = delete;
};

#endif