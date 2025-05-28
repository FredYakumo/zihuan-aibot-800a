#ifndef CHAT_SESSION_HPP
#define CHAT_SESSION_HPP

#include <optional>
#include <string>
#include <chrono>
#include <nlohmann/json.hpp>
#include <deque>

struct ChatMessage {
    std::string role{};
    std::string content{};
    std::optional<std::string> tool_call_id = std::nullopt;
    std::chrono::system_clock::time_point timestamp{std::chrono::system_clock::now()};

    ChatMessage() = default;
    ChatMessage(const std::string_view role, const std::string_view content)
        : role(role), content(content), timestamp(std::chrono::system_clock::now()) {}
    ChatMessage(const std::string_view role, const std::string_view content, const std::chrono::system_clock::time_point &timestamp)
        : role(role), content(content), timestamp(timestamp) {}
    ChatMessage(const std::string_view role, const std::string_view content, const std::string_view tool_id)
        : role(role), content(content), tool_call_id(std::string(tool_id)), timestamp(std::chrono::system_clock::now()) {}
    ChatMessage(const std::string_view role, const std::string_view content, const std::string_view tool_id, const std::chrono::system_clock::time_point &timestamp)
        : role(role), content(content), tool_call_id(std::string(tool_id)), timestamp(timestamp) {}

    std::string get_formatted_timestamp() const {
        auto in_time_t = std::chrono::system_clock::to_time_t(timestamp);
        std::tm bt{};
        localtime_r(&in_time_t, &bt);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &bt);
        return std::string(buffer);
    }
    void to_json(nlohmann::json &j, const ChatMessage &message) {
        j = nlohmann::json{
            {"role", message.role}, 
            {"content", message.content}, 
            {"timestamp", message.get_formatted_timestamp()}
        };
        if (message.tool_call_id) {
            j["tool_call_id"] = *message.tool_call_id;
        }
    }
};

struct ChatSession {
    std::string nick_name;
    std::deque<ChatMessage> message_list {};
    size_t user_msg_count = 0;

    ChatSession() : nick_name("") {}

    ChatSession(const std::string_view name) : nick_name(name) {}

    ChatSession(const std::string_view name, const ChatMessage& initial_message)
        : nick_name(name), user_msg_count(1) {
        message_list.push_back(initial_message);
    }
};


#endif