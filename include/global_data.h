#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "mutex_data.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utils.h"
#include "mutex_data.hpp"

constexpr size_t USER_SESSION_MSG_LIMIT = 5;

struct ChatMessage {
    std::string role{};
    std::string content{};
    std::chrono::system_clock::time_point timestamp{std::chrono::system_clock::now()};

    ChatMessage() = default;
    ChatMessage(const std::string_view role, const std::string_view content)
        : role(role), content(content), timestamp(std::chrono::system_clock::now()) {}
    ChatMessage(const std::string_view role, const std::string_view content, const std::chrono::system_clock::time_point &timestamp)
        : role(role), content(content), timestamp(timestamp) {}

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
            {"role", message.role}, {"content", message.content}, {"timestamp", message.get_formatted_timestamp()}};
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

struct DBKnowledge {
    std::string content;
    std::string creator_name;
    std::string create_dt;

    DBKnowledge() = default;
    DBKnowledge(const std::string_view content, const std::string_view creator_name): DBKnowledge(content, creator_name, get_current_time_db()) {}
    DBKnowledge(const std::string_view content, const std::string_view creator_name, const std::string_view create_dt):
        content(content), creator_name(creator_name), create_dt(create_dt) {}
};

extern MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;
extern std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;
extern MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

#endif