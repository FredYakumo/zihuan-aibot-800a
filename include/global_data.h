#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include <chrono>
#include <general-wheel-cpp/mutex_data.hpp>
#include <general-wheel-cpp/collection/concurrent_unordered_map.hpp>
#include <general-wheel-cpp/collection/concurrent_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <chat_session.hpp>
#include <set>
#include "agent/agent.h"
#include "agent/simple_chat_action_agent.h"
#include "embedding_message_id_list.hpp"
#include "individual_message_storage.hpp"
#include "db_knowledge.hpp"
#include "constant_types.hpp"

constexpr size_t USER_SESSION_MSG_LIMIT = 60000;
constexpr size_t MAX_KNOWLEDGE_LENGTH = 4096;

extern std::chrono::system_clock::time_point g_bot_start_time;

inline std::string get_bot_start_time_str() {
    std::time_t start_time = std::chrono::system_clock::to_time_t(g_bot_start_time);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&start_time));
    return std::string(buf);
}

inline std::string get_bot_run_duration_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto duration = duration_cast<seconds>(now - g_bot_start_time).count();

    int days = static_cast<int>(duration / (24 * 3600));
    duration %= (24 * 3600);
    int hours = static_cast<int>(duration / 3600);
    duration %= 3600;
    int minutes = static_cast<int>(duration / 60);
    int seconds = static_cast<int>(duration % 60);

    std::string result = "已运行";
    if (days > 0) result += std::to_string(days) + "天";
    if (hours > 0) result += std::to_string(hours) + "小时";
    if (minutes > 0) result += std::to_string(minutes) + "分钟";
    result += std::to_string(seconds) + "秒";
    return result;
}

/// 用户/user chat session map
/// key = user QQ号
extern wheel::concurrent_unordered_map<qq_id_t, ChatSession> g_chat_session_map;
/// 用户/user chat knowledge map
/// key = user QQ号
extern wheel::concurrent_unordered_map<uint64_t, std::set<std::string>> g_chat_session_knowledge_list_map;
/// 用户/user chat is processing lock map
/// key = user QQ号
extern wheel::concurrent_unordered_map<uint64_t, bool> g_chat_processing_map;

extern wheel::concurrent_vector<DBKnowledge> g_wait_add_knowledge_list;



/// 群聊/group chat msg storage, individual id = group id/群号
extern IndividualMessageStorage g_group_message_storage;
/// 个人/person chat msg storage, individual id = friend id/好友QQ号
extern IndividualMessageStorage g_person_message_storage;
/// bot send message to group storage. individual id = send to groupID/群号 or friend id/好友QQ号
extern IndividualMessageStorage g_bot_send_group_message_storage;

/// person chat with bot last message time map
/// key = person QQ号/id
extern wheel::concurrent_unordered_map<qq_id_t, std::chrono::system_clock::time_point> g_last_chat_message_time_map;

/// 群聊message content(or any text) embedding related message_id map
/// key = group id/群号
extern wheel::concurrent_unordered_map<qq_id_t, embedding_message_id_list> g_group_message_embedding_to_id_list_map;

/// 群聊message content(or any text) embedding related message_id map
/// key = group id/群号
/// Bot send only
extern wheel::concurrent_unordered_map<qq_id_t, embedding_message_id_list> g_bot_send_group_message_embedding_to_id_list_map;

extern std::shared_ptr<agent::LLMAPIAgentBase> g_llm_chat_agent;
extern std::shared_ptr<agent::SimpleChatActionAgent> g_simple_chat_action_agent;


#endif