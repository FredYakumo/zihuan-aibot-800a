#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "mutex_data.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mutex_data.hpp"
#include "chat_session.hpp"
#include "db_knowledge.hpp"

constexpr size_t USER_SESSION_MSG_LIMIT = 60000;
constexpr size_t MAX_KNOWLEDGE_LENGTH = 4096;

extern MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;
extern MutexData<std::unordered_map<uint64_t, std::set<std::string>>> g_chat_session_knowledge_list_map;
extern std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;
extern MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

std::optional<ChatSession> get_user_chat_session(uint64_t chat_session);

#endif