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
#include "chat_session.hpp"
#include "db_knowledge.hpp"

constexpr size_t USER_SESSION_MSG_LIMIT = 5;

extern MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;
extern std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;
extern MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;

#endif