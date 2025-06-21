#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include <general-wheel-cpp/mutex_data.hpp>
#include <general-wheel-cpp/collection/concurrent_unordered_map.hpp>
#include <general-wheel-cpp/collection/concurrent_vector.hpp>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <chat_session.hpp>
#include <set>
#include "individual_message_storage.hpp"
#include "db_knowledge.hpp"
#include "constant_types.hpp"

constexpr size_t USER_SESSION_MSG_LIMIT = 60000;
constexpr size_t MAX_KNOWLEDGE_LENGTH = 4096;

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
/// 私聊/friend chat msg storage, individual id = friend id/好友QQ号
extern IndividualMessageStorage g_friend_message_storage;
/// bot send message to group storage. individual id = send to groupID/群号 or friend id/好友QQ号
extern IndividualMessageStorage g_bot_send_group_message_storage;

#endif