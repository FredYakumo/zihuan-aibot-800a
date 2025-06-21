#include "global_data.h"
#include "constant_types.hpp"
#include <general-wheel-cpp/mutex_data.hpp>
#include <set>

using namespace wheel;

concurrent_unordered_map<qq_id_t, ChatSession> g_chat_session_map;

concurrent_unordered_map<qq_id_t, std::set<std::string>> g_chat_session_knowledge_list_map;

concurrent_unordered_map<qq_id_t, bool> g_chat_processing_map;

concurrent_vector<DBKnowledge> g_wait_add_knowledge_list;

IndividualMessageStorage g_group_message_storage;

IndividualMessageStorage g_friend_message_storage;

IndividualMessageStorage g_bot_send_group_message_storage;