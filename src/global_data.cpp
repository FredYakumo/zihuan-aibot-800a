#include "global_data.h"
#include "constant_types.hpp"
#include "individual_message_storage.hpp"
#include <general-wheel-cpp/mutex_data.hpp>
#include <set>

using namespace wheel;

// Bot start time (defined here; declared in global_data.h)
std::chrono::system_clock::time_point g_bot_start_time{};

concurrent_unordered_map<qq_id_t, ChatSession> g_chat_session_map;

concurrent_unordered_map<qq_id_t, std::set<std::string>> g_chat_session_knowledge_list_map;

concurrent_unordered_map<qq_id_t, bool> g_chat_processing_map;

concurrent_vector<DBKnowledge> g_wait_add_knowledge_list;

IndividualMessageStorage g_group_message_storage;

IndividualMessageStorage g_person_message_storage;

IndividualMessageStorage g_bot_send_group_message_storage;

wheel::concurrent_unordered_map<qq_id_t, std::chrono::system_clock::time_point> g_last_chat_message_time_map;

std::shared_ptr<agent::LLMAPIAgentBase> g_llm_chat_agent; // defined previously in main
std::shared_ptr<agent::SimpleChatActionAgent> g_simple_chat_action_agent; // new global chat action agent