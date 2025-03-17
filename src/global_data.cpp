#include "global_data.h"
#include "mutex_data.hpp"

MutexData<std::unordered_map<MiraiCP::QQID, ChatSession>> g_chat_session_map;

std::pair<std::mutex, std::unordered_map<MiraiCP::QQID, bool>> g_chat_processing_map;

MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;