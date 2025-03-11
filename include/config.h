#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include "MiraiCP.hpp"

extern std::string LLM_API_URL;
extern std::string LLM_API_TOKEN;
extern std::string LLM_MODEL_NAME;
extern std::string CUSTOM_SYSTEM_PROMPT;
// extern std::string BOT_NAME;
// extern MiraiCP::QQID BOT_QQID;


void init_config();


#endif