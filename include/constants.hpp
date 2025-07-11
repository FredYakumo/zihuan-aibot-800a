#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "constant_types.hpp"
#include <cstddef>

constexpr const char *ROLE_SYSTEM = "system";
constexpr const char *ROLE_USER = "user";
constexpr const char *ROLE_TOOL = "tool";
constexpr const char *ROLE_ASSISTANT = "assistant";

constexpr const char *EMPTY_MSG_TAG = "<未输入任何信息>";

constexpr const char *EMPTY_JSON_STR_VALUE = "<NULL>";

constexpr const char *LLM_API_SUFFIX = "chat/completions";
constexpr const char *LLM_MODEL_INFO_SUFFIX = "model_info";

constexpr const char *SEARCH_WEB_SUFFIX = "search_web";
constexpr const char *SEARCH_URL_SUFFIX = "curl";

constexpr qq_id_t UNKNOWN_ID = -1;

constexpr const char *UNKNOWN_VALUE = "<未知>";

constexpr size_t MAX_OUTPUT_LENGTH = 500;

#endif