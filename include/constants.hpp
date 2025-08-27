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

// BUILD_VERSION is defined by CMake via -DBUILD_VERSION
#ifndef BUILD_VERSION
#define BUILD_VERSION "unknown"
#endif

// DREPOS_ADDR is defined by CMake via -DREPOS_ADDR
#ifndef DREPOS_ADDR
#define DREPOS_ADDR "https://github.com/FredYakumo/zihuan-aibot-800a"
#endif

// Declare BUILD_VERSION_STRING as a constant string
extern const char* const BUILD_VERSION_STRING;

// Declare DREPOS_ADDR_STRING as a constant string
extern const char* const DREPOS_ADDR_STRING;

#endif