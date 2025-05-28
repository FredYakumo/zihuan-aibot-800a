#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string_view>

constexpr const char *ROLE_SYSTEM = "system";
constexpr const char *ROLE_USER = "user";
constexpr const char *ROLE_TOOL = "tool";
constexpr const char *ROLE_ASSISTANT = "assistant";

constexpr const char *EMPTY_MSG_TAG = "<未输入任何信息>";

constexpr const char *EMPTY_JSON_STR_VALUE = "<NULL>";

using qq_id_t = uint64_t;

constexpr qq_id_t UNKNOWN_ID = -1;

constexpr const char *UNKNOWN_VALUE = "<未知>";

constexpr size_t MAX_OUTPUT_LENGTH = 500;

#endif