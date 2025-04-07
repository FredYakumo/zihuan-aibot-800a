#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string_view>

constexpr std::string_view ROLE_SYSTEM = "system";
constexpr std::string_view ROLE_USER = "user";
constexpr std::string_view ROLE_ASSISTANT = "assistant";

constexpr std::string_view EMPTY_MSG_TAG = "<未输入任何信息>";

constexpr std::string_view EMPTY_JSON_STR_VALUE = "<NULL>";

constexpr size_t MAX_OUTPUT_LENGTH = 200;

#endif