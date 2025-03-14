#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string_view>

constexpr std::string_view ROLE_SYSTEM = "system";
constexpr std::string_view ROLE_USER = "user";
constexpr std::string_view ROLE_ASSISTANT = "assistant";

constexpr std::string_view EMPTY_MSG_TAG = "<未输入任何信息>";

#endif