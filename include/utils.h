#ifndef UTILS_H
#define UTILS_H


#include <string>
#include <string_view>
#include <chrono>
#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include "MiraiCP.hpp"

std::string gen_common_prompt(const std::string_view bot_name, const MiraiCP::QQID bot_id, const std::string_view user_name, const uint64_t user_id);

inline std::string_view ltrim(const std::string_view str) {
    auto start = str.find_first_not_of(' ');
    if (start == std::string_view::npos) {
        return "";
    }
    start = str.find_first_not_of('\n', start);
    if (start == std::string_view::npos) {
        return "";
    }
    start = str.find_first_not_of('\r', start);
    if (start == std::string_view::npos) {
        return "";
    }
    return str.substr(start);
}

inline std::string get_current_time_formatted() {
    auto now = std::chrono::system_clock::now();

    // 使用 fmt 格式化时间
    std::string formatted_time = fmt::format("{:%Y年%m月%d日 %H:%M:%S}", now + std::chrono::hours(8));
    return formatted_time;
}

#endif