#ifndef UTILS_H
#define UTILS_H

#include "MiraiCP.hpp"
#include <chrono>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <string_view>

std::string gen_common_prompt(const std::string_view bot_name, const MiraiCP::QQID bot_id,
                              const std::string_view user_name, const uint64_t user_id);

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

    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm = *std::localtime(&now_time_t);

    // 使用 fmt 格式化时间
    std::string formatted_time = fmt::format("{:%Y年%m月%d日 %H:%M:%S}", now_tm);
    return formatted_time;
}

inline std::string get_current_time_db() {
    auto now = std::chrono::system_clock::now();

    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm = *std::localtime(&now_time_t);
    // 获取时区偏移
    char tz_offset[6];
    std::strftime(tz_offset, sizeof(tz_offset), "%z", &now_tm);
    // 手动调整时区偏移格式（例如 +0800 -> +08:00）
    std::string tz_offset_str(tz_offset);
    if (tz_offset_str.size() == 5) {
        tz_offset_str.insert(3, ":");
    }
    // 使用 fmt 格式化时间为 RFC3339 格式（包含时区偏移）
    return fmt::format("{:%Y-%m-%dT%H:%M:%S}{}", now_tm, tz_offset_str);
}

struct MessageProperties {
    bool is_at_me = false;
    std::shared_ptr<std::string> ref_msg_content = nullptr;
    std::shared_ptr<std::string> plain_content = nullptr;

    MessageProperties() = default;
    MessageProperties(bool is_at_me, std::shared_ptr<std::string> ref_msg_content,
                      std::shared_ptr<std::string> ref_plain_content)
        : is_at_me(is_at_me), ref_msg_content(ref_msg_content), plain_content(ref_plain_content) {}
};

MessageProperties get_msg_prop_from_event(const MiraiCP::GroupMessageEvent &e);

#include <string>
#include <thread>

#ifdef __linux__
#include <pthread.h>
#elif defined(_WIN32)
#include <stringapiset.h>
#include <windows.h>
#endif

inline void set_thread_name(const std::string &name) {
#ifdef __linux__
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str()); // 限制15字符
#elif defined(_WIN32)
    // 转换字符串编码为宽字符
    wchar_t wide_name[256];
    MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, wide_name, 256);
    SetThreadDescription(GetCurrentThread(), wide_name);
#elif defined(__APPLE__)
    pthread_setname_np(name.substr(0, 63).c_str()); // macOS 限制63字符
#endif
}

#endif