#ifndef UTILS_H
#define UTILS_H

#include "adapter_model.h"
#include "i18n.hpp"
#include <chrono>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

inline std::string get_current_time_formatted() {
    auto now = std::chrono::system_clock::now();

    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);

    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm = *std::localtime(&now_time_t);

    // 使用 fmt 格式化时间
    std::string formatted_time = fmt::format("{:%Y年%m月%d日 %H:%M:%S}", now_tm);
    return formatted_time;
}

inline std::string time_point_to_db_str(const std::chrono::system_clock::time_point &time_point) {

    auto time_point_time_t = std::chrono::system_clock::to_time_t(time_point);

    std::tm time_tm = *std::localtime(&time_point_time_t);

    char tz_offset[6];
    std::strftime(tz_offset, sizeof(tz_offset), "%z", &time_tm);

    std::string tz_offset_str(tz_offset);
    if (tz_offset_str.size() == 5) {
        tz_offset_str.insert(3, ":");
    }

    return fmt::format("{:%Y-%m-%dT%H:%M:%S}{}", time_tm, tz_offset_str);
}

inline std::string get_current_time_db() {
    auto now = std::chrono::system_clock::now();

    return time_point_to_db_str(now);
}

inline std::string get_today_date_str() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    return fmt::format("{:%Y年%m月%d日}", tm);
}

/**
 * Extracts the content inside the parentheses following the first occurrence of #keyword in the string.
 *
 * @param s The input string to search.
 * @param keyword The keyword to search for, expected to be in the format "#keyword".
 * @return A string_view containing the content inside the parentheses. If the keyword or parentheses are not found, an
 * empty string_view is returned.
 */
std::string_view extract_parentheses_content_after_keyword(const std::string_view s, const std::string_view keyword);

/**
 * Replaces the first occurrence of #keyword and its following parentheses content in the string.
 *
 * @param original_str The original string to search and modify.
 * @param keyword The keyword to search for, expected to be in the format "#keyword".
 * @param replacement The string to replace the matched content with.
 * @return A new string with the replacement applied. If the keyword or parentheses are not found, the original string
 * is returned.
 */
std::string replace_keyword_and_parentheses_content(const std::string_view original_str, const std::string_view keyword,
                                                    const std::string_view replacement);

#include <string>

/**
 * Checks if the string strictly follows the format: "#keyword(content)".
 *
 * @param s The input string to validate.
 * @param keyword The keyword to check for, expected to be in the format "#keyword".
 * @return True if the string strictly matches the format, false otherwise.
 */
bool is_strict_format(const std::string_view s, const std::string_view keyword);

#include <string>

#ifdef __linux__
#include <pthread.h>
#elif defined(_WIN32)
#include <stringapiset.h>
#include <windows.h>
#endif

inline void set_thread_name(const std::string &name) {
#ifdef __linux__
    // limit to 15 chars
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#elif defined(_WIN32)
    // convert to wchar
    wchar_t wide_name[256];
    MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, wide_name, 256);
    SetThreadDescription(GetCurrentThread(), wide_name);
#elif defined(__APPLE__)
    // macOS limit 63 chars
    pthread_setname_np(name.substr(0, 63).c_str());
#endif
}











template <typename T, typename KEY_GENERATOR>
inline auto group_by(const std::vector<T> &collection, KEY_GENERATOR key_generator)
    -> std::unordered_map<decltype(key_generator(std::declval<T>())), T> {
    using KEY_T = decltype(key_generator(std::declval<T>()));
    std::unordered_map<KEY_T, T> ret;
    for (const auto &item : collection) {
        ret.insert(std::make_pair(key_generator(item), item));
    }
    return std::move(ret);
}

#endif