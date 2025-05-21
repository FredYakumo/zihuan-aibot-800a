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

inline std::string_view ltrim(const std::string_view str) {
    const size_t start = str.find_first_not_of(" \n\r");
    return (start == std::string_view::npos) ? std::string_view() : str.substr(start);
}

inline std::string_view rtrim(const std::string_view str) {
    const size_t end = str.find_last_not_of(" \n\r");
    return (end == std::string_view::npos) ? std::string_view() : str.substr(0, end + 1);
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

class SplitString {
  public:
    SplitString(const std::string_view str, const char delimiter) : m_str(str), m_delimiter(delimiter), m_start(0) {}

    class Iterator {
      public:
        Iterator(const std::string_view str, const char delimiter, const size_t start)
            : m_str(str), m_delimiter(delimiter), m_start(start) {
            if (str.empty()) {
                m_start = m_end = std::string_view::npos;
            } else {
                find_next();
            }
        }

        std::string_view operator*() const {
            return m_str.substr(m_start, (m_end == std::string_view::npos) ? std::string_view::npos : m_end - m_start);
        }

        Iterator &operator++() {
            m_start = m_end;
            if (m_start != std::string_view::npos) {
                ++m_start;
                find_next();
            }
            return *this;
        }

        bool operator!=(const Iterator &other) const { return m_start != other.m_start; }
        Iterator operator++(int) {
            const Iterator tmp = *this;
            ++*this;
            return tmp;
        }

      private:
        void find_next() { m_end = m_str.find(m_delimiter, m_start); }

        std::string_view m_str;
        char m_delimiter;
        size_t m_start;
        size_t m_end{};
    };

    [[nodiscard]] Iterator begin() const { return {m_str, m_delimiter, m_start}; }
    [[nodiscard]] Iterator end() const { return {m_str, m_delimiter, std::string_view::npos}; }

  private:
    std::string_view m_str;
    char m_delimiter;
    size_t m_start;
};

#include <cassert>
#include <string_view>

class Utf8Splitter {
  public:
    Utf8Splitter(std::string_view str, size_t max_chars) : m_str(str), m_max_chars(max_chars) {
        assert(max_chars > 0 && "Max characters must be greater than 0");
    }

    class Iterator {
      public:
        Iterator(std::string_view str, size_t max_chars, size_t start_pos)
            : m_str(str), m_max_chars(max_chars), m_current_pos(start_pos) {
            if (start_pos != std::string_view::npos) {
                find_next_boundary();
            }
        }

        std::string_view operator*() const {
            if (m_current_pos == std::string_view::npos)
                return {};
            return m_str.substr(m_current_pos, m_next_pos - m_current_pos);
        }

        Iterator &operator++() {
            m_current_pos = m_next_pos;
            if (m_current_pos != std::string_view::npos) {
                find_next_boundary();
            }
            return *this;
        }

        bool operator!=(const Iterator &other) const { return m_current_pos != other.m_current_pos; }

        Iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

      private:
        void find_next_boundary() {
            m_next_pos = m_current_pos;
            size_t chars_count = 0;

            while (chars_count < m_max_chars && m_next_pos < m_str.size()) {
                const auto code_point_len = utf8_code_point_length(m_str[m_next_pos]);

                // 检查是否有足够的字节组成完整字符
                if (m_next_pos + code_point_len > m_str.size())
                    break;

                m_next_pos += code_point_len;
                ++chars_count;
            }

            if (chars_count == 0) {
                m_current_pos = m_next_pos = std::string_view::npos;
            }
        }

        static size_t utf8_code_point_length(char first_byte) noexcept {
            const auto uc = static_cast<unsigned char>(first_byte);
            if (uc < 0x80)
                return 1; // ASCII
            if ((uc & 0xE0) == 0xC0)
                return 2; // 2-byte
            if ((uc & 0xF0) == 0xE0)
                return 3; // 3-byte
            if ((uc & 0xF8) == 0xF0)
                return 4; // 4-byte
            return 1;     // 无效序列按单字节处理
        }

        std::string_view m_str;
        size_t m_max_chars;
        size_t m_current_pos;
        size_t m_next_pos = std::string_view::npos;
    };

    Iterator begin() const { return {m_str, m_max_chars, 0}; }
    Iterator end() const { return {m_str, m_max_chars, std::string_view::npos}; }

  private:
    std::string_view m_str;
    size_t m_max_chars;
};

/**
 * Replaces all occurrences of a pattern in a string with a replacement string, ensuring UTF-8 compatibility.
 *
 * @param str The input string to process.
 * @param pattern The pattern to search for and replace.
 * @param replace The string to replace the pattern with.
 * @return A new string with all occurrences of the pattern replaced.
 */
inline std::string replace_str(std::string_view str, std::string_view pattern, std::string_view replace) {
    if (pattern.empty()) {
        return std::string(str); // If the pattern is empty, return the original string.
    }
    std::string result;
    size_t start = 0;
    while (start <= str.size()) {
        // Find the next occurrence of the pattern.
        const size_t pos = str.find(pattern, start);
        if (pos == std::string_view::npos) {
            break; // No more matches fsnd.
        }
        // Ensure the match starts at a valid UTF-8 leader byte.
        if (!is_utf8_leader_byte(str[pos])) {
            start = pos + 1; // Skip invalid UTF-8 sequences.
            continue;
        }
        // Verify the match is complete and valid.
        if (pos + pattern.size() > str.size() || str.substr(pos, pattern.size()) != pattern) {
            const size_t step = utf8_char_length(str[pos]); // Handle multi-byte UTF-8 characters.
            start = pos + (step > 0 ? step : 1);            // Move to the next character.
            continue;
        }
        // Append the part of the string before the match.
        result.append(str.data() + start, pos - start);
        // Append the replacement string.
        result.append(replace.data(), replace.size());
        // Move the start position past the matched pattern.
        start = pos + pattern.size();
    }
    // Append the remaining part of the string after the last match.
    result.append(str.data() + start, str.size() - start);
    return result;
}

inline void remove_text_between_markers(std::string &str, const std::string &start_marker,
                                        const std::string &end_marker) {
    size_t start_pos = str.find(start_marker);
    size_t end_pos = str.find(end_marker);

    if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
        str.erase(start_pos, end_pos - start_pos + end_marker.length());
    }
}

template <typename STR_ITER>
inline std::string join_str(STR_ITER cbegin, STR_ITER cend, const std::string_view delimiter = ",") {
    std::string result;
    for (auto it = cbegin; it != cend; ++it) {
        if (it->empty()) {
            continue;
        }
        if (!result.empty()) {
            result += delimiter;
        }
        result += *it;
    }
    return result;
}

template <typename STR_ITER>
inline std::string
join_str(STR_ITER cbegin, STR_ITER cend, const std::string_view delimiter,
         std::function<std::string(const typename std::iterator_traits<STR_ITER>::value_type &)> transform) {
    std::string result;
    for (auto it = cbegin; it != cend; ++it) {
        auto mapped = transform(*it);
        if (mapped.empty()) {
            continue;
        }
        if (!result.empty()) {
            result += delimiter;
        }
        result += mapped;
    }
    return result;
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