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

inline std::string get_current_time_db() {
    auto now = std::chrono::system_clock::now();

    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm = *std::localtime(&now_time_t);

    char tz_offset[6];
    std::strftime(tz_offset, sizeof(tz_offset), "%z", &now_tm);

    std::string tz_offset_str(tz_offset);
    if (tz_offset_str.size() == 5) {
        tz_offset_str.insert(3, ":");
    }

    return fmt::format("{:%Y-%m-%dT%H:%M:%S}{}", now_tm, tz_offset_str);
}

std::string_view extract_quoted_content(const std::string_view s, const std::string_view keyword);
// 替换字符串中第一个 #语录(...) 的内容
std::string replace_quoted_content(const std::string_view original_str, const std::string_view keyword,
                                   const std::string_view replacement);

#include <string>

bool is_strict_format(const std::string_view s, const std::string_view keyword);


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
    Utf8Splitter(std::string_view str, size_t max_chars)
        : m_str(str), m_max_chars(max_chars) {
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
            if (m_current_pos == std::string_view::npos) return {};
            return m_str.substr(m_current_pos, m_next_pos - m_current_pos);
        }

        Iterator& operator++() {
            m_current_pos = m_next_pos;
            if (m_current_pos != std::string_view::npos) {
                find_next_boundary();
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return m_current_pos != other.m_current_pos;
        }

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
                if (m_next_pos + code_point_len > m_str.size()) break;
                
                m_next_pos += code_point_len;
                ++chars_count;
            }
            
            if (chars_count == 0) {
                m_current_pos = m_next_pos = std::string_view::npos;
            }
        }

        static size_t utf8_code_point_length(char first_byte) noexcept {
            const auto uc = static_cast<unsigned char>(first_byte);
            if (uc < 0x80) return 1;          // ASCII
            if ((uc & 0xE0) == 0xC0) return 2; // 2-byte
            if ((uc & 0xF0) == 0xE0) return 3; // 3-byte
            if ((uc & 0xF8) == 0xF0) return 4; // 4-byte
            return 1;                          // 无效序列按单字节处理
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


#endif