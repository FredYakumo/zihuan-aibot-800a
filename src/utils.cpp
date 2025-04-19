#include "utils.h"
#include "adapter_model.h"
#include "config.h"
#include <fmt/format.h>
#include <string_view>

std::string_view extract_parentheses_content_after_keyword(
    const std::string_view s,
    const std::string_view keyword
) {
    size_t keyword_pos = s.find(keyword);
    if (keyword_pos == std::string::npos) {
        return ""; // Keyword not found
    }
    size_t start_paren = s.find('(', keyword_pos + keyword.size());
    if (start_paren == std::string::npos) {
        return ""; // Opening parenthesis not found
    }
    size_t end_paren = s.find(')', start_paren + 1);
    if (end_paren == std::string::npos) {
        return ""; // Closing parenthesis not found
    }
    return s.substr(start_paren + 1, end_paren - start_paren - 1);
}


std::string replace_keyword_and_parentheses_content(
    const std::string_view original_str,
    const std::string_view keyword,
    const std::string_view replacement
) {
    size_t keyword_pos = original_str.find(keyword);
    if (keyword_pos == std::string::npos) {
        return std::string{original_str}; // Keyword not found, return the original string
    }
    size_t start_paren = original_str.find('(', keyword_pos + keyword.size());
    if (start_paren == std::string::npos) {
        return std::string{original_str}; // Opening parenthesis not found
    }

    size_t end_paren = original_str.find(')', start_paren + 1);
    if (end_paren == std::string::npos) {
        return std::string{original_str}; // Closing parenthesis not found
    }

    size_t replace_start = keyword_pos;
    size_t replace_length = end_paren - keyword_pos + 1;

    std::string result = std::string{original_str};
    result.replace(replace_start, replace_length, replacement);
    return result;
}


#include <string>

bool is_strict_format(const std::string_view s, const std::string_view keyword) {
    const size_t key_len = keyword.size();

    if (s.size() < key_len + 2) return false;

    if (s.substr(0, key_len) != keyword) return false;

    size_t start_quote = s.find('(', key_len);
    if (start_quote == std::string_view::npos || start_quote < key_len) return false;

    size_t end_quote = s.find(')', start_quote + 1);
    if (end_quote != s.size() - 1) return false;

    return (end_quote > start_quote);
}
