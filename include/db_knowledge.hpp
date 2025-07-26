#ifndef DB_KNOWLEDGE_HPP
#define DB_KNOWLEDGE_HPP

#include "utils.h"
#include <string>
#include <string_view>
#include <vector>

struct DBKnowledge {
    std::vector<std::string> keyword;
    std::string content;
    std::string creator_name;
    std::string create_dt;
    std::string knowledge_class_filter;

    float certainty = 0.0f;

    DBKnowledge() = default;
    DBKnowledge(const std::vector<std::string> &keyword, const std::string_view value,
                const std::string_view creator_name, float certainty)
        : DBKnowledge(keyword, value, creator_name, get_current_time_db(), certainty) {}
    DBKnowledge(const std::vector<std::string> &keyword, const std::string_view value,
                const std::string_view creator_name, const std::string_view create_dt, float certainty)
        : keyword(keyword), content(value), creator_name(creator_name), create_dt(create_dt), certainty(certainty) {}
};

#endif