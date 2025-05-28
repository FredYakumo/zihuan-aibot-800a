#ifndef DB_KNOWLEDGE_HPP
#define DB_KNOWLEDGE_HPP

#include "utils.h"
#include <string>
#include <string_view>
#include <vector>

struct DBKnowledge {
    std::string content;
    std::string creator_name;
    std::string create_dt;
    std::vector<std::string> class_list;
    float certainty = 0.0f;

    DBKnowledge() = default;
    DBKnowledge(const std::string_view content, const std::string_view creator_name, const std::vector<std::string> &keywords, float certainty): DBKnowledge(content, creator_name, get_current_time_db(), keywords, certainty) {}
    DBKnowledge(const std::string_view content, const std::string_view creator_name, const std::string_view create_dt, const std::vector<std::string> &class_list, float certainty):
        content(content), creator_name(creator_name), create_dt(create_dt), class_list(class_list), certainty(certainty) {}
};

#endif