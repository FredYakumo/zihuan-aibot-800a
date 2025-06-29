#ifndef DB_KNOWLEDGE_HPP
#define DB_KNOWLEDGE_HPP

#include "utils.h"
#include <string>
#include <string_view>
#include <vector>

struct DBKnowledge {
    std::string key;
    std::string value;
    std::string creator_name;
    std::string create_dt;
    float certainty = 0.0f;

    DBKnowledge() = default;
    DBKnowledge(const std::string_view key, const std::string_view value, const std::string_view creator_name, float certainty)
        : DBKnowledge(key, value, creator_name, get_current_time_db(), certainty) {}
    DBKnowledge(const std::string_view key, const std::string_view value, const std::string_view creator_name, const std::string_view create_dt, float certainty)
        : key(key), value(value), creator_name(creator_name), create_dt(create_dt),
          certainty(certainty) {}
};

#endif