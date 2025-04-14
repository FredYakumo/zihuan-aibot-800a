#ifndef DB_KNOWLEDGE_HPP
#define DB_KNOWLEDGE_HPP

#include <string>
#include <string_view>

struct DBKnowledge {
    std::string content;
    std::string creator_name;
    std::string create_dt;

    DBKnowledge() = default;
    DBKnowledge(const std::string_view content, const std::string_view creator_name): DBKnowledge(content, creator_name, get_current_time_db()) {}
    DBKnowledge(const std::string_view content, const std::string_view creator_name, const std::string_view create_dt):
        content(content), creator_name(creator_name), create_dt(create_dt) {}
};

#endif