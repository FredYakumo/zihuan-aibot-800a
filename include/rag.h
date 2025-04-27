#ifndef MSG_DB_H
#define MSG_DB_H

#include "global_data.h"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rag {

    struct DBGroupMessage {
        std::string content;
        std::string sender_name;
        std::string group_name;
        std::string group_id;
        std::string sender_id;
        std::string send_time;
    };

    struct NetSearchImage {
        std::string url;
        std::string description;
    };

    struct NetSearchResult {
        std::string title;
        std::string url;
        std::string content;
        double score;
    };

    std::vector<std::pair<DBGroupMessage, double>>
    query_group_msg(const std::string_view query, std::optional<uint64_t> group_id_option = std::nullopt);

    std::vector<DBKnowledge> query_knowledge(const std::string_view query);

    void insert_group_msg(uint64_t group_id, const std::string_view group_name,uint64_t sender_id,
                          const std::string_view sender_name, const std::string_view content);

    void insert_knowledge(const DBKnowledge &knowledge);

    std::vector<NetSearchResult> net_search_content(const std::string_view query);

    std::vector<NetSearchImage> net_search_image(const std::string_view query);

    std::string url_search_content(const std::vector<std::string> &url_list);

    
} // namespace rag

#endif