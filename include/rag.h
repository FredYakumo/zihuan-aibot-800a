#ifndef MSG_DB_H
#define MSG_DB_H

#include "chat_session.hpp"
#include "constants.hpp"
#include "global_data.h"
#include "msg_prop.h"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
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

    struct UrlSearchResult {
        std::vector<std::pair<std::string, std::string>> results;
        std::vector<std::pair<std::string, std::string>> failed_reason;
    };

    std::vector<DBKnowledge> query_knowledge(const std::string_view query, bool exactly_match = false);

    void insert_knowledge(const DBKnowledge &knowledge);

    std::vector<NetSearchResult> net_search_content(const std::string_view query);

    std::vector<NetSearchImage> net_search_image(const std::string_view query);

    UrlSearchResult url_search_content(const std::vector<std::string> &url_list);
} // namespace rag

#endif