#pragma once
#include <string>

namespace vec_db {
    struct DBGroupMessage {
        std::string content;
        std::string sender_name;
        std::string group_name;
        std::string group_id;
        std::string sender_id;
        std::string send_time;
    };
} // namespace vec_db