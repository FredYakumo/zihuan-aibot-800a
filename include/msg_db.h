#ifndef MSG_DB_H
#define MSG_DB_H


#include "MiraiCP.hpp"
#include <string>
#include <string_view>

struct DBGroupMessage {
    std::string_view sender_name;
    std::string_view group_name;
    MiraiCP::QQID group_id;
};

void insert_group_msg(MiraiCP::QQID group_id, const std::string_view group_name, MiraiCP::QQID sender_id, const std::string_view sender_name, const std::string_view content);

#endif