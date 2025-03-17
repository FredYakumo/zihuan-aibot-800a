#ifndef MSG_DB_H
#define MSG_DB_H


#include "MiraiCP.hpp"
#include "global_data.h"
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct DBGroupMessage {
    std::string content;
    std::string sender_name;
    std::string group_name;
    std::string group_id;
    std::string sender_id;
    std::string send_time;
};

std::vector<std::pair<DBGroupMessage,double>> query_group_msg(const std::string_view query, std::optional<MiraiCP::QQID> group_id_option = std::nullopt);

std::vector<std::pair<DBKnowledge, double>> query_knowledge(const std::string_view query);

void insert_group_msg(MiraiCP::QQID group_id, const std::string_view group_name, MiraiCP::QQID sender_id, const std::string_view sender_name, const std::string_view content);

void insert_knowledge(const DBKnowledge &knowledge);

#endif