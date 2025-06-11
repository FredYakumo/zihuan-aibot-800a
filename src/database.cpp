#include "database.h"
#include "config.h"
#include <memory>
#include <stdexcept>
#include <general-wheel-cpp/string_utils.hpp>

using namespace database;

std::shared_ptr<DBConnection> g_db_connection;

void database::init_db_connection() {
    const auto &config = Config::instance();
    g_db_connection =
        std::make_shared<DBConnection>(std::string(config.database_host), config.database_port,
                                       std::string(config.database_user), std::string(config.database_password));
    spdlog::info("Init db connection successed. host: {}:{}", config.database_host, config.database_port);
}

DBConnection &database::get_global_db_connection() {
    if (g_db_connection == nullptr) {
        throw std::runtime_error("get_global_db_connection() Error: Database connection is not initialized.");
    }
    return *g_db_connection;
}

void DBConnection::insert_message(const std::string &content, const bot_adapter::Sender &sender,
                                  const std::chrono::system_clock::time_point send_time,
                                  const std::optional<std::set<uint64_t>> at_target_set) {
    try {
        const auto at_target_value = (at_target_set && !at_target_set->empty())
                                         ? wheel::join_str(std::cbegin(*at_target_set), std::cend(*at_target_set), ",",
                                                    [](const auto i) { return std::to_string(i); })
                                         : mysqlx::nullvalue;
        if (const auto &group_sender = bot_adapter::try_group_sender(sender)) {
            const auto &g = group_sender->get();
            get_message_record_table()
                .insert("sender_name", "sender_id", "content", "send_time", "group_name", "group_id",
                        "group_permission", "at_target_list")
                .values(g.name, g.id, content, time_point_to_db_str(send_time), g.group.name, g.group.id, g.permission,
                        at_target_value)
                .execute();
        } else {
            get_message_record_table()
                .insert("sender_name", "sender_id", "content", "send_time", "at_target_list")
                .values(sender.name, sender.id, content, time_point_to_db_str(send_time), at_target_value)
                .execute();
        }
        spdlog::info("Insert message successed.");
    } catch (const mysqlx::Error &e) {
        spdlog::error("Insert message error at MySQL X DevAPI Error: {}", e.what());
    }
}

void DBConnection::insert_tool_calls_record(const std::string &sender_name, qq_id_t sender_id,
                                            const std::string &origin_chat_session_view,
                                            const std::chrono::system_clock::time_point &send_time,
                                            const std::string &tool_calls, const std::string &tool_calls_content) {
                                                try {
                                                    get_tool_calls_record_table()
                                                        .insert("sender_name", "sender_id", "origin_chat_session_view", "send_time", "tool_calls", "tool_calls_content")
                                                        .values(sender_name, sender_id, origin_chat_session_view, time_point_to_db_str(send_time), 
                                                                tool_calls, tool_calls_content)
                                                        .execute();
                                                    spdlog::info("Insert tool calls record successed.");
                                                } catch (const mysqlx::Error &e) {
                                                    spdlog::error("Insert tool calls record error at MySQL X DevAPI Error: {}", e.what());
                                                }
                                            }

std::vector<GroupMessageRecord> DBConnection::query_group_user_message(uint64_t sender_id, uint64_t group_id,
                                                                       size_t count_limit) {
    mysqlx::RowResult sql_result = get_message_record_table()
                                       .select("content", "send_time")
                                       .where("sender_id = :sender_id and group_id = :group_id")
                                       .orderBy("send_time DESC")
                                       .limit(count_limit)
                                       .bind("sender_id", sender_id)
                                       .bind("group_id", group_id)
                                       .execute();
    std::vector<GroupMessageRecord> result;
    for (auto row : sql_result) {
        // result.emplace_back(row.get(0), )
    }
    return result;
}