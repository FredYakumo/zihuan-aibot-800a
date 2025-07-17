#include "database.h"
#include "config.h"
#include "constant_types.hpp"
#include <general-wheel-cpp/string_utils.hpp>
#include <memory>
#include <stdexcept>
#include <time_utils.h>

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

void DBConnection::insert_message(message_id_t message_id,
                                  const std::string &content,
                                  const bot_adapter::Sender &sender,
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
                .insert("message_id", "sender_name", "sender_id", "content", "send_time", "group_name", "group_id",
                        "group_permission", "at_target_list")
                .values(message_id, g.name, g.id, content, time_point_to_db_str(send_time), g.group.name, g.group.id, g.permission,
                        at_target_value)
                .execute();
        } else {
            get_message_record_table()
                .insert("message_id", "sender_name", "sender_id", "content", "send_time", "at_target_list")
                .values(message_id, sender.name, sender.id, content, time_point_to_db_str(send_time), at_target_value)
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
            .insert("sender_name", "sender_id", "origin_chat_session_view", "send_time", "tool_calls",
                    "tool_calls_content")
            .values(sender_name, sender_id, origin_chat_session_view, time_point_to_db_str(send_time), tool_calls,
                    tool_calls_content)
            .execute();
        spdlog::info("Insert tool calls record successed.");
    } catch (const mysqlx::Error &e) {
        spdlog::error("Insert tool calls record error at MySQL X DevAPI Error: {}", e.what());
    }
}



std::vector<GroupMessageRecord>
DBConnection::query_group_message(qq_id_t group_id, std::optional<qq_id_t> filter_sender, size_t count_limit) {
    auto sql = get_message_record_table().select(
        "message_id", "content", "UNIX_TIMESTAMP(send_time) as send_time", "sender_id", "sender_name", "group_id", "group_name", "group_permission"
    );
    if (filter_sender) {
        sql.where("group_id = :group_id and sender_id = :sender_id")
            .bind("group_id", group_id)
            .bind("sender_id", *filter_sender);
    } else {
        sql.where("group_id = :group_id").bind("group_id", group_id);
    }
    sql.orderBy("send_time DESC").limit(count_limit);
    mysqlx::RowResult sql_result = sql.execute();

    std::vector<GroupMessageRecord> result;
    for (auto row : sql_result) {
        auto message_id_val = row[0];
        std::string content = row[1].get<std::string>();
        // Convert MySQL DATETIME to timestamp
        auto send_time = std::chrono::system_clock::from_time_t(
            row[2].get<uint64_t>());
        uint64_t sender_id = std::stoull(row[3].get<std::string>());
        std::string sender_name = row[4].get<std::string>();
        uint64_t group_id_val = std::stoull(row[5].get<std::string>());
        std::string group_name = row[6].get<std::string>();
        std::string group_permission = row[7].get<std::string>();

        // Construct Group, GroupSender 
        bot_adapter::Group group(group_id_val, group_name, group_permission);
        auto sender = std::make_shared<bot_adapter::GroupSender>(
            sender_id, sender_name, "", // id, name, remark
            group_permission, // permission
            std::nullopt, // join_time
            send_time, // last_speak_time  
            group // group
        );
        GroupMessageRecord record(content, send_time, *sender);
        if (!message_id_val.isNull()) {
            try {
                record.message_id_opt = std::stoull(message_id_val.get<std::string>());
            } catch (const std::exception& e) {
                spdlog::warn("Failed to convert message_id to integer: {}", e.what());
                record.message_id_opt = std::nullopt;
            }
        }
        result.push_back(std::move(record));
    }
    return result;
}

void DBConnection::insert_or_update_user_preferences(
    const std::vector<std::pair<qq_id_t, UserPreference>> &user_preferences) {
    if (user_preferences.empty()) {
        return;
    }

    try {
        std::vector<std::string> user_ids;
        user_ids.reserve(user_preferences.size());
        for (const auto &pref_pair : user_preferences) {
            user_ids.push_back(std::to_string(pref_pair.first));
        }

        std::string user_ids_list_str = wheel::join_str(user_ids.cbegin(), user_ids.cend(), ",",
                                                        [](const std::string &id) { return "'" + id + "'"; });

        // Execute delete operation
        auto delete_result =
            get_user_preference_table().remove().where("user_id IN (" + user_ids_list_str + ")").execute();
        auto deleted_count = delete_result.getAffectedItemsCount();

        // Execute insert operation with fresh table reference
        auto insert = get_user_preference_table().insert("user_id", "render_markdown_output", "text_output",
                                                         "auto_new_chat_session");
        for (const auto &pref_pair : user_preferences) {
            insert.values(std::to_string(pref_pair.first), pref_pair.second.render_markdown_output,
                          pref_pair.second.text_output,
                          pref_pair.second.auto_new_chat_session_sec.has_value()
                              ? std::to_string(pref_pair.second.auto_new_chat_session_sec.value())
                              : mysqlx::nullvalue);
        }
        auto insert_result = insert.execute();
        auto inserted_count = insert_result.getAffectedItemsCount();

        spdlog::info("Batch insert or update of user preferences succeeded. Deleted: {}, Inserted: {}, Delta: {}",
                     deleted_count, inserted_count, (long long)inserted_count - (long long)deleted_count);
    } catch (const mysqlx::Error &e) {
        spdlog::error("Batch insert or update user preferences error at MySQL X DevAPI Error: {}", e.what());
    }
}

void DBConnection::insert_user_protait(qq_id_t id, const std::string &protait,
                                       const std::chrono::system_clock::time_point &create_time, double favorability) {
    try {
        get_user_protait_table()
            .insert("user_id", "protait", "create_time", "favorability")
            .values(std::to_string(id), protait, time_point_to_db_str(create_time), favorability)
            .execute();
        spdlog::info("Insert user protait successed.");
    } catch (const mysqlx::Error &e) {
        spdlog::error("Insert user protait error at MySQL X DevAPI Error: {}", e.what());
    }
}

std::vector<MessageRecord>
DBConnection::query_user_message(qq_id_t friend_id, size_t count_limit) {
    auto sql = get_message_record_table().select(
        "message_id", "content", "UNIX_TIMESTAMP(send_time) as send_time", "sender_id", "sender_name"
    );
    sql.where("sender_id = :friend_id")
        .bind("friend_id", std::to_string(friend_id));
    sql.orderBy("send_time DESC").limit(count_limit);
    mysqlx::RowResult sql_result = sql.execute();

    std::vector<MessageRecord> result;
    for (auto row : sql_result) {
        auto message_id_val = row[0];
        std::string content = row[1].get<std::string>();
        // Convert MySQL DATETIME to timestamp
        auto send_time = std::chrono::system_clock::from_time_t(
            row[2].get<uint64_t>());
        uint64_t sender_id = std::stoull(row[3].get<std::string>());
        std::string sender_name = row[4].get<std::string>();

        // Construct Sender with name and empty remark
        bot_adapter::Sender sender(sender_id, sender_name, std::nullopt);
        MessageRecord record(content, send_time, sender);
        if (!message_id_val.isNull()) {
            try {
                record.message_id_opt = std::stoull(message_id_val.get<std::string>());
            } catch (const std::exception& e) {
                spdlog::warn("Failed to convert message_id to integer: {}", e.what());
                record.message_id_opt = std::nullopt;
            }
        }
        result.push_back(std::move(record));
    }
    return result;
}

void DBConnection::insert_user_protait(const std::vector<std::pair<qq_id_t, UserProtait>> &user_protaits) {
    if (user_protaits.empty()) {
        return;
    }

    try {
        auto table = get_user_protait_table();
        auto insert = table.insert("user_id", "protait", "create_time", "favorability");
        for (const auto &protait_pair : user_protaits) {
            insert.values(std::to_string(protait_pair.first), protait_pair.second.protait,
                          time_point_to_db_str(protait_pair.second.create_time), protait_pair.second.favorability);
        }
        auto insert_result = insert.execute();
        auto inserted_count = insert_result.getAffectedItemsCount();

        spdlog::info("Batch insert of user protaits succeeded. Inserted: {}", inserted_count);
    } catch (const mysqlx::Error &e) {
        spdlog::error("Batch insert user protaits error at MySQL X DevAPI Error: {}", e.what());
    }
}