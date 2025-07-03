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

void DBConnection::insert_or_update_user_preferences(
    const std::vector<std::pair<qq_id_t, UserPreference>> &user_preferences) {
    if (user_preferences.empty()) {
        return;
    }

    try {
        auto table = get_user_preference_table();

        std::vector<qq_id_t> user_ids;
        user_ids.reserve(user_preferences.size());
        for (const auto &pref_pair : user_preferences) {
            user_ids.push_back(pref_pair.first);
        }

        std::string user_ids_list_str =
            wheel::join_str(user_ids.cbegin(), user_ids.cend(), ",", [](const qq_id_t id) { return std::to_string(id); });
        auto delete_result = table.remove().where("user_id IN (" + user_ids_list_str + ")").execute();
        auto deleted_count = delete_result.getAffectedItemsCount();

        auto insert = table.insert("user_id", "render_markdown_output", "text_output");
        for (const auto &pref_pair : user_preferences) {
            insert.values(pref_pair.first, pref_pair.second.render_markdown_output, pref_pair.second.text_output);
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
                                       const std::chrono::system_clock::time_point &create_time) {
    try {
        get_user_protait_table().insert("user_id", "protait", "create_time").values(id, protait, time_point_to_db_str(create_time)).execute();
        spdlog::info("Insert user protait successed.");
    } catch (const mysqlx::Error &e) {
        spdlog::error("Insert user protait error at MySQL X DevAPI Error: {}", e.what());
    }
}

void DBConnection::insert_user_protait(const std::vector<std::pair<qq_id_t, UserProtait>> &user_protaits) {
    if (user_protaits.empty()) {
        return;
    }

    try {
        auto table = get_user_protait_table();
        auto insert = table.insert("user_id", "protait", "create_time");
        for (const auto &protait_pair : user_protaits) {
            insert.values(protait_pair.first, protait_pair.second.protait,
                          time_point_to_db_str(protait_pair.second.create_time));
        }
        auto insert_result = insert.execute();
        auto inserted_count = insert_result.getAffectedItemsCount();

        spdlog::info("Batch insert of user protaits succeeded. Inserted: {}", inserted_count);
    } catch (const mysqlx::Error &e) {
        spdlog::error("Batch insert user protaits error at MySQL X DevAPI Error: {}", e.what());
    }
}