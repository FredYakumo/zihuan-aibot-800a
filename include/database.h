#ifndef DATABASE_H
#define DATABASE_H

#include "adapter_model.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <fmt/format.h>
#include <mysqlx/xdevapi.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <vector>

namespace database {
    using mysqlx::SessionOption;

    constexpr std::string DEFAULT_MYSQL_SCHEMA_NAME = "aibot_800a";
    constexpr std::string DEFAULT_MESSAGE_RECORD_TABLE_NAME = "message_record";
    constexpr std::string DEFAULT_OPTIM_MESSAGE_RESULT_TABLE = "optim_message_result";

    struct GroupMessageRecord {
        std::string content;
        std::chrono::system_clock::time_point send_time;
        bot_adapter::GroupSender sender;

        GroupMessageRecord(const std::string_view content, const std::chrono::system_clock::time_point &send_time,
                           const bot_adapter::GroupSender &sender)
            : content(content), send_time(send_time), sender(sender) {}
    };

    class DBConnection {
      public:
        DBConnection(const std::string &host, unsigned port, const std::string &user, const std::string &password,
                     const std::string &schema = DEFAULT_MYSQL_SCHEMA_NAME)
            : session(SessionOption::HOST, host, SessionOption::PORT, port, SessionOption::USER, user,
                      SessionOption::PWD, password, SessionOption::DB, schema),
              schema(session.getSchema(std::string(schema), true)) {}

        void create_message_record_table(const std::string &table_name = DEFAULT_MESSAGE_RECORD_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 "  sender_name VARCHAR(255) NOT NULL, "
                                 "  sender_id VARCHAR(255) NOT NULL, "
                                 "  content TEXT NOT NULL, "
                                 "  send_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                                 "  group_name VARCHAR(255) NULL, "
                                 "  group_id VARCHAR(255) NULL, "
                                 "  group_permission VARCHAR(255) NULL"
                                 "  at_target_list TEXT NULL"
                                 ")",
                                 table_name))
                .execute();
            spdlog::info("Table '{}' created successfully.", table_name);
        }

        void create_optim_message_result_table(const std::string &table_name = DEFAULT_OPTIM_MESSAGE_RESULT_TABLE) {
            session.sql(fmt::format(R"(
                CREATE TABLE IF NOT EXISTS {} (
                    sender_name VARCHAR(255) NOT NULL,
                    sender_id VARCHAR(255) NOT NULL,
                    origin_chat_session_view TEXT NOT NULL,
                    
                )
                )", table_name));
        }

        void insert_message(const std::string &content, const bot_adapter::Sender &sender,
                            const std::chrono::system_clock::time_point send_time,
                            const std::optional<std::set<uint64_t>> at_target_set = std::nullopt) {
            try {
                const auto at_target_value = (at_target_set && !at_target_set->empty())
                                                 ? join_str(std::cbegin(*at_target_set), std::cend(*at_target_set), ",",
                                                            [](const auto i) { return std::to_string(i); })
                                                 : mysqlx::nullvalue;
                if (const auto &group_sender = bot_adapter::try_group_sender(sender)) {
                    const auto &g = group_sender->get();
                    get_message_record_table()
                        .insert("sender_name", "sender_id", "content", "send_time", "group_name", "group_id",
                                "group_permission", "at_target_list")
                        .values(g.name, g.id, content, time_point_to_db_str(send_time), g.group.name, g.group.id,
                                g.permission, at_target_value)
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

        std::vector<GroupMessageRecord> query_group_user_message(uint64_t sender_id, uint64_t group_id,
                                                                 size_t count_limit = 10) {
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

      private:
        mysqlx::Session session;
        mysqlx::Schema schema;
        std::optional<mysqlx::Table> message_record_table = std::nullopt;

        mysqlx::Table &get_message_record_table() {
            if (!message_record_table) {
                message_record_table = schema.getTable(std::string(DEFAULT_MESSAGE_RECORD_TABLE_NAME), true);
            }
            return *message_record_table;
        }
    };

    void init_db_connection();
    DBConnection &get_global_db_connection();
} // namespace database

#endif