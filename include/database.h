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

namespace database {
    using mysqlx::SessionOption;

    constexpr std::string DEFAULT_MYSQL_SCHEMA_NAME = "aibot_800a";
    constexpr std::string DEFAULT_MESSAGE_RECORD_TABLE_NAME = "message_record";

    class DBConnection {
    public:
        DBConnection(const std::string &host, unsigned port, const std::string &user,
                     const std::string &password, const std::string &schema = DEFAULT_MYSQL_SCHEMA_NAME)
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
                                 ")",
                                 table_name))
                .execute();
            spdlog::info("Table '{}' created successfully.", table_name);
        }

        void insert_message(const std::string & content, const bot_adapter::Sender &sender,
                            const std::chrono::system_clock::time_point send_time) {
            try {
                if (const auto &group_sender = bot_adapter::try_group_sender(sender)) {
                    const auto &g = group_sender->get();
                    get_message_record_table()
                        .insert("sender_name", "sender_id", "content", "send_time", "group_name", "group_id",
                                "group_permission")
                        .values(g.name, g.id, content, time_point_to_db_str(send_time), g.group.name, g.group.id, g.group.permission)
                        .execute();
                } else {
                    get_message_record_table()
                        .insert("sender_name", "sender_id", "content", "send_time")
                        .values(sender.name, sender.id, content, time_point_to_db_str(send_time))
                        .execute();
                }
                spdlog::info("Insert message successed.");
            } catch (const mysqlx::Error &e) {
                spdlog::error("Insert message error at MySQL X DevAPI Error: {}", e.what());
            }
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
} // namespace MySql

#endif