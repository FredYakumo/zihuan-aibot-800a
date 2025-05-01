#ifndef MYSQL_CONNECTOR_H
#define MYSQL_CONNECTOR_H

#include <fmt/format.h>
#include <mysqlx/xdevapi.h>
#include <string_view>
#include <spdlog/spdlog.h>

namespace MySql {
    using mysqlx::SessionOption;

    constexpr std::string_view DEFAULT_MYSQL_SCHEMA_NAME = "aibot_800a";
    constexpr std::string_view DEFAULT_GROUP_MESSAGE_TABLE_NAME = "group_message";

    class DBConnection {
        DBConnection(const std::string_view host, uint16_t port, const std::string_view user,
                     const std::string_view password, const std::string_view schema = DEFAULT_MYSQL_SCHEMA_NAME)
            : session(SessionOption::HOST, host, SessionOption::PORT, port, SessionOption::USER, user,
                      SessionOption::PWD, password, SessionOption::DB, schema),
              schema(session.getSchema(std::string(schema), true)) {}

        void create_group_message_table(const std::string_view table_name = DEFAULT_GROUP_MESSAGE_TABLE_NAME) {
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

      private:
        mysqlx::Session session;
        mysqlx::Schema schema;
    };
} // namespace MySql

#endif