#ifndef DATABASE_H
#define DATABASE_H

#include "adapter_model.h"
#include "constants.hpp"
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
    constexpr std::string DEFAULT_TOOLS_CALL_RECORD_TABLE = "tool_calls_record";
    constexpr std::string DEFAULT_USER_CHAT_PROMPT_TABLE_NAME = "user_chat_prompt";
    constexpr std::string DEFAULT_USER_PORTAIT_TABLE_NAME = "user_protait";

    struct GroupMessageRecord {
        std::string content;
        std::chrono::system_clock::time_point send_time;
        bot_adapter::GroupSender sender;

        GroupMessageRecord(const std::string_view content, const std::chrono::system_clock::time_point &send_time,
                           const bot_adapter::GroupSender &sender)
            : content(content), send_time(send_time), sender(sender) {}
    };

    struct ToolCallsRecord {
        std::string sender_name;
        qq_id_t sender_id;
        std::string origin_chat_session_view;
        std::chrono::system_clock::time_point send_time;
        std::string tool_calls;
        std::string tool_calls_content;
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
                                 ");",
                                 table_name))
                .execute();
            spdlog::info("Table '{}' created successfully.", table_name);
        }

        void create_tools_call_record_table(const std::string &table_name = DEFAULT_TOOLS_CALL_RECORD_TABLE) {
            session
                .sql(fmt::format(R"(
                CREATE TABLE IF NOT EXISTS {} (
                    sender_name VARCHAR(255) NOT NULL,
                    sender_id VARCHAR(255) NOT NULL,
                    origin_chat_session_view TEXT NOT NULL,
                    send_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    tool_calls TEXT NOT NULL,
                    tool_calls_content TEXT NOT NULL
                );
                )",
                                 table_name))
                .execute();
            spdlog::info("Table '{}' created successfully.", table_name);
        }

        void insert_message(const std::string &content, const bot_adapter::Sender &sender,
                            const std::chrono::system_clock::time_point send_time,
                            const std::optional<std::set<uint64_t>> at_target_set = std::nullopt);

        void insert_tool_calls_record(const std::string &sender_name, qq_id_t sender_id,
                                      const std::string &origin_chat_session_view,
                                      const std::chrono::system_clock::time_point &send_time,

                                      const std::string &tool_calls, const std::string &tool_calls_content);

        std::vector<GroupMessageRecord> query_group_user_message(uint64_t sender_id, uint64_t group_id,
                                                                 size_t count_limit = 10);

      private:
        mysqlx::Session session;
        mysqlx::Schema schema;
        std::optional<mysqlx::Table> message_record_table = std::nullopt;
        std::optional<mysqlx::Table> tool_calls_record_table = std::nullopt;

        mysqlx::Table &get_message_record_table() {
            if (!message_record_table) {
                message_record_table = schema.getTable(std::string(DEFAULT_MESSAGE_RECORD_TABLE_NAME), true);
            }
            return *message_record_table;
        }

        mysqlx::Table &get_tool_calls_record_table() {
            if (!tool_calls_record_table) {
                tool_calls_record_table = schema.getTable(std::string(DEFAULT_TOOLS_CALL_RECORD_TABLE), true);
            }
            return *tool_calls_record_table;
        }
    };

    void init_db_connection();
    DBConnection &get_global_db_connection();
} // namespace database

#endif