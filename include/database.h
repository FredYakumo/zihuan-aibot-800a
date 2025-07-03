#ifndef DATABASE_H
#define DATABASE_H

#include "adapter_model.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <fmt/format.h>
#include <mysqlx/xdevapi.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace database {
    using mysqlx::SessionOption;

    constexpr std::string DEFAULT_MYSQL_SCHEMA_NAME = "aibot_800a";
    constexpr std::string DEFAULT_MESSAGE_RECORD_TABLE_NAME = "message_record";
    constexpr std::string DEFAULT_TOOLS_CALL_RECORD_TABLE = "tool_calls_record";
    constexpr std::string DEFAULT_USER_CHAT_PROMPT_TABLE_NAME = "user_chat_prompt";
    constexpr std::string DEFAULT_USER_PORTAIT_TABLE_NAME = "user_protait";
    constexpr std::string DEFAULT_USER_PREFERENCE_TABLE_NAME = "user_preference";

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

    struct UserProtait {
        std::string protait;
        std::chrono::system_clock::time_point create_time;
    };

    struct UserPreference {
        bool render_markdown_output = true;
        bool text_output = false;
        std::optional<int64_t> auto_new_chat_session_sec = 600;

        /**
         * @brief Returns a string representation of the UserPreference object.
         *
         * @return std::string
         */
        std::string to_string() const {
            return fmt::format("偏好:\n- 输出渲染: {}\n- 输出文本: {}\n- 自动新对话: {}",
                               render_markdown_output, text_output,
                               auto_new_chat_session_sec.has_value() ? std::to_string(auto_new_chat_session_sec.value())
                                                                   : "null");
        }
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

        void create_user_protait_table(const std::string &table_name = DEFAULT_USER_PORTAIT_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 " user_id int NOT NULL,"
                                 " protait TEXT NOT NULL,"
                                 " create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
                                 ")",
                                 table_name))
                .execute();
        }

        void create_user_preference_table(const std::string &table_name = DEFAULT_USER_PREFERENCE_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 " user_id int NOT NULL PRIMARY KEY,"
                                 " render_markdown_output tinyint(4) NOT NULL DEFAULT 1,"
                                 " text_output tinyint(4) NOT NULL DEFAULT 0,"
                                 " auto_new_chat_session int default null"
                                 ")",
                                 table_name))
                .execute();
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

        std::optional<UserPreference> get_user_preference(qq_id_t id) {
            auto &table = get_user_preference_table();
            auto result = table.select("render_markdown_output", "text_output")
                              .where("user_id = :user_id")
                              .bind("user_id", id)
                              .execute();

            if (auto row = result.fetchOne()) {
                // The C++ connector returns tinyint as integer.
                return UserPreference{static_cast<bool>(row[0].get<int>()), static_cast<bool>(row[1].get<int>())};
            }
            return std::nullopt;
        }

        std::optional<UserProtait> get_user_protait(qq_id_t id) {
            auto &table = get_user_protait_table();
            auto result =
                table.select("protait", "create_time").where("user_id = :user_id").bind("user_id", id).execute();

            if (auto row = result.fetchOne()) {
                return UserProtait{row[0].get<std::string>(), row[1].get<std::chrono::system_clock::time_point>()};
            }
            return std::nullopt;
        }

        void insert_user_protait(qq_id_t id, const std::string &protait,
                                 const std::chrono::system_clock::time_point &create_time);

        void insert_user_protait(const std::vector<std::pair<qq_id_t, UserProtait>> &user_protaits);

        void insert_or_update_user_preferences(const std::vector<std::pair<qq_id_t, UserPreference>> &user_preferences);

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
        std::optional<mysqlx::Table> user_preference_table = std::nullopt;
        std::optional<mysqlx::Table> user_protait_table = std::nullopt;

        inline mysqlx::Table &get_message_record_table() {
            if (!message_record_table) {
                message_record_table = schema.getTable(std::string(DEFAULT_MESSAGE_RECORD_TABLE_NAME), true);
            }
            return *message_record_table;
        }

        inline mysqlx::Table &get_tool_calls_record_table() {
            if (!tool_calls_record_table) {
                tool_calls_record_table = schema.getTable(std::string(DEFAULT_TOOLS_CALL_RECORD_TABLE), true);
            }
            return *tool_calls_record_table;
        }

        inline mysqlx::Table &get_user_preference_table() {
            if (!user_preference_table) {
                user_preference_table = schema.getTable(std::string(DEFAULT_USER_PREFERENCE_TABLE_NAME), true);
            }
            return *user_preference_table;
        }

        inline mysqlx::Table &get_user_protait_table() {
            if (!user_protait_table) {
                user_protait_table = schema.getTable(std::string(DEFAULT_USER_PORTAIT_TABLE_NAME), true);
            }
            return *user_protait_table;
        }
    };

    void init_db_connection();
    DBConnection &get_global_db_connection();
} // namespace database

#endif