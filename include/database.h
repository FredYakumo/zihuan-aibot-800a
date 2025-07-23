#ifndef DATABASE_H
#define DATABASE_H

#include "adapter_model.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "user_protait.h"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <fmt/format.h>
#include <mysqlx/xdevapi.h>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <time_utils.h>
#include <vector>
#include <functional>
#include <thread>

namespace database {
    using mysqlx::SessionOption;

    constexpr std::string_view DEFAULT_MYSQL_SCHEMA_NAME = "aibot_800a";
    constexpr std::string_view DEFAULT_MESSAGE_RECORD_TABLE_NAME = "message_record";
    constexpr std::string_view DEFAULT_TOOLS_CALL_RECORD_TABLE = "tool_calls_record";
    constexpr std::string_view DEFAULT_USER_CHAT_PROMPT_TABLE_NAME = "user_chat_prompt";
    constexpr std::string_view DEFAULT_USER_PORTAIT_TABLE_NAME = "user_protait";
    constexpr std::string_view DEFAULT_USER_PREFERENCE_TABLE_NAME = "user_preference";

    struct MessageRecord {
        std::optional<message_id_t> message_id_opt;
        std::string content;
        std::chrono::system_clock::time_point send_time;
        bot_adapter::Sender sender;

        MessageRecord(const std::string_view content, const std::chrono::system_clock::time_point &send_time,
                      const bot_adapter::Sender &sender)
            : content(content), send_time(send_time), sender(sender) {}
    };

    struct GroupMessageRecord : public MessageRecord {
        bot_adapter::GroupSender group_sender;

        GroupMessageRecord(const std::string_view content, const std::chrono::system_clock::time_point &send_time,
                           const bot_adapter::GroupSender &sender)
            : MessageRecord(content, send_time, sender), group_sender(sender) {}
    };

    struct ToolCallsRecord {
        std::string sender_name;
        qq_id_t sender_id;
        std::string origin_chat_session_view;
        std::chrono::system_clock::time_point send_time;
        std::string tool_calls;
        std::string tool_calls_content;
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
            return fmt::format(
                "偏好:\n- 输出渲染: {}\n- 输出文本: {}\n- 自动新对话: {}", render_markdown_output, text_output,
                auto_new_chat_session_sec.has_value() ? std::to_string(auto_new_chat_session_sec.value()) : "null");
        }
    };

    class DBConnection {
      public:
        DBConnection(const std::string &host, unsigned port, const std::string &user, const std::string &password,
                     const std::string_view schema = DEFAULT_MYSQL_SCHEMA_NAME)
            : session(SessionOption::HOST, host, SessionOption::PORT, port, SessionOption::USER, user,
                      SessionOption::PWD, password, SessionOption::DB, std::string(schema)),
              schema(session.getSchema(std::string(schema), true)) {}

        void create_message_record_table(const std::string_view table_name = DEFAULT_MESSAGE_RECORD_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 "  sender_name VARCHAR(255) NOT NULL, "
                                 "  sender_id VARCHAR(255) NOT NULL, "
                                 "  content TEXT NOT NULL, "
                                 "  send_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                                 "  group_name VARCHAR(255) NULL, "
                                 "  group_id VARCHAR(255) NULL, "
                                 "  group_permission VARCHAR(255) NULL,"
                                 "  at_target_list TEXT NULL,"
                                 "  message_id VARCHAR(255) NULL,"
                                 "  INDEX idx_sender_id (sender_id),"
                                 "  INDEX idx_sender_name (sender_name),"
                                 "  INDEX idx_send_time (send_time),"
                                 "  INDEX idx_group_id (group_id),"
                                 "  INDEX idx_group_name (group_name)"
                                 ");",
                                 table_name))
                .execute();
            spdlog::info("Table '{}' created successfully.", table_name);
        }

        void create_user_protait_table(const std::string_view table_name = DEFAULT_USER_PORTAIT_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 " user_id varchar(255) NOT NULL,"
                                 " favorability double NOT NULL,"
                                 " protait TEXT NOT NULL,"
                                 " create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
                                 ")",
                                 table_name))
                .execute();
        }

        void create_user_preference_table(const std::string_view table_name = DEFAULT_USER_PREFERENCE_TABLE_NAME) {
            session
                .sql(fmt::format("CREATE TABLE IF NOT EXISTS {} ("
                                 " user_id varchar(255) NOT NULL PRIMARY KEY,"
                                 " render_markdown_output tinyint(4) NOT NULL DEFAULT 1,"
                                 " text_output tinyint(4) NOT NULL DEFAULT 0,"
                                 " auto_new_chat_session int default null"
                                 ")",
                                 table_name))
                .execute();
        }

        void create_tools_call_record_table(const std::string_view table_name = DEFAULT_TOOLS_CALL_RECORD_TABLE) {
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
            try {
                auto table = get_user_preference_table();
                auto result = table.select("render_markdown_output", "text_output", "auto_new_chat_session")
                                  .where("user_id = :user_id")
                                  .bind("user_id", std::to_string(id))
                                  .execute();

                if (auto row = result.fetchOne()) {
                    // The C++ connector returns tinyint as integer.
                    UserPreference pref{
                        static_cast<bool>(row[0].get<int>()), static_cast<bool>(row[1].get<int>()),
                        std::nullopt // 默认为 nullopt
                    };

                    // 如果 auto_new_chat_session 不为 NULL，则设置具体值
                    if (!row[2].isNull()) {
                        pref.auto_new_chat_session_sec = row[2].get<int64_t>();
                    }

                    return pref;
                }
                return std::nullopt;
            } catch (const mysqlx::Error &e) {
                spdlog::error("Failed to get user preference for user {}: {}", id, e.what());
                return std::nullopt;
            }
        }

        /**
         * @brief Get a list of user protait records by user id, ordered by create_time.
         * @param id User id
         * @param limit The maximum number of records to return (default 1)
         * @return std::vector<UserProtait>
         */
        std::vector<UserProtait> get_user_protait(qq_id_t id, size_t limit = 1) {
            auto &table = get_user_protait_table();
            auto result = table.select("protait", "favorability", "create_time")
                              .where("user_id = :user_id")
                              .orderBy("create_time DESC")
                              .limit(static_cast<int>(limit))
                              .bind("user_id", std::to_string(id))
                              .execute();
            std::vector<UserProtait> protaits;
            for (auto row : result) {
                protaits.emplace_back(UserProtait{row[0].get<std::string>(), row[1].get<double>(),
                                                  db_str_to_time_point(row[2].get<std::string>())});
            }
            return protaits;
        }

        void insert_user_protait(qq_id_t id, const std::string &protait,
                                 const std::chrono::system_clock::time_point &create_time, double favorability);

        void insert_user_protait(const std::vector<std::pair<qq_id_t, UserProtait>> &user_protaits);

        void insert_or_update_user_preferences(const std::vector<std::pair<qq_id_t, UserPreference>> &user_preferences);

        void insert_message(message_id_t message_id, const std::string &content, const bot_adapter::Sender &sender,
                            const std::chrono::system_clock::time_point send_time,
                            const std::optional<std::set<uint64_t>> at_target_set = std::nullopt);

        void insert_tool_calls_record(const std::string &sender_name, qq_id_t sender_id,
                                      const std::string &origin_chat_session_view,
                                      const std::chrono::system_clock::time_point &send_time,

                                      const std::string &tool_calls, const std::string &tool_calls_content);

        std::vector<MessageRecord> query_user_message(qq_id_t friend_id, size_t count_limit = 10);
        
        std::vector<GroupMessageRecord> query_group_message(qq_id_t group_id,
                                                            std::optional<qq_id_t> filter_sender = std::nullopt,
                                                            size_t count_limit = 10);

      private:
        mysqlx::Session session;
        mysqlx::Schema schema;
        std::optional<mysqlx::Table> message_record_table = std::nullopt;
        std::optional<mysqlx::Table> tool_calls_record_table = std::nullopt;
        std::optional<mysqlx::Table> user_protait_table = std::nullopt;

        /**
         * @brief Generic table retrieval method with retry mechanism for handling connection issues
         * @param table_name The name of the table to retrieve
         * @return mysqlx::Table object for the specified table
         */
        inline mysqlx::Table get_table_with_retry(const std::string &table_name) {
            try {
                return schema.getTable(table_name, true);
            } catch (const mysqlx::Error &e) {
                spdlog::error("Failed to get table '{}': {}. Attempting to refresh schema...", table_name, e.what());
                try {
                    // Refresh schema to handle potential session issues
                    schema = session.getSchema(schema.getName(), true);
                    return schema.getTable(table_name, true);
                } catch (const mysqlx::Error &retry_e) {
                    spdlog::error("Failed to get table '{}' after schema refresh: {}", table_name, retry_e.what());
                    throw;
                }
            }
        }

        /**
         * @brief Execute database operation with retry mechanism for connection issues
         * @param operation Lambda function that performs the database operation
         * @param operation_name Description of the operation for logging
         * @param reset_table_cache Function to reset cached table references (optional)
         */
        template<typename Operation>
        void execute_with_retry(Operation&& operation, const std::string& operation_name, 
                               std::function<void()> reset_cache = [](){}) {
            const int max_retries = 3;
            int retry_count = 0;
            
            while (retry_count < max_retries) {
                try {
                    operation();
                    return; // Success, exit the retry loop
                } catch (const mysqlx::Error &e) {
                    retry_count++;
                    const std::string error_msg = e.what();
                    
                    // Check if this is a connection-related error that might benefit from retry
                    const bool is_connection_error = error_msg.find("CDK Error") != std::string::npos ||
                                                   error_msg.find("incorrect resume") != std::string::npos ||
                                                   error_msg.find("Connection") != std::string::npos ||
                                                   error_msg.find("timeout") != std::string::npos;
                    
                    if (is_connection_error && retry_count < max_retries) {
                        spdlog::warn("{} failed (attempt {}/{}), retrying... Error: {}", 
                                   operation_name, retry_count, max_retries, error_msg);
                        
                        // Reset cached table references to force reconnection
                        reset_cache();
                        
                        // Small delay before retry
                        std::this_thread::sleep_for(std::chrono::milliseconds(100 * retry_count));
                        continue;
                    } else {
                        spdlog::error("{} error at MySQL X DevAPI Error (final attempt {}/{}): {}", 
                                    operation_name, retry_count, max_retries, error_msg);
                        break;
                    }
                }
            }
        }

        inline mysqlx::Table &get_message_record_table() {
            if (!message_record_table) {
                message_record_table = get_table_with_retry(std::string(DEFAULT_MESSAGE_RECORD_TABLE_NAME));
            }
            return *message_record_table;
        }

        inline mysqlx::Table &get_tool_calls_record_table() {
            if (!tool_calls_record_table) {
                tool_calls_record_table = get_table_with_retry(std::string(DEFAULT_TOOLS_CALL_RECORD_TABLE));
            }
            return *tool_calls_record_table;
        }

        inline mysqlx::Table get_user_preference_table() {
            return get_table_with_retry(std::string(DEFAULT_USER_PREFERENCE_TABLE_NAME));
        }

        inline mysqlx::Table &get_user_protait_table() {
            if (!user_protait_table) {
                user_protait_table = get_table_with_retry(std::string(DEFAULT_USER_PORTAIT_TABLE_NAME));
            }
            return *user_protait_table;
        }
    };

    void init_db_connection();
    DBConnection &get_global_db_connection();
} // namespace database

#endif