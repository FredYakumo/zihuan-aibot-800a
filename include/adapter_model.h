#ifndef ADAPTER_MODEL_H
#define ADAPTER_MODEL_H

#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
namespace bot_adapter {
    /**
     * @brief Represents a message sender in the chat system
     *
     * Contains information about a user who can send messages,
     * including their identification, permissions, and activity timestamps.
     */
    struct Sender {
        uint64_t id;                                                    ///< Unique identifier for the sender
        std::string name;                                               ///< Display name of the sender
        std::string permission;                                         ///< Permission level (e.g., "admin", "user")
        std::chrono::system_clock::time_point last_speak_time;          ///< When the sender last spoke
        std::optional<std::chrono::system_clock::time_point> join_time; ///< Optional join time if known

        /**
         * @brief Constructs a new Sender object
         *
         * @param id Unique identifier
         * @param name Display name
         * @param permission Permission level
         * @param last_speak_time When the sender last spoke
         * @param join_time Optional join time (defaults to nullopt)
         */
        Sender(uint64_t id, std::string name, std::string permission,
               std::chrono::system_clock::time_point last_speak_time,
               std::optional<std::chrono::system_clock::time_point> join_time = std::nullopt)
            : id(id), name(std::move(name)), permission(std::move(permission)), last_speak_time(last_speak_time),
              join_time(join_time) {}

        Sender(const nlohmann::json::value_type &sender) {
            auto join_timestamp = sender["joinTimestamp"];
            Sender{sender["id"], sender["memberName"], sender["permission"],
                   std::chrono::system_clock::from_time_t(sender["lastSpeakTimeStamp"]),
                   join_timestamp.is_null()
                       ? std::nullopt
                       : std::make_optional(std::chrono::system_clock::from_time_t(join_timestamp))};
        }
    };

    /**
     * @brief Represents a chat group
     *
     * Contains information about a group where messages can be sent,
     * including its identification and permission settings.
     */
    struct Group {
        uint64_t id;            ///< Unique identifier for the group
        std::string name;       ///< Display name of the group
        std::string permission; ///< Default permission level for the group

        /**
         * @brief Constructs a new Group object
         *
         * @param id Unique identifier
         * @param name Display name
         * @param permission Default permission level
         */
        Group(uint64_t id, std::string name, std::string permission)
            : id(id), name(std::move(name)), permission(std::move(permission)) {}
    };
} // namespace bot_adapter

#endif