#ifndef ADAPTER_MODEL_H
#define ADAPTER_MODEL_H

#include "utils.h"
#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
namespace bot_adapter {
    /**
     * @brief Represents a message sender in the chat system
     *
     * Contains information about a user who can send messages,
     * including their identification, permissions, and activity timestamps.
     */
    struct Sender {
        // Unique identifier for the sender
        uint64_t id;
        // Display name of the sender
        std::string name;
        // Remark
        std::optional<std::string> remark;

        /**
         * @brief Constructs a new Sender object
         *
         * @param id Unique identifier
         * @param name Display name
         * @param remark Remark
         */
        Sender(uint64_t id, std::string name, std::optional<std::string> remark)
            : id(id), name(std::move(name)), remark(remark) {}
    };

    /**
     * @brief Represents a group message sender with additional group-specific information
     *
     * Extends the basic Sender with group-related metadata like join time and last activity.
     */
    struct GroupSender : public Sender {
        // Permission level in the group (e.g., "MEMBER", "ADMINISTRATOR")
        std::string permission;
        // When the sender joined the group (empty if unknown)
        std::optional<std::chrono::system_clock::time_point> join_time;
        // When the sender last spoke in the group
        std::chrono::system_clock::time_point last_speak_time;

        /**
         * @brief Constructs a new GroupSender object
         *
         * @param id Unique identifier
         * @param name Display name
         * @param remark Optional remark about the sender
         * @param permission Permission level in group
         * @param join_time Optional join time in group
         * @param last_speak_time Last speaking time in group
         */
        GroupSender(uint64_t id, std::string name, std::optional<std::string> remark, std::string permission,
                    std::optional<std::chrono::system_clock::time_point> join_time,
                    std::chrono::system_clock::time_point last_speak_time)
            : Sender(id, std::move(name), std::move(remark)), permission(std::move(permission)), join_time(join_time),
              last_speak_time(last_speak_time) {}

        /**
         * @brief Constructs from JSON data
         *
         * @param sender JSON object containing sender information
         */
        GroupSender(const nlohmann::json &sender)
            : Sender(get_optional<uint64_t>(sender, "id").value_or(0),
                     get_optional<std::string>(sender, "memberName").value_or(""),
                     get_optional<std::string>(sender, "remark").value_or("")),
              permission(get_optional<std::string>(sender, "permission").value_or("")),
              join_time([&sender]() -> std::optional<std::chrono::system_clock::time_point> {
                  auto join_timestamp = get_optional<time_t>(sender, "joinTimestamp");
                  if (join_timestamp) {
                      return std::chrono::system_clock::from_time_t(*join_timestamp);
                  }
                  return std::nullopt;
              }()),
              last_speak_time(std::chrono::system_clock::from_time_t(
                  get_optional<time_t>(sender, "lastSpeakTimeStamp").value_or(0))) {}
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

        /**
         * @brief Constructs a new Group object from JSON data
         *
         * @param group JSON object containing group information
         */
        Group(const nlohmann::json::value_type &group)
            : id(get_optional<uint64_t>(group, "id").value_or(0)),
              name(get_optional<std::string>(group, "name").value_or("")),
              permission(get_optional<std::string>(group, "permission").value_or("")) {}
    };
} // namespace bot_adapter

#endif