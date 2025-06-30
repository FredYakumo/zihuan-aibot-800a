#ifndef ADAPTER_MODEL_H
#define ADAPTER_MODEL_H

#include "constant_types.hpp"
#include "constants.hpp"
#include "get_optional.hpp"
#include "neural_network/text_model.h"
#include <chrono>
#include <collection/concurrent_vector.hpp>
#include <cstdint>
#include <fmt/format.h>
#include <general-wheel-cpp/collection/concurrent_unordered_map.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace bot_adapter {
    /**
     * @brief Represents a message sender in the chat system
     *
     * Contains information about a user who can send messages,
     * including their identification, permissions, and activity timestamps.
     */
    /**
     * @brief Represents a message sender in the chat system
     *
     * Contains information about a user who can send messages,
     * including their unique identifier, display name, and optional remark.

     */

    struct Sender {
        // QQ号/id
        qq_id_t id;
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

        /**
         * @brief Constructs a new Sender object from JSON datas
         */
        Sender(const nlohmann::json &sender)
            : id(get_optional<uint64_t>(sender, "id").value_or(0)),
              name(get_optional<std::string>(sender, "name")
                       .value_or(get_optional<std::string>(sender, "nickname").value_or(""))),
              remark(get_optional<std::string>(sender, "remark")) {}

        virtual nlohmann::json to_json() const {
            nlohmann::json js = {{"id", id}, {"name", name}};
            if (const auto &r = remark) {
                js["remark"] = *r;
            }
            return js;
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

        /**
         * @brief Constructs a new Group object from JSON data
         *
         * @param group JSON object containing group information
         */
        Group(const nlohmann::json::value_type &group)
            : id(get_optional<uint64_t>(group, "id").value_or(0)),
              name(get_optional<std::string>(group, "name").value_or("")),
              permission(get_optional<std::string>(group, "permission").value_or("")) {}

        /**
         * @brief Implicit conversion operator to std::string
         *
         * @return std::string String representation of the Group object
         */
        operator std::string() const { return fmt::format("[Group]{}({})", name, id); }

        /**
         * @brief Converts Group object to JSON representation
         *
         * @return nlohmann::json JSON object containing group information
         */
        nlohmann::json to_json() const {
            return nlohmann::json{{"id", id}, {"name", name}, {"permission", permission}};
        }
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

        Group group;

        /**
         * @brief Constructs a new GroupSender object
         *
         * @param id Unique identifier
         * @param name Display name
         * @param remark Optional remark about the sender
         * @param permission Permission level in group
         * @param join_time Optional join time in group
         * @param last_speak_time Last speaking time in group
         * @param group Group where sender send.
         */
        GroupSender(uint64_t id, std::string name, std::optional<std::string> remark, std::string permission,
                    std::optional<std::chrono::system_clock::time_point> join_time,
                    std::chrono::system_clock::time_point last_speak_time, Group group)
            : Sender(id, std::move(name), std::move(remark)), permission(std::move(permission)), join_time(join_time),
              last_speak_time(last_speak_time), group(std::move(group)) {}

        /**
         * @brief Constructs from JSON data
         *
         * @param sender JSON object containing sender information
         */
        GroupSender(const nlohmann::json &sender, const nlohmann::json &group)
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
                  get_optional<time_t>(sender, "lastSpeakTimeStamp").value_or(0))),
              group(group) {}

        /**
         * @brief Converts GroupSender object to JSON representation
         *
         * @return nlohmann::json JSON object containing sender and group information
         */
        nlohmann::json to_json() const override {
            nlohmann::json js = Sender::to_json(); // Get base sender info

            // Add group-specific information
            js["permission"] = permission;

            // Convert time points to timestamps if they exist
            if (join_time) {
                js["joinTimestamp"] = std::chrono::system_clock::to_time_t(*join_time);
            }

            js["lastSpeakTimeStamp"] = std::chrono::system_clock::to_time_t(last_speak_time);

            // Add the complete group information
            js["group"] = group.to_json();

            return js;
        }
    };

    inline std::optional<std::reference_wrapper<const GroupSender>> try_group_sender(const Sender &sender) {
        try {
            const GroupSender &group_sender = dynamic_cast<const GroupSender &>(sender);
            return std::cref(group_sender);
        } catch (const std::bad_cast &) {
            return std::nullopt;
        }
    }

    enum class ProfileSex { UNKNOWN = 0, MALE, FEMALE };

    // String to enum conversion
    inline ProfileSex from_string(const std::string_view str) {
        if (str == "MALE" || str == "male")
            return ProfileSex::MALE;
        if (str == "FEMALE" || str == "female")
            return ProfileSex::FEMALE;
        return ProfileSex::UNKNOWN; // default case
    }

    inline std::string to_chs_string(const ProfileSex profile_sex) {
        switch (profile_sex) {
        case ProfileSex::FEMALE:
            return "女";
        case ProfileSex::MALE:
            return "男";
        default:
            return "未知";
        }
    }

    /**
     * @brief Represents a user profile containing personal information.
     */
    struct Profile {
        uint64_t id;       ///< QQ ID
        std::string name;  ///< Full name of the user
        std::string email; ///< Email address of the user
        uint32_t age;      ///< Age of the user in years
        uint32_t level;    ///< User level or rank
        ProfileSex sex;    ///< Gender of the user

        Profile() = default;

        /**
         * @brief Constructs a new Profile object
         *
         * @param id_ QQ Id
         * @param name_ Full name of the user
         * @param email_ Email address of the user
         * @param age_ Age of the user in years
         * @param level_ User level or rank
         * @param sex_ Gender of the user
         */
        Profile(uint64_t id_, const std::string &name_, const std::string &email_, uint32_t age_, uint32_t level_,
                ProfileSex sex_)
            : id(id_), name(name_), email(email_), age(age_), level(level_), sex(sex_) {}
    };

    /**
     * @brief Converts a bot_adapter::Sender object to its string representation
     * @param sender The sender object to convert
     * @return A formatted string containing the sender's name and ID in the format "[Sender]name(id)"
     */
    inline std::string to_string(const bot_adapter::Sender &sender) {
        return fmt::format("[Sender]{}({})", sender.name, sender.id);
    }

    /**
     * @brief Converts a Group object to a formatted std::string representation
     *
     * This function provides a string representation of a Group object in a specific format:
     * "[Group]name(id)", where `name` is the group's display name and `id` is its unique identifier.
     *
     * @param group The Group object to convert to a string
     * @return std::string A formatted string representing the Group object
     */

    inline std::string to_string(const bot_adapter::Group &group) {
        return fmt::format("[Group]{}({})", group.name, group.id);
    }

    enum class GroupPermission { UNKNOWN = 0, MEMBER, ADMINISTRATOR, OWNER };

    inline constexpr std::string to_string(const GroupPermission &permission) {
        switch (permission) {
        case GroupPermission::MEMBER:
            return "MEMBER";
        case GroupPermission::ADMINISTRATOR:
            return "ADMINISTRATOR";
        case GroupPermission::OWNER:
            return "OWNER";
        default:
            return UNKNOWN_VALUE;
        }
    }

    inline constexpr GroupPermission get_group_permission(const std::string_view permission) {
        if (permission == "MEMBER")
            return GroupPermission::MEMBER;
        if (permission == "ADMINISTRATOR")
            return GroupPermission::ADMINISTRATOR;
        if (permission == "OWNER")
            return GroupPermission::OWNER;
        return GroupPermission::UNKNOWN;
    }

    inline std::string get_permission_chs(const GroupPermission perm) {
        switch (perm) {
        case GroupPermission::OWNER:
            return "群主";
        case GroupPermission::ADMINISTRATOR:
            return "管理员";
        default:
            return "普通群友";
        }
    }

    struct GroupInfo {
        qq_id_t group_id;
        std::string name;
        GroupPermission bot_in_group_permission;

        GroupInfo(uint64_t group_id, const std::string_view name, GroupPermission bot_in_group_permission)
            : group_id(group_id), name(name), bot_in_group_permission(bot_in_group_permission) {}
    };

    // @deprecated
    struct GroupMemberProfile : public Profile {
        GroupPermission permission;

        GroupMemberProfile() = default;

        GroupMemberProfile(uint64_t id_, const std::string &name_, const std::string &email_, uint32_t age_,
                           uint32_t level_, ProfileSex sex_, GroupPermission permission_)
            : Profile(id_, name_, email_, age_, level_, sex_), permission(permission_) {}
    };

    struct GroupMemberInfo {
      public:
        qq_id_t id;
        qq_id_t group_id;
        std::string member_name;
        std::optional<std::string> special_title;
        GroupPermission permission;
        std::optional<std::chrono::system_clock::time_point> join_time;
        std::optional<std::chrono::system_clock::time_point> last_speak_time;
        float mute_time_remaining;

        GroupMemberInfo(qq_id_t id, qq_id_t group_id, std::string member_name, std::optional<std::string> special_title,
                        GroupPermission permission, std::optional<std::chrono::system_clock::time_point> join_time,
                        std::optional<std::chrono::system_clock::time_point> last_speak_time, float mute_time_remaining)
            : id(id), group_id(group_id), member_name(std::move(member_name)), special_title(std::move(special_title)),
              permission(permission), join_time(join_time), last_speak_time(last_speak_time),
              mute_time_remaining(mute_time_remaining) {}
    };

    /**
     * @brief Represents a group announcement.
     *
     * This structure holds all the relevant information about a single announcement
     * within a group, such as its content, author, and timing details.
     */
    struct GroupAnnouncement {
        Group group;
        std::string content;
        qq_id_t sender_id;
        std::string fid;
        bool all_confirmed;
        int confirmed_members_count;
        std::chrono::system_clock::time_point publication_time;

        GroupAnnouncement(Group group, std::string content, qq_id_t sender_id, std::string fid, bool all_confirmed,
                          int confirmed_members_count, std::chrono::system_clock::time_point publication_time)
            : group(std::move(group)), content(std::move(content)), sender_id(sender_id), fid(std::move(fid)),
              all_confirmed(all_confirmed), confirmed_members_count(confirmed_members_count),
              publication_time(publication_time) {}

        /**
         * @brief Constructs a GroupAnnouncement from a JSON object.
         * @param anno_json The JSON object representing an announcement.
         */
        GroupAnnouncement(const nlohmann::json &anno_json)
            : group(anno_json["group"]), content(get_optional<std::string>(anno_json, "content").value_or("")),
              sender_id(get_optional<qq_id_t>(anno_json, "senderId").value_or(0)),
              fid(get_optional<std::string>(anno_json, "fid").value_or("")),
              all_confirmed(get_optional<bool>(anno_json, "allConfirmed").value_or(false)),
              confirmed_members_count(get_optional<int>(anno_json, "confirmedMembersCount").value_or(0)),
              publication_time(std::chrono::system_clock::from_time_t(
                  get_optional<time_t>(anno_json, "publicationTime").value_or(0))) {}
    };

    struct GroupWrapper {
        GroupInfo group_info;
        std::unique_ptr<wheel::concurrent_unordered_map<qq_id_t, GroupMemberInfo>> member_info_list;

        // Save group member name embedding vector and group id
        std::unique_ptr<wheel::concurrent_vector<std::pair<neural_network::emb_vec_t, qq_id_t>>>
            member_name_emb_vec_list;

        GroupWrapper(GroupInfo group_info) : group_info(std::move(group_info)) {
            member_info_list = std::make_unique<wheel::concurrent_unordered_map<qq_id_t, GroupMemberInfo>>();
            member_name_emb_vec_list =
                std::make_unique<wheel::concurrent_vector<std::pair<neural_network::emb_vec_t, qq_id_t>>>();
        }
    };
} // namespace bot_adapter

#endif