#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "neural_network/text_model.h"
#include <collection/concurrent_unordered_map.hpp>
#include <cstdint>
#include <easywsclient.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bot_adapter {
    using CommandResHandleFunc = std::function<void(const nlohmann::json &command_res_json)>;
    using GroupAnnouncementResHandleFunc = std::function<void(const std::vector<GroupAnnouncement> &)>;
    class BotAdapter {
      public:
        BotAdapter(const std::string_view url, std::optional<uint64_t> bot_id_option = std::nullopt) {
#ifdef _WIN32
            INT rc;
            WSADATA wsaData;
            rc = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (rc) {
                spdlog::error("WSAStartup Failed.");
                throw rc;
            }
#endif
            ws = easywsclient::WebSocket::from_url(std::string(url));
            if (bot_id_option) {
                bot_profile.id = *bot_id_option;
            }
        }
        ~BotAdapter();

        int start();

        template <typename EventFuncT>
        inline void register_event(std::function<void(std::shared_ptr<EventFuncT> e)> func) {
            static_assert(std::is_base_of<Event, EventFuncT>::value, "EventFuncT must be derived from Event");
            msg_handle_func_list.push_back([func](std::shared_ptr<Event> e) {
                if (auto specific_event = std::dynamic_pointer_cast<EventFuncT>(e)) {
                    func(specific_event);
                }
            });
        }

        /**
         * @brief Send a message to a specific friend
         *
         * This is the fundamental function for bot message sending to friends. All other bot friend message sending
         * functions should ultimately call this function to complete the final transmission.
         *
         * @param sender The target friend to send the message to
         * @param message_chain The message chain containing the content to send
         * @param sync_id_option Optional sync ID for tracking the message
         * @param out_message_id_option Optional callback function to receive the sent message ID
         */

        void
        send_message(const Sender &sender, const MessageChainPtrList &message_chain,
                     std::optional<std::string_view> sync_id_option = std::nullopt,
                     std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        /**
         * @brief Send a message to a specific group
         *
         * This is the fundamental function for bot message sending to groups. All other bot group message sending
         * functions should ultimately call this function to complete the final transmission.
         *
         * @param group The target group to send the message to
         * @param message_chain The message chain containing the content to send
         * @param sync_id_option Optional sync ID for tracking the message
         * @param out_message_id_option Optional callback function to receive the sent message ID
         */

        void send_group_message(
            const Group &group, const MessageChainPtrList &message_chain,
            std::optional<std::string_view> sync_id_option = std::nullopt,
            std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_replay_msg(
            const Sender &sender, const MessageChainPtrList &message_chain, bool at_target = true,
            std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_long_plain_text_reply(const Sender &sender, std::string text, bool at_target = true,
                                        uint64_t msg_length_limit = MAX_OUTPUT_LENGTH,
                                        std::optional<std::function<void(uint64_t &)>> out_message_id_option = std::nullopt);

        void update_bot_profile();

        inline void get_message_id(uint64_t message_id, uint64_t target_id, CommandResHandleFunc out_func) {
            const std::string sync_id = fmt::format("get_message_id_{}", message_id);
            send_command(AdapterCommand(sync_id, "messageFromId",
                                        std::make_shared<CommandJsonContent>(
                                            CommandJsonContent({{"messageId", message_id}, {"target", target_id}}))),
                         out_func);
        }

        inline std::optional<std::reference_wrapper<const GroupWrapper>>
        fetch_group_member_info(qq_id_t group_id) const {
            if (auto f = group_info_map.find(group_id); f.has_value()) {
                return std::cref(f->get());
            }
            return std::nullopt;
        }

        void update_group_info_sync();

        const Profile &get_bot_profile() const { return bot_profile; }

        inline const GroupWrapper &get_group(qq_id_t group_id) const { return group_info_map.find(group_id)->get(); }

        /**
         * @brief Get the group announcement object
         *
         * @param group_id
         * @param out_func
         * @param offset
         * @param size
         */
        inline void get_group_announcement(qq_id_t group_id, GroupAnnouncementResHandleFunc out_func, int offset = 0,
                                           int size = 10) {
            const std::string sync_id = fmt::format("get_group_announcement_{}_{}", group_id,
                                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        std::chrono::system_clock::now().time_since_epoch())
                                                        .count());
            send_command(AdapterCommand(sync_id, "anno_list",
                                        std::make_shared<CommandJsonContent>(CommandJsonContent(
                                            {{"id", group_id}, {"offset", offset}, {"size", size}}))),
                         [out_func](const nlohmann::json &command_res_json) {
                             std::vector<GroupAnnouncement> announcements;
                             if (command_res_json.contains("data")) {
                                 for (const auto &item : command_res_json["data"]) {
                                     announcements.emplace_back(item);
                                 }
                             }
                             out_func(announcements);
                         });
        }

        /**
         * @brief A map storing name embeddings for group members
         *
         * This map maintains a mapping between QQ group IDs (qq_id_t) and
         * their corresponding name embeddings (GroupMemberNameEmbeddngMatrix).
         *
         * Key: QQ group ID 群号 (qq_id_t)
         * Value: Matrix containing name embeddings for the member (GroupMemberNameEmbeddngMatrix)
         */
        wheel::concurrent_unordered_map<qq_id_t, GroupMemberNameEmbeddngMatrix> group_member_name_embedding_map;

        /**
         * @brief Get group announcements synchronously.
         *
         * This function blocks until a response is received or a timeout occurs.
         *
         * @param group_id
         * @param offset
         * @param size
         * @param timeout The maximum time to wait for a response.
         * @return A list of group announcements, or std::nullopt if the request fails or times out.
         */
        inline std::optional<std::vector<GroupAnnouncement>>
        get_group_announcement_sync(qq_id_t group_id, int offset = 0, int size = 10,
                                    std::chrono::milliseconds timeout = std::chrono::milliseconds(20000)) {
            const std::string sync_id = fmt::format("get_group_announcement_sync_{}_{}", group_id,
                                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        std::chrono::system_clock::now().time_since_epoch())
                                                        .count());
            auto res_json_option =
                send_command_sync(AdapterCommand(sync_id, "anno_list",
                                                 std::make_shared<CommandJsonContent>(CommandJsonContent(
                                                     {{"id", group_id}, {"offset", offset}, {"size", size}}))),
                                  timeout);

            if (!res_json_option.has_value()) {
                return std::nullopt;
            }

            std::vector<GroupAnnouncement> announcements;
            if (res_json_option->contains("data")) {
                for (const auto &item : (*res_json_option)["data"]) {
                    announcements.emplace_back(item);
                }
            }
            return announcements;
        }

      private:
        void handle_message(const std::string &message);
        std::vector<std::function<void(std::shared_ptr<Event> e)>> msg_handle_func_list;

        wheel::concurrent_unordered_map<std::string, CommandResHandleFunc> command_result_handle_map;

        std::optional<Profile> fetch_group_member_profile_sync(qq_id_t group_id, qq_id_t id);

        wheel::concurrent_unordered_map<qq_id_t, GroupWrapper> group_info_map;

        /**
         * @brief
         *
         * @param sync_id
         * @param data_json
         * @return true A command handle function match a called.
         * @return false not any match handle function found.
         */
        bool handle_command_result(const std::string &sync_id, const nlohmann::json &data_json);

        void send_command(const bot_adapter::AdapterCommand &cmd,
                          const std::optional<CommandResHandleFunc> command_res_handle_func_option = std::nullopt);

        std::optional<nlohmann::json>
        send_command_sync(const bot_adapter::AdapterCommand &cmd,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(20000));

        std::queue<std::string> send_cmd_queue;

        Profile bot_profile;

        std::vector<GroupInfo> fetch_bot_group_list_info_sync();

        std::vector<GroupMemberInfo> fetch_group_member_list_sync(const GroupInfo &group_info);

        bool is_running = false;

        easywsclient::WebSocket::pointer ws;
    };

} // namespace bot_adapter
#endif