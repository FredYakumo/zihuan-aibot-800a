#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "mutex_data.hpp"
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

        void
        send_message(const Sender &sender, const MessageChainPtrList &message_chain,
                     std::optional<std::string_view> sync_id_option = std::nullopt,
                     std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        // /**
        // Send message to friend (async version)
        // with timeout retry
        //  */
        // void
        // send_message_async(const Sender &sender, const MessageChainPtrList &message_chain,
        //     size_t max_retry_count = 3,
        //     std::chrono::milliseconds timeout = std::chrono::milliseconds(10000));
        

        void send_group_message(
            const Group &group, const MessageChainPtrList &message_chain,
            std::optional<std::string_view> sync_id_option = std::nullopt,
            std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_replay_msg(
            const Sender &sender, const MessageChainPtrList &message_chain, bool at_target = true,
            std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_long_plain_text_replay(const Sender &sender, const std::string_view text, bool at_target = true,
                                         uint64_t msg_length_limit = MAX_OUTPUT_LENGTH);

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
            const auto &map = group_info_map.read();
            if (auto iter = map->find(group_id); iter != map->cend()) {
                return std::cref(iter->second);
            }
            return std::nullopt;
        }

        void update_group_info_sync();

        const Profile &get_bot_profile() const { return bot_profile; }

      private:
        void handle_message(const std::string &message);
        std::vector<std::function<void(std::shared_ptr<Event> e)>> msg_handle_func_list;

        MutexData<std::unordered_map<std::string, CommandResHandleFunc>> command_result_handle_map;

        std::optional<Profile> fetch_group_member_profile_sync(qq_id_t group_id, qq_id_t id);

        MutexData<std::unordered_map<qq_id_t, GroupWrapper>> group_info_map;

        void handle_command_result(const std::string &sync_id, const nlohmann::json &data_json);

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