#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "mutex_data.hpp"
#include <easywsclient.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bot_adapter {
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

        template <typename EventFuncT> inline void register_event(std::function<void(std::shared_ptr<EventFuncT> e)> func) {
            static_assert(std::is_base_of<Event, EventFuncT>::value,
                          "EventFuncT must be derived from Event");
            msg_handle_func_list.push_back([func](std::shared_ptr<Event> e) {
                if (auto specific_event = std::dynamic_pointer_cast<EventFuncT>(e)) {
                    func(specific_event);
                }
            });
        }

        void
        send_message(const Group &group, const MessageChainPtrList &message_chain,
                     std::optional<std::string_view> sync_id_option = std::nullopt,
                     std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_replay_msg(
            const Sender &sender, const MessageChainPtrList &message_chain,
            std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);

        void send_long_plain_text_replay(const Sender &sender, const std::string_view text,
                                         uint64_t msg_length_limit = MAX_OUTPUT_LENGTH);

        void update_bot_profile();

        const Profile &get_bot_profile() const { return bot_profile; }

      private:
        void handle_message(const std::string &message);
        std::vector<std::function<void(std::shared_ptr<Event> e)>> msg_handle_func_list;

        MutexData<std::unordered_map<std::string, std::function<void(const nlohmann::json &result_data)>>>
            command_result_handle_map;

        void handle_command_result(const std::string &sync_id, const nlohmann::json &data_json);

        void send_command(const bot_adapter::AdapterCommand &cmd,
                          const std::optional<std::function<void(const nlohmann::json &command_res_json)>>
                              command_res_handle_func_option = std::nullopt);

        Profile bot_profile;

        easywsclient::WebSocket::pointer ws;
    };

} // namespace bot_adapter
#endif