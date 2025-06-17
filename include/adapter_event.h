#ifndef ADAPTER_EVENT_H
#define ADAPTER_EVENT_H

#include "adapter_message.h"
#include "adapter_model.h"
#include "constant_types.hpp"
#include "nlohmann/json_fwd.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

namespace bot_adapter {
    struct Event {
        virtual ~Event() = default;
        virtual std::string_view get_typename() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };

    struct MessageEvent : public Event {
        MessageEvent(message_id_t message_id, std::shared_ptr<Sender> sender_ptr, MessageChainPtrList message_chain,
                     std::chrono::system_clock::time_point send_time)
            : sender_ptr(sender_ptr), message_chain(std::move(message_chain)), send_time(std::move(send_time)) {}

        std::string_view get_typename() const override = 0;

        nlohmann::json to_json() const override {
            nlohmann::json message_chain_json = nlohmann::json::array();

            for (const auto &msg : message_chain) {
                if (msg) {
                    message_chain_json.push_back(msg->to_json());
                }
            }
            nlohmann::json ret_json = {{"type", get_typename()},
                                       {"id", message_id},
                                       {"messageChain", std::move(message_chain_json)},
                                       {"send_time", send_time.time_since_epoch().count()}};

            if (sender_ptr != nullptr) {
                ret_json["sender"] = sender_ptr->to_json();
            }

            return ret_json;
        }

        std::shared_ptr<Sender> sender_ptr;
        MessageChainPtrList message_chain;
        message_id_t message_id;
        std::chrono::system_clock::time_point send_time;
    };

    struct FriendMessageEvent final : public MessageEvent {
        FriendMessageEvent(message_id_t message_id, std::shared_ptr<Sender> sender_ptr,
                           MessageChainPtrList message_chain, std::chrono::system_clock::time_point send_time)
            : MessageEvent(message_id, sender_ptr, std::move(message_chain), std::move(send_time)) {}

        std::string_view get_typename() const override { return "FriendMessageEvent"; }
    };

    struct GroupMessageEvent final : public MessageEvent {
        GroupMessageEvent(message_id_t message_id, std::shared_ptr<GroupSender> sender_ptr,
                          MessageChainPtrList message_chain, std::chrono::system_clock::time_point send_time)
            : MessageEvent(message_id, sender_ptr, std::move(message_chain), std::move(send_time)) {}

        std::string_view get_typename() const override { return "GroupMessageEvent"; }

        const GroupSender &get_group_sender() const {
            auto *gs = dynamic_cast<const GroupSender *>(sender_ptr.get());
            if (!gs) {
                throw std::bad_cast();
            }
            return *gs;
        }
    };

} // namespace bot_adapter

// Specialize nlohmann::adl_serializer for MessageEvent
namespace nlohmann {
    template <> struct adl_serializer<bot_adapter::Event> {
        static void to_json(json &j, const bot_adapter::Event &message_event) { j = message_event.to_json(); }
    };
} // namespace nlohmann

#endif