#ifndef ADAPTER_EVENT_H
#define ADAPTER_EVENT_H

#include "adapter_message.h"
#include "adapter_model.h"
#include "nlohmann/json_fwd.hpp"
#include <algorithm>
#include <string_view>
#include <vector>

namespace bot_adapter {
    struct MessageEvent {
        virtual std::string_view get_typename() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };

    struct GroupMessageEvent : public MessageEvent {
        GroupMessageEvent(Sender sender, Group group, std::vector<MessageBase> messge_chain)
            : sender(std::move(sender)), group(std::move(group)), message_chain(std::move(messge_chain)) {}
        inline virtual std::string_view get_typename() const override { return "GroupMessageEvent"; }
        inline virtual nlohmann::json to_json() const override {
            return nlohmann::json{{"type", get_typename()}

            };
        }
        Sender sender;
        Group group;
        std::vector<MessageBase> message_chain;
    };
} // namespace bot_adapter


// Specialize nlohmann::adl_serializer for MessageEvent
namespace nlohmann {
    template <> struct adl_serializer<bot_adapter::MessageEvent> {
        static void to_json(json &j, const bot_adapter::MessageEvent &message_event) { 
            j = message_event.to_json();
        }
    };
} // namespace nlohmann

#endif