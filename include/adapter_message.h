#ifndef ADAPTER_MSG_H
#define ADAPTER_MSG_H

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include <string_view>

namespace bot_adapter {
    struct MessageBase {
        virtual std::string_view get_type() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };

    struct PlainTextMessage : public MessageBase {
        PlainTextMessage(const std::string_view text) : text(text) {}

        inline std::string_view get_type() const override { return "Plain"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"text", text}}; }

        std::string text;
    };

    struct AtTargetMessage : public MessageBase {
        AtTargetMessage(uint64_t target) : target(target) {}

        inline std::string_view get_type() const override { return "At"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"target", target}}; }

        uint64_t target;
    };

} // namespace bot_adapter

// Specialize nlohmann::adl_serializer for MessageBase
namespace nlohmann {
    template <> struct adl_serializer<bot_adapter::MessageBase> {
        static void to_json(json &j, const bot_adapter::MessageBase &message) { j = message.to_json(); }
    };
} // namespace nlohmann

#endif