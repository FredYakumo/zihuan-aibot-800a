#ifndef ADAPTER_MSG_H
#define ADAPTER_MSG_H

#include "nlohmann/json.hpp"
#include <string_view>

namespace bot_adapter {
    struct MessageBase {
        virtual std::string_view get_type() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };
} // namespace bot_adapter

// Specialize nlohmann::adl_serializer for MessageBase
namespace nlohmann {
    template <> struct adl_serializer<bot_adapter::MessageBase> {
        static void to_json(json &j, const bot_adapter::MessageBase &message) { 
            j = message.to_json();
        }
    };
} // namespace nlohmann

#endif