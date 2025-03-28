#ifndef ADAPTER_EVENT_H
#define ADAPTER_EVENT_H

#include "adapter_model.h"
#include "nlohmann/json_fwd.hpp"
#include <string_view>

namespace bot_adapter {
    struct MessageEvent {
        virtual std::string_view get_typename() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };


    struct GroupMessageEvent : public MessageEvent {
        Sender sender;
        Group group;
    };
}

#endif