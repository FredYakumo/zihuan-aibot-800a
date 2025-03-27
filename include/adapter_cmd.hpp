#ifndef ADAPTER_CMD_H
#define ADAPTER_CMD_H

#include "adapter_message.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <string>

namespace bot_adapter {
    struct AdapterCommandContentBase {
        ~AdapterCommandContentBase() = default;
        virtual nlohmann::json to_json() const = 0;
    };

    struct SendGroupMsgContent : public AdapterCommandContentBase {
        std::string target;
        std::vector<MessageBase> message_chain;
        inline nlohmann::json to_json() const override {
            return nlohmann::json{
            {"target", target}, 
            {"messageChain", message_chain}
            };
        }
    };

    struct AdapterCommand {
        uint64_t sync_id;
        std::string command;
        std::shared_ptr<AdapterCommandContentBase> content;
    };

} // namespace bot_adapter

#endif