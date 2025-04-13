#ifndef ADAPTER_CMD_H
#define ADAPTER_CMD_H

#include "adapter_message.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace bot_adapter {
    struct AdapterCommandContentBase {
        ~AdapterCommandContentBase() = default;
        virtual nlohmann::json to_json() const = 0;
    };

    struct CommandJsonContent : public AdapterCommandContentBase {
        nlohmann::json content;

        CommandJsonContent(nlohmann::json json) : content(json) {}

        inline nlohmann::json to_json() const override { return content; }
    };

    struct SendGroupMsgContent : public AdapterCommandContentBase {
        uint64_t target;
        std::vector<std::shared_ptr<MessageBase>> message_chain;

        SendGroupMsgContent(uint64_t target, const std::vector<std::shared_ptr<MessageBase>> message_chain)
            : target(target), message_chain(message_chain) {}

        inline nlohmann::json to_json() const override {
            std::vector<nlohmann::json> message_chain_json;
            return nlohmann::json{{"target", target}, {"messageChain", ::bot_adapter::to_json(message_chain)}};
        }
    };

    struct AdapterCommand {
        std::string sync_id;
        std::string command;
        std::shared_ptr<AdapterCommandContentBase> content_option_ptr;

        AdapterCommand(const std::string_view sync_id, const std::string_view command,
                       std::shared_ptr<AdapterCommandContentBase> content_option_ptr = nullptr)
            : sync_id(sync_id), command(command), content_option_ptr(content_option_ptr) {}

        // AdapterCommand(const std::string_view sync_id, const std::string_view command,
        //                const AdapterCommandContentBase &content)
        //     : AdapterCommand(sync_id, command, std::make_shared<std::decay_t<decltype(content)>>(content)) {}

        nlohmann::json to_json() const {
            nlohmann::json js = {{"syncId", sync_id}, {"command", command}};

            if (content_option_ptr != nullptr) {
                js["content"] = content_option_ptr->to_json();
            }

            return js;
        }
    };

    struct AdapterCommandRes {
        uint64_t sync_id;
        std::optional<nlohmann::json> data_json_option;

        AdapterCommandRes(uint64_t sync_id, std::optional<nlohmann::json> data_json_option = std::nullopt)
            : sync_id(sync_id), data_json_option(data_json_option) {}
    };

} // namespace bot_adapter

#endif