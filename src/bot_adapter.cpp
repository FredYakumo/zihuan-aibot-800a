#include "bot_adapter.h"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "easywsclient.hpp"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bot_adapter {

    BotAdapter::~BotAdapter() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    int BotAdapter::start() {

        while (ws->getReadyState() != easywsclient::WebSocket::CLOSED) {
            ws->poll();
            ws->dispatch([this](const std::string &msg) { handle_message(msg); });
        }

        return 0;
    }

    std::vector<std::shared_ptr<MessageBase>> parse_message_chain(const nlohmann::json &msg_chain) {
        std::vector<std::shared_ptr<MessageBase>> ret;
        for (const auto &msg : msg_chain) {

            const auto type = get_optional<std::string>(msg, "type");
            const auto text = get_optional<std::string>(msg, "text");

            spdlog::debug("Message type: {}, text: \"{}\"", type.value_or(std::string{EMPTY_JSON_STR_VALUE}),
                          text.value_or(std::string{EMPTY_JSON_STR_VALUE}));

            if (!type.has_value()) {
                continue;
            }

            if (*type == "Plain") {
                ret.push_back(std::make_shared<PlainTextMessage>(text.value_or(std::string(EMPTY_MSG_TAG))));
            } else if (*type == "At") {
                // 使用 get_optional 获取 target
                const auto target = get_optional<uint64_t>(msg, "target");
                if (!target.has_value()) {
                    continue;
                }
                ret.push_back(std::make_shared<AtTargetMessage>(*target));
            } else if (*type == "Quote") {
                //
            }
        }

        return ret;
    }

    void BotAdapter::handle_message(const std::string &message) {
        spdlog::info("On recv message: {}", message);
        try {
            spdlog::debug("Parse recv json");
            auto msg_json = nlohmann::json::parse(message);
            const auto data = get_optional(msg_json, "data");
            if (data->empty()) {
                return;
            }
            const auto &type = get_optional<std::string>(*data, "type");
            if (type->empty()) {
                return;
            }
            spdlog::debug("Check event type");
            if ("GroupMessage" == type) {

                auto sender_json = get_optional(*data, "sender");
                if (!sender_json) {
                    spdlog::warn("GroupMessage event中, 收到的数据没有sender");
                    return;
                }
                GroupSender sender{*sender_json};

                auto group_json = get_optional(*sender_json, "group");
                if (!group_json) {
                    spdlog::warn("GroupMessage event中, 收到的数据没有group");
                    return;
                }
                Group group{*group_json};

                auto msg_chain_json = get_optional(*data, "messageChain");
                if (!msg_chain_json) {
                    spdlog::warn("GroupMessage event中, 收到的数据没有messageChain");
                    return;
                }
                spdlog::debug("parse message chain");
                const auto message_chain = parse_message_chain(*msg_chain_json);
                auto message_event = GroupMessageEvent(sender, group, message_chain);
                spdlog::debug("Call register event functions");
                for (const auto &func : msg_handle_func_list) {
                    func(message_event);
                }
            }

        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error: {}", e.what());
            return;
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}", e.what());
            return;
        }
    }

    int BotAdapter::send_message(const Group &group, const MessageChainPtrList &message_chain) {
        spdlog::info("Send message to group: {}", to_string(group));
        const auto message_json = to_json(message_chain);
        auto ws_json = nlohmann::json {
            {"syncId", ""},
            {"command", "sendGroupMessage"},
            {"content", {
                {"target", group.id},
                {"messageChain", message_json}
            }}
        };
        ws->send(ws_json.dump());
        return 0;
    }

} // namespace bot_adapter