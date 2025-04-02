#include "bot_adapter.h"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "easywsclient.hpp"
#include "nlohmann/json_fwd.hpp"
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
        for  (const auto &msg : msg_chain) {
            const std::optional<std::string> &type = msg["type"];
            const std::optional<std::string> &text = msg["text"];
            spdlog::debug("Message type: {}, text: \"{}\"", msg["type"], msg["text"]);
            if (!type.has_value()) {
                continue;
            }

            if ("Plain" == type) {
                ret.push_back(std::make_shared<PlainTextMessage>(text.value_or(std::string(EMPTY_MSG_TAG))));
            } else if ("At" == type) {
                const std::optional<uint64_t> target = msg["target"];
                if (!target.has_value()) {
                    continue;
                }
                ret.push_back(std::make_shared<AtTargetMessage>(target.value()));
            } else if ("Quo")
        }

        return ret;
    }


    void BotAdapter::handle_message(const std::string &message) {
        spdlog::info("On recv message: {}", message);
        try {
            auto msg_json = nlohmann::json::parse(message);
            const auto data = msg_json["data"];
            const std::string type = data["type"];

            if ("GroupMessage" == type) {
                Sender sender { data["sender"]};
                Group group { data["group"]};
                auto message_event = GroupMessageEvent(sender, group, ))
            }
            for (const auto &func : msg_handle_func_list) {
            }

        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error: {}", e.what());
            return;
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}", e.what());
            return;
        }
    }

} // namespace bot_adapter