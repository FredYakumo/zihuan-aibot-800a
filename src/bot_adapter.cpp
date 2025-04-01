#include "bot_adapter.h"
#include "adapter_event.h"
#include "adapter_model.h"
#include "easywsclient.hpp"
#include "nlohmann/json_fwd.hpp"
#include <chrono>
#include <optional>
#include <string>

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





    void BotAdapter::handle_message(const std::string &message) {
        spdlog::info("On recv message: {}", message);
        try {
            auto msg_json = nlohmann::json::parse(message);
            const auto data = msg_json["data"];
            const std::string type = data["type"];

            if ("GroupMessage" == type) {
                auto message_event = GroupMessageEvent();
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