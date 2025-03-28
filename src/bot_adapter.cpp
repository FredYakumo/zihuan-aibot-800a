#include "bot_adapter.h"
#include "easywsclient.hpp"
#include "nlohmann/json_fwd.hpp"

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
        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error: {}", e.what());
            return;
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}", e.what());
            return;
        }

        

        for (const auto &func : msg_handle_func_list) {
        }
    }

} // namespace bot_adapter