#include "bot_adapter.h"
#include "easywsclient.hpp"

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

    }

} // namespace bot_adapter