#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include <easywsclient.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <functional>

namespace bot_adapter {
    class BotAdapter {
      public:
        BotAdapter(const std::string_view url) {
#ifdef _WIN32
            INT rc;
            WSADATA wsaData;
            rc = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (rc) {
                spdlog::error("WSAStartup Failed.");
                throw rc;
            }
#endif
            ws = easywsclient::WebSocket::from_url(std::string(url));
        }
        ~BotAdapter();
        int start();

      private:
        void handle_message(const std::string &message);

        easywsclient::WebSocket::pointer ws;
    };

} // namespace bot_adapter
#endif