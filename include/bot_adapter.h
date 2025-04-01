#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include "adapter_message.h"
#include <easywsclient.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include "adapter_event.h"

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
        std::vector<std::function<void(const MessageEvent &msg)>> msg_handle_func_list;

        easywsclient::WebSocket::pointer ws;
    };

} // namespace bot_adapter
#endif