#ifndef BOT_ADAPTER_H
#define BOT_ADAPTER_H

#include <boost/beast.hpp>
#include <string_view>
#include <thread>

namespace bot_adapter {
    namespace beast = boost::beast;
    namespace websocket = beast::websocket;
    namespace net = boost::asio;
    using tcp = boost::asio::ip::tcp;
    class adapter : public std::enable_shared_from_this<adapter> {
    public:
        adapter();
        ~adapter();
        void start(const std::string_view target_url, uint16_t port);
        void on_send(const std::string& message);
    private:
        net::io_context ioc_;
        tcp::resolver resolver_;
        websocket::stream<beast::tcp_stream> ws_;
        beast::flat_buffer buffer_;
        std::thread io_thread_;
        bool is_writing_;
        std::queue<std::string> write_queue_;
        void stop();
        void on_resolve(beast::error_code ec, tcp::resolver::results_type results);
        void on_connect(beast::error_code ec, tcp::resolver::results_type::endpoint_type endpoint);
        void on_handshake(beast::error_code ec);
        void on_read(beast::error_code ec, std::size_t bytes_transferred);
        void queue_message(const std::string& message);
        void do_write();
        void on_write(beast::error_code ec, std::size_t bytes_transferred);
        void process_message(const std::string& msg);
        void handle_error(beast::error_code ec, const char* what);
    };
}

#endif