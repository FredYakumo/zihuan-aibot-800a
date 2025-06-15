#ifndef ADAPTER_MSG_H
#define ADAPTER_MSG_H

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "time_utils.h"
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace bot_adapter {
    struct MessageBase {
        virtual std::string_view get_type() const = 0;
        virtual nlohmann::json to_json() const = 0;
    };

    using MessageChainPtrList = std::vector<std::shared_ptr<MessageBase>>;

    struct PlainTextMessage : public MessageBase {
        PlainTextMessage(const std::string_view text) : text(text) {}

        inline std::string_view get_type() const override { return "Plain"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"text", text}}; }

        std::string text;
    };

    inline std::optional<std::reference_wrapper<const PlainTextMessage>>
    try_plain_text_message(const MessageBase &msg) {
        if (msg.get_type() == "Plain") {
            auto *ptr = dynamic_cast<const PlainTextMessage *>(&msg);
            if (ptr) {
                return std::cref(*ptr);
            }
        }
        return std::nullopt;
    }

    struct QuoteMessage : public MessageBase {
        QuoteMessage(const std::string_view text, uint64_t message_id,
                     std::shared_ptr<MessageBase> origin_message_ptr = nullptr)
            : text(text), message_id(message_id), origin_message_ptr(origin_message_ptr) {}

        inline std::string_view get_type() const override { return "Quote"; }

        inline std::string get_quote_text() const {
            // std::string ret = "";
            // if (origin_message_ptr == nullptr) {
            //     return ret;
            // }
            // if (const auto plain_text = try_plain_text_message(*origin_message_ptr)) {
            //     ret = plain_text->get().text;
            // }
            return text;
        }

        nlohmann::json to_json() const override {
            nlohmann::json json_obj = {{"type", get_type()}, {"text", text}, {"messageId", message_id}};

            if (origin_message_ptr) {
                json_obj["origin"] = origin_message_ptr->to_json();
            }

            return json_obj;
        }

        std::string text;
        uint64_t message_id;
        std::shared_ptr<MessageBase> origin_message_ptr = nullptr;
    };

    struct ForwardMessageNode {
        // Constructor remains the same as before
        ForwardMessageNode(uint64_t sender_id, std::chrono::system_clock::time_point time, std::string sender_name,
                           MessageChainPtrList message_chain, std::optional<uint64_t> message_id = std::nullopt,
                           std::optional<uint64_t> message_ref = std::nullopt)
            : sender_id(sender_id), time(time), sender_name(std::move(sender_name)),
              message_chain(std::move(message_chain)), message_id(message_id), message_ref(message_ref) {}

        nlohmann::json to_json(bool is_format_dt = false) const {

            nlohmann::json node_json = {
                {"senderId", sender_id}, {"senderName", sender_name}, {"messageChain", nlohmann::json::array()}};

            if (is_format_dt) {
                node_json["time"] = system_clock_to_string(time);
            } else {
                auto time_t = std::chrono::system_clock::to_time_t(time);
                node_json["time"] = time_t;
            }

            // Add messageId only if it exists
            if (message_id.has_value()) {
                node_json["messageId"] = *message_id;
            }

            // Add messageRef only if it exists
            if (message_ref.has_value()) {
                node_json["messageRef"] = *message_ref;
            }

            // Serialize each message in the chain
            for (const auto &msg_ptr : message_chain) {
                if (msg_ptr) {
                    node_json["messageChain"].push_back(msg_ptr->to_json());
                }
            }

            return node_json;
        }

        uint64_t sender_id = 0;
        std::chrono::system_clock::time_point time;
        std::string sender_name;
        MessageChainPtrList message_chain;
        std::optional<uint64_t> message_id;
        std::optional<uint64_t> message_ref;
    };

    struct DisplayNode {
        std::string title;
        std::optional<std::string> summary;

        DisplayNode(std::string title, std::optional<std::string> summary = std::nullopt) : title(std::move(title)), summary(std::move(summary)) {}

        DisplayNode(const nlohmann::json &json) {
            title = json.at("title").get<std::string>();
            if (json.contains("summary")) {
                summary = json.at("summary").get<std::string>();
            }
        }

        nlohmann::json to_json() const {
            nlohmann::json ret_json {{"title", title}};
            if (summary) {
                ret_json["summary"] = *summary;
            }
            return ret_json;
        }
    };

    struct ForwardMessage : public MessageBase {
        ForwardMessage(std::vector<ForwardMessageNode> nodes, std::optional<DisplayNode> display = std::nullopt)
            : node_list(std::move(nodes)), display(display) {}

        std::string_view get_type() const override { return "Forward"; }

        nlohmann::json to_json() const override {
            nlohmann::json json_msg = {{"type", get_type()}, {"nodeList", nlohmann::json::array()}};
            if (const auto &d = display) {
                json_msg["display"] = d->to_json();
            }
            for (const auto &node : node_list) {
                json_msg["nodeList"].push_back(node.to_json());
            }

            return json_msg;
        }

        std::vector<ForwardMessageNode> node_list;
        std::optional<DisplayNode> display;
    };

    struct ImageMessage : public MessageBase {
        ImageMessage(const std::string_view url) : url(url) {}

        std::string_view get_type() const override { return "Image"; }

        nlohmann::json to_json() const override { return nlohmann::json{{"type", get_type()}, {"url", url}}; }

        std::string url;
    };

    struct LocalImageMessage : public MessageBase {
        LocalImageMessage(const std::string_view path) : path(path) {}

        std::string_view get_type() const override { return "Image"; }

        nlohmann::json to_json() const override { return nlohmann::json{{"type", get_type()}, {"path", path}}; }

        std::string path;
    };

    struct AtTargetMessage : public MessageBase {
        AtTargetMessage(uint64_t target) : target(target) {}

        inline std::string_view get_type() const override { return "At"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"target", target}}; }

        uint64_t target;
    };

    inline nlohmann::json to_json(const MessageChainPtrList &message_chain) {
        auto ret = nlohmann::json::array();
        for (const auto &item : message_chain) {
            if (item != nullptr) {
                ret.push_back(item->to_json());
            }
        }
        return ret;
    }

    template <typename... Args> MessageChainPtrList make_message_chain_list(Args &&...args) {
        return {std::make_shared<std::decay_t<Args>>(std::forward<Args>(args))...};
    }

    inline std::optional<std::reference_wrapper<const AtTargetMessage>> try_at_target_message(const MessageBase &msg) {
        if (msg.get_type() == "At") {
            auto *ptr = dynamic_cast<const AtTargetMessage *>(&msg);
            if (ptr) {
                return std::cref(*ptr);
            }
        }
        return std::nullopt;
    }

    inline std::optional<std::reference_wrapper<const QuoteMessage>> try_quote_message(const MessageBase &msg) {
        if (msg.get_type() == "Quote") {
            auto *ptr = dynamic_cast<const QuoteMessage *>(&msg);
            if (ptr) {
                return std::cref(*ptr);
            }
        }
        return std::nullopt;
    }

} // namespace bot_adapter

// Specialize nlohmann::adl_serializer for MessageBase
namespace nlohmann {
    template <> struct adl_serializer<bot_adapter::MessageBase> {
        static void to_json(json &j, const bot_adapter::MessageBase &message) { j = message.to_json(); }
    };
} // namespace nlohmann

#endif