#ifndef ADAPTER_MSG_H
#define ADAPTER_MSG_H

#include "constant_types.hpp"
#include "constants.hpp"
#include "get_optional.hpp"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "time_utils.h"
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace bot_adapter {
    struct MessageBase {
        virtual std::string_view get_type() const = 0;
        virtual nlohmann::json to_json() const = 0;
        virtual const std::string &display_text() const = 0;
    };

    using MessageChainPtrList = std::vector<std::shared_ptr<MessageBase>>;

    struct MessageStorageEntry {
        uint64_t message_id;
        std::string sender_name;
        uint64_t sender_id;
        std::chrono::system_clock::time_point send_time;
        std::shared_ptr<MessageChainPtrList> message_chain_list;
    };

    struct PlainTextMessage : public MessageBase {
        PlainTextMessage(const std::string_view text) : text(text) {}

        inline std::string_view get_type() const override { return "Plain"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"text", text}}; }

        inline const std::string &display_text() const override { return text; }

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
        QuoteMessage(std::string quoted_text, message_id_t ref_msg_id, std::optional<qq_id_t> ref_group_id_opt,
                     std::optional<qq_id_t> ref_friend_id_opt)
            : text(std::move(quoted_text)), ref_msg_id(ref_msg_id), ref_group_id_opt(ref_group_id_opt),
              ref_friend_id_opt(ref_friend_id_opt) {}

        inline std::string_view get_type() const override { return "Quote"; }

        inline const std::string &display_text() const override { return text; }

        nlohmann::json to_json() const override {
            nlohmann::json json_obj = {{"type", get_type()}, {"text", text}, {"messageId", ref_msg_id}};
            if (ref_group_id_opt) {
                json_obj["groupId"] = *ref_group_id_opt;
            }
            if (ref_friend_id_opt) {
                json_obj["friendId"] = *ref_friend_id_opt;
            }
            return json_obj;
        }

        std::string text;
        message_id_t ref_msg_id;
        std::optional<qq_id_t> ref_group_id_opt;
        std::optional<qq_id_t> ref_friend_id_opt;
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

        DisplayNode(std::string title, std::optional<std::string> summary = std::nullopt)
            : title(std::move(title)), summary(std::move(summary)) {}

        DisplayNode(const nlohmann::json &json) {
            title = get_optional<std::string>(json, "title").value_or(EMPTY_JSON_STR_VALUE);
            summary = get_optional<std::string>(json, "summary");
        }

        nlohmann::json to_json() const {
            nlohmann::json ret_json{{"title", title}};
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

        const std::string &display_text() const override {
            static std::string cached_text;
            cached_text.clear();
            cached_text = "[Forward Message]\n";
            
            for (const auto &node : node_list) {
                cached_text += "From " + node.sender_name + " (" + std::to_string(node.sender_id) + "):\n";
                for (const auto &msg : node.message_chain) {
                    if (msg) {
                        cached_text += msg->display_text();
                        cached_text += "\n";
                    }
                }
                cached_text += "---\n";
            }
            
            if (display && display->summary) {
                cached_text += "Summary: " + *display->summary + "\n";
            }
            
            return cached_text;
        }

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
        ImageMessage(const std::string_view url, std::optional<std::string> describe_text = std::nullopt)
            : url(url), describe_text(std::move(describe_text)) {}

        std::string_view get_type() const override { return "Image"; }

        nlohmann::json to_json() const override {
            nlohmann::json json_obj{{"type", get_type()}, {"url", url}};
            if (describe_text) {
                json_obj["description"] = *describe_text;
            }
            return json_obj;
        }

        inline const std::string &display_text() const override {
            static std::string cached_text;
            if (describe_text) {
                cached_text = "[图片描述: " + *describe_text + " (网址: " + url + ")]";
            } else {
                cached_text = "[图片网址: " + url + "]";
            }
            return cached_text;
        }

        std::string url;
        std::optional<std::string> describe_text;
    };

    struct LocalImageMessage : public MessageBase {
        LocalImageMessage(const std::string_view path, std::optional<std::string> describe_text = std::nullopt)
            : path(path), describe_text(std::move(describe_text)) {}

        std::string_view get_type() const override { return "Image"; }

        nlohmann::json to_json() const override {
            nlohmann::json json_obj{{"type", get_type()}, {"path", path}};
            if (describe_text) {
                json_obj["description"] = *describe_text;
            }
            return json_obj;
        }

        inline const std::string &display_text() const override {
            static std::string cached_text;
            if (describe_text) {
                cached_text = "[图片描述: " + *describe_text + " (路径: " + path + ")]";
            } else {
                cached_text = "[图片路径: " + path + "]";
            }
            return cached_text;
        }

        std::string path;
        std::optional<std::string> describe_text;
    };

    struct AtTargetMessage : public MessageBase {
        AtTargetMessage(uint64_t target) : target(target) {}

        inline std::string_view get_type() const override { return "At"; }

        inline nlohmann::json to_json() const override { return {{"type", get_type()}, {"target", target}}; }

        inline const std::string &display_text() const override {
            static std::string cached_text;
            cached_text = "[提到了 " + std::to_string(target) + "]";
            return cached_text;
        }

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