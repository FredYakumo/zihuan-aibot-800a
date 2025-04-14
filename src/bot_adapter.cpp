#include "bot_adapter.h"
#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "constants.hpp"
#include "easywsclient.hpp"
#include "get_optional.hpp"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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
                std::string quote_text = "";
                get_optional(msg, "origin")
                    .and_then([&msg](const auto &origin) { return get_optional(origin[0], "text"); })
                    .and_then([&quote_text](const auto &text) {
                        quote_text = text;
                        return std::optional<std::string>{text};
                    });
                spdlog::debug("quote text: {}, json: {}", quote_text, msg.dump());
                const auto id = get_optional<uint64_t>(msg, "id");
                const auto group_id = get_optional<uint64_t>(msg, "group_id");
                if (id && group_id) {
                    // TODO: 获取原始message
                }
                ret.push_back(std::make_shared<QuoteMessage>(quote_text, id.value_or(0)));
            } else if (*type == "Forward") {
                const auto display_option = get_optional(msg, "display");
                std::vector<ForwardMessageNode> node_vec;
                if (const auto &node_list = get_optional<nlohmann::json>(msg, "nodeList")) {
                    for (const nlohmann::json &node : *node_list) {
                        MessageChainPtrList msg_chain_vec;
                        if (const auto &msg_chain_json = get_optional<nlohmann::json>(node, "messageChain")) {
                            for (const auto &msg : *msg_chain_json) {
                                get_optional<std::string>(msg, "text").and_then([&msg_chain_vec](const auto &text) {
                                    msg_chain_vec.push_back(std::make_shared<PlainTextMessage>(text));
                                    return std::optional(text);
                                });
                            }
                        }
                        node_vec.emplace_back(
                            get_optional(node, "senderId").value_or(0),
                            std::chrono::system_clock::from_time_t(get_optional(node, "time").value_or(0)),
                            get_optional(node, "senderName").value_or(EMPTY_JSON_STR_VALUE), std::move(msg_chain_vec),
                            get_optional(node, "messageId"), get_optional(node, "messageRef"));
                    }
                }
                ret.push_back(std::make_shared<ForwardMessage>(node_vec, display_option));
            }
        }

        return ret;
    }

    void BotAdapter::handle_command_result(const std::string &sync_id, const nlohmann::json &data_json) {

        try {
            spdlog::info("Send command success, data json: {}", data_json.dump());
            auto handle_map = command_result_handle_map.write();
            auto iter = handle_map->find(sync_id);
            if (iter != handle_map->cend()) {
                iter->second(data_json);
                handle_map->erase(iter);
            }

        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parsing error: {}, json is: {}", e.what(), data_json.dump());
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}, msg is: {}", e.what(), data_json.dump());
        }
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

            const auto sync_id = get_optional(msg_json, "syncId");
            if (sync_id.has_value()) {
                // std::optional<std::function<void(uint64_t message_id)>> func_option = std::nullopt;
                bool have_sync_id = false;
                {
                    const auto handle_map = command_result_handle_map.read();
                    auto iter = handle_map->find(*sync_id);
                    if (iter != handle_map->cend()) {
                        // func_option = iter->second;
                        have_sync_id = true;
                    }
                }

                // if (const auto func = func_option) {
                //     (*func)(*data);
                // }
                if (have_sync_id) {
                    handle_command_result(*sync_id, *data);
                    return;
                }
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

                auto group_json = get_optional(*sender_json, "group");
                if (!group_json) {
                    spdlog::warn("GroupMessage event中, 收到的数据没有group");
                    return;
                }
                Group group{*group_json};

                std::shared_ptr<GroupSender> sender_ptr = std::make_shared<GroupSender>(*sender_json, *group_json);

                auto msg_chain_json = get_optional(*data, "messageChain");
                if (!msg_chain_json) {
                    spdlog::warn("GroupMessage event中, 收到的数据没有messageChain");
                    return;
                }
                spdlog::debug("parse message chain");
                const auto message_chain = parse_message_chain(*msg_chain_json);
                spdlog::debug("Sender: {}", sender_ptr->to_json().dump());
                auto message_event = GroupMessageEvent(sender_ptr, message_chain);
                spdlog::info("Event json: {}", message_event.to_json().dump());
                spdlog::debug("Call register event functions");
                for (const auto &func : msg_handle_func_list) {
                    func(std::make_shared<bot_adapter::GroupMessageEvent>(message_event));
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

    std::string generate_send_message_sync_id(const Group &group) {
        return fmt::format("send_group_msg_{}_{}", group.id, get_current_time_formatted());
    }

    std::string generate_send_replay_sync_id(const Sender &sender) {
        return fmt::format("sender_replay_target_{}_{}", sender.id, get_current_time_formatted());
    }

    void BotAdapter::send_command(const AdapterCommand &command,
                                  const std::optional<std::function<void(const nlohmann::json &command_res_json)>>
                                      command_res_handle_func_option) {
        ws->send(command.to_json().dump());

        // Add command result handle
        if (auto func = command_res_handle_func_option) {
            auto handle_map = command_result_handle_map.write();
            handle_map->insert(std::make_pair(
                command.sync_id, [func](const nlohmann::json &cmd_res_data_json) { (*func)(cmd_res_data_json); }));
        }
    }

    void BotAdapter::send_message(const Group &group, const MessageChainPtrList &message_chain,
                                  std::optional<std::string_view> sync_id_option,
                                  std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = sync_id_option.value_or(generate_send_message_sync_id(group));
        spdlog::info("Send message to group: {}, sync id: {}", to_string(group), sync_id);
        // const auto message_json = to_json(message_chain);

        send_command(AdapterCommand(sync_id, "sendGroupMessage",
                                    std::make_shared<bot_adapter::SendGroupMsgContent>(group.id, message_chain)));
    }

    void
    BotAdapter::send_replay_msg(const Sender &sender, const MessageChainPtrList &message_chain,
                                std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = generate_send_replay_sync_id(sender);
        spdlog::info("Send replay message to {}({}), sync id: {}", sender.name, sender.id, sync_id);

        if (const auto &group_sender = try_group_sender(sender)) {
            MessageChainPtrList msg_chain_list =
                make_message_chain_list(AtTargetMessage(sender.id), PlainTextMessage(" "));
            msg_chain_list.insert(msg_chain_list.cend(), message_chain.cbegin(), message_chain.cend());

            send_message(group_sender->get().group, msg_chain_list);
        } else {
            // TODO: 实现私聊发送
        }
    }

    void BotAdapter::send_long_plain_text_replay(const Sender &sender, const std::string_view text,
                                                 uint64_t msg_length_limit) {
        const auto sync_id_base = generate_send_replay_sync_id(sender);
        const auto split_output = Utf8Splitter(text, msg_length_limit);
        std::function<void(const std::string_view msg, const std::string_view sync_id)> send_func;
        bool first_msg = true;
        if (const auto group_sender = try_group_sender(sender)) {
            send_func = [this, group_sender, &first_msg](const std::string_view msg, const std::string_view sync_id) {
                send_message(group_sender->get().group,
                             first_msg ? make_message_chain_list(AtTargetMessage(group_sender->get().id),
                                                                 PlainTextMessage(" "), PlainTextMessage(msg))
                                       : make_message_chain_list(PlainTextMessage(msg)),
                             sync_id);
                first_msg = false;
            };
        } else {
            send_func = [this, sender](const std::string_view msg, const std::string_view sync_id) {
                // TODO: 实现私聊发送
            };
        }
        size_t index = 0;
        for (auto chunk : split_output) {
            const auto sync_id = fmt::format("{}_{}", sync_id_base, index);
            spdlog::info("正在输出块: {}, syncId: {}", index, sync_id);
            send_func(chunk, sync_id);

            ++index;
        }
    }

    void BotAdapter::update_bot_profile() {
        send_command(AdapterCommand("get_bot_profile_" + get_current_time_formatted(), "botProfile"),
                     [this](const auto &json) {
                         spdlog::info("Get bot profile successed.");
                         if (const auto id = get_optional(json, "id")) {
                             this->bot_profile.id = *id;
                         }
                         if (const auto name = get_optional(json, "nickname")) {
                             this->bot_profile.name = *name;
                         }
                         get_optional(json, "email").transform([this](const auto email) {
                             return this->bot_profile.email = email;
                         });
                         if (const auto age = get_optional<uint32_t>(json, "age")) {
                             this->bot_profile.age = *age;
                         }
                         if (const auto level = get_optional(json, "level")) {
                             this->bot_profile.level = *level;
                         }
                         if (const auto sex = get_optional<std::string>(json, "sex")) {
                             this->bot_profile.sex = from_string(*sex);
                         }
                     });
    }

} // namespace bot_adapter