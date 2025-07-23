#include "bot_adapter.h"
#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "config.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "database.h"
#include "easywsclient.hpp"
#include "get_optional.hpp"
#include "global_data.h"
#include "neural_network/model_set.h"
#include "neural_network/text_model.h"
#include "nlohmann/json_fwd.hpp"
#include "time_utils.h"
#include "utils.h"
#include <base64.hpp>
#include <chrono>
#include <cpr/cpr.h>
#include <cstdint>
#include <future>
#include <general-wheel-cpp/markdown_utils.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bot_adapter {

    using namespace wheel;

    void fetch_message_list_from_db(bot_adapter::BotAdapter &adapter) {
        spdlog::info("从持久化数据中加载并初始化bot的记忆数据");

        spdlog::info("从Database中获取1000条消息记录");
        spdlog::info("获取Bot的所有群信息");
        const auto group_list = adapter.get_bot_all_group_info();

        for (auto group : group_list) {
            spdlog::info("Fetch message list in group '{}'({})", group.name, group.group_id);
            auto message_list =
                database::get_global_db_connection().query_group_message(group.group_id, std::nullopt, 1000);
            spdlog::info("Actual fetch count: {}", message_list.size());

            spdlog::info("Prepare batch add message storage data");
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
            std::vector<qq_id_t> this_group_id_vec(message_list.size(), group.group_id);
            std::vector<qq_id_t> this_message_id_vec;
            this_message_id_vec.reserve(message_list.size());
            message_id_t padding_message_id = 0;
            std::vector<std::shared_ptr<MessageStorageEntry>> this_msg_entry_ptr_vec;
            this_msg_entry_ptr_vec.reserve(message_list.size());

            std::set<qq_id_t> member_id_set;
            std::vector<std::string> member_name_vec;

            for (const auto &msg : message_list) {
                message_id_t msg_id = msg.message_id_opt.value_or(padding_message_id++);
                this_message_id_vec.push_back(msg_id);
                this_msg_entry_ptr_vec.push_back(std::make_shared<MessageStorageEntry>(
                    msg_id, msg.sender.name, msg.sender.id, msg.send_time,
                    std::make_shared<MessageChainPtrList>(make_message_chain_list(PlainTextMessage(msg.content)))));
                if (member_id_set.find(msg.sender.id) == member_id_set.end()) {
                    member_id_set.insert(msg.sender.id);
                    member_name_vec.push_back(msg.sender.name);
                }
            }
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Prepare batch add message storage data cost: {}ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
            spdlog::info("Start batch add database message records to group message storage");
            start_time = std::chrono::high_resolution_clock::now();
            g_group_message_storage.batch_add_message(this_group_id_vec, this_message_id_vec, this_msg_entry_ptr_vec);
            end_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Batch add database message records cost: {}ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
            auto embedding_map = adapter.group_member_name_embedding_map.get_or_emplace_value(
                group.group_id, bot_adapter::GroupMemberNameEmbeddngMatrix{});
            std::vector<qq_id_t> member_id_vec{std::make_move_iterator(member_id_set.begin()),
                                               std::make_move_iterator(member_id_set.end())};
            member_id_set.clear();
            // TODO: need optim performance
            spdlog::info("Calculate group member name embedding matrix");
            start_time = std::chrono::high_resolution_clock::now();
            embedding_map->batch_add_member(member_id_vec, member_name_vec);
            end_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Calculate group member name embedding matrix cost: {}ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
        }

        spdlog::info("获取Bot的所有好友信息");
        const auto friend_list = adapter.get_friend_list_sync();
        for (const auto &friend_info : friend_list) {
            spdlog::info("Fetch message list from '{}'({})", friend_info.name, friend_info.id);

            auto message_list = database::get_global_db_connection().query_user_message(friend_info.id, 1000);
            spdlog::info("Actual fetch count: {}", message_list.size());

            spdlog::info("Prepare batch add message storage data");
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
            std::vector<qq_id_t> this_friend_id_vec(message_list.size(), friend_info.id);
            std::vector<message_id_t> this_message_id_vec;
            this_message_id_vec.reserve(message_list.size());
            message_id_t padding_message_id = 0;
            std::vector<std::shared_ptr<MessageStorageEntry>> this_msg_entry_ptr_vec;
            this_msg_entry_ptr_vec.reserve(message_list.size());

            for (const auto &msg : message_list) {
                message_id_t msg_id = msg.message_id_opt.value_or(padding_message_id++);
                this_message_id_vec.push_back(msg_id);
                this_msg_entry_ptr_vec.push_back(std::make_shared<MessageStorageEntry>(
                    msg_id, msg.sender.name, msg.sender.id, msg.send_time,
                    std::make_shared<MessageChainPtrList>(make_message_chain_list(PlainTextMessage(msg.content)))));
            }
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Prepare batch add message storage data cost: {}ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

            spdlog::info("Start batch add database message records to friend message storage");
            start_time = std::chrono::high_resolution_clock::now();
            g_person_message_storage.batch_add_message(this_friend_id_vec, this_message_id_vec, this_msg_entry_ptr_vec);
            end_time = std::chrono::high_resolution_clock::now();
            spdlog::info("Batch add database message records cost: {}ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
        }
    }

    BotAdapter::~BotAdapter() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    int BotAdapter::start() {
        is_running = true;

        std::thread init_thread([this]() {
            const auto &config = Config::instance();

            update_group_info_sync();
            spdlog::info("从Database中获取持久化的消息记录并初始化到内存中");
            fetch_message_list_from_db(*this);
            spdlog::info("完成从Database中获取消息记录");

            spdlog::info("等待{}秒后再次运行update_group_info()", config.update_group_info_period_sec);
            std::this_thread::sleep_for(std::chrono::seconds(config.update_group_info_period_sec));
            while (is_running) {
                spdlog::info("周期性运行update_group_info()");
                update_group_info_sync();
                spdlog::info("等待{}秒后再次运行update_group_info()", config.update_group_info_period_sec);
                std::this_thread::sleep_for(std::chrono::seconds(config.update_group_info_period_sec));
            }
        });

        while (ws->getReadyState() != easywsclient::WebSocket::CLOSED) {
            ws->poll();
            ws->dispatch([this](const std::string &msg) { handle_message(msg); });
            if (!send_cmd_queue.empty()) {
                ws->send(send_cmd_queue.front());
                send_cmd_queue.pop();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        is_running = false;
        init_thread.join();

        // TODO: Handle reconnection logic

        return 0;
    }

    struct ParseMessageChainResult {
        MessageChainPtrList message_chain;
        message_id_t message_id;
        std::chrono::system_clock::time_point send_time;
    };

    ParseMessageChainResult parse_message_chain(const nlohmann::json &msg_chain, std::optional<qq_id_t> group_id_opt,
                                                std::optional<qq_id_t> friend_id_opt) {
        std::vector<std::shared_ptr<MessageBase>> ret;
        std::optional<message_id_t> message_id;
        std::optional<std::chrono::system_clock::time_point> send_time;
        for (const auto &msg : msg_chain) {

            const auto type = get_optional<std::string>(msg, "type");
            const auto text = get_optional<std::string>(msg, "text");

            spdlog::debug("Message type: {}, text: \"{}\"", type.value_or(std::string{EMPTY_JSON_STR_VALUE}),
                          text.value_or(std::string{EMPTY_JSON_STR_VALUE}));

            if (!type.has_value()) {
                continue;
            }

            // Parse message metadata
            if (*type == "Source") {
                message_id = get_optional(msg, "id");
                send_time = get_optional<uint64_t>(msg, "time").transform([](uint64_t time) {
                    return std::chrono::system_clock::from_time_t(time);
                });
                spdlog::info("Source message id: {}, time: {}", message_id.value_or(-1),
                             send_time.value_or(std::chrono::system_clock::now()));
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
                ret.push_back(std::make_shared<QuoteMessage>(quote_text, id.value_or(0), group_id_opt, friend_id_opt));
            } else if (*type == "Forward") {
                const nlohmann::json display_option = get_optional(msg, "display");
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
            } else if (*type == "Image") {
                const auto url = get_optional<std::string>(msg, "url");
                if (url.has_value()) {
                    ret.push_back(std::make_shared<ImageMessage>(*url));
                } else {
                    ret.push_back(std::make_shared<PlainTextMessage>("[图片加载失败]"));
                }
            }
        }

        return {std::move(ret), message_id.value_or(0), send_time.value_or(std::chrono::system_clock::now())};
    }

    bool BotAdapter::handle_command_result(const std::string &sync_id, const nlohmann::json &data_json) {

        try {
            if (const auto &handle_func = command_result_handle_map.pop(sync_id); handle_func.has_value()) {
                spdlog::debug("Send command success, data json: {}", data_json.dump());
                handle_func.value()(data_json);
                return true;
            }

        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parsing error: {}, json is: {}", e.what(), data_json.dump());
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}, msg is: {}", e.what(), data_json.dump());
        }
        return false;
    }

    void handle_message_event(const std::string &type, const nlohmann::json &data,
                              const std::vector<std::function<void(std::shared_ptr<Event> e)>> &msg_handle_func_list) {
        auto sender_json = get_optional(data, "sender");
        if (!sender_json) {
            spdlog::warn("{} event中, 收到的数据没有sender", type);
            return;
        }

        auto msg_chain_json = get_optional(data, "messageChain");
        if (!msg_chain_json) {
            spdlog::warn("{} event中, 收到的数据没有messageChain", type);
            return;
        }

        spdlog::debug("parse message chain");

        auto process_event = [&](auto sender_ptr, auto create_event, ParseMessageChainResult parse_result) {
            spdlog::debug("Sender: {}", sender_ptr->to_json().dump());
            auto message_event = create_event(sender_ptr, parse_result);
            spdlog::info("Event json: {}", message_event.to_json().dump());
            spdlog::debug("Call register event functions");
            for (const auto &func : msg_handle_func_list) {
                func(std::make_shared<std::decay_t<decltype(message_event)>>(message_event));
            }
        };

        if (type == "GroupMessage") {
            auto group_json = get_optional(*sender_json, "group");
            if (!group_json) {
                spdlog::warn("GroupMessage event中, 收到的数据没有group");
                return;
            }
            auto group_sender_ptr = std::make_shared<GroupSender>(*sender_json, *group_json);
            auto parse_result = parse_message_chain(*msg_chain_json, group_sender_ptr->group.id, std::nullopt);
            process_event(
                group_sender_ptr,
                [](auto sender, ParseMessageChainResult parse_result) {
                    return GroupMessageEvent(parse_result.message_id, sender,
                                             std::make_shared<MessageChainPtrList>(parse_result.message_chain),
                                             parse_result.send_time);
                },
                parse_result);
        } else if (type == "FriendMessage") {
            auto sender_ptr = std::make_shared<Sender>(*sender_json);
            auto parse_result = parse_message_chain(*msg_chain_json, std::nullopt, sender_ptr->id);

            process_event(
                sender_ptr,
                [](auto sender, ParseMessageChainResult parse_result) {
                    return FriendMessageEvent(parse_result.message_id, sender,
                                              std::make_shared<MessageChainPtrList>(parse_result.message_chain),
                                              parse_result.send_time);
                },
                parse_result);
        }
    }

    void BotAdapter::handle_message(const std::string &message) {
        spdlog::debug("On recv message: {}", message);
        try {
            spdlog::debug("Parse recv json");
            auto msg_json = nlohmann::json::parse(message);

            const auto data = get_optional(msg_json, "data");
            if (data->empty()) {
                return;
            }

            const auto sync_id = get_optional<std::string>(msg_json, "syncId");
            if (sync_id.has_value() && !sync_id->empty()) {
                if (handle_command_result(*sync_id, *data)) {
                    return;
                }
            }

            const auto &type = get_optional<std::string>(*data, "type");
            if (type->empty()) {
                return;
            }

            spdlog::debug("Check event type");
            if (*type == "GroupMessage" || *type == "FriendMessage") {
                handle_message_event(*type, *data, msg_handle_func_list);
            }

        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("JSON parse error: {}", e.what());
            return;
        } catch (const std::exception &e) {
            spdlog::error("Unexpected error: {}", e.what());
            return;
        }
    }

    std::string generate_send_message_sync_id(const Sender &sender) {
        return fmt::format("send_msg_{}_{}", sender.id, get_current_time_formatted());
    }

    std::string generate_send_group_message_sync_id(const Group &group) {
        return fmt::format("send_group_msg_{}_{}", group.id, get_current_time_formatted());
    }

    std::string generate_send_replay_sync_id(const Sender &sender) {
        return fmt::format("sender_replay_target_{}_{}", sender.id, get_current_time_formatted());
    }

    void BotAdapter::send_command(const AdapterCommand &command,
                                  const std::optional<std::function<void(const nlohmann::json &command_res_json)>>
                                      command_res_handle_func_option) {
        // ws->send(cmd_str);
        send_cmd_queue.push(command.to_json().dump());

        // Add command result handle
        if (auto func = command_res_handle_func_option) {
            command_result_handle_map.insert_or_assign(
                command.sync_id, [func](const nlohmann::json &cmd_res_data_json) { (*func)(cmd_res_data_json); });
        }
    }

    std::optional<nlohmann::json> BotAdapter::send_command_sync(const AdapterCommand &command,
                                                                std::chrono::milliseconds timeout) {
        std::promise<nlohmann::json> promise;
        auto future = promise.get_future();

        send_command(command, [&promise](const nlohmann::json &res) { promise.set_value(res); });

        if (future.wait_for(timeout) == std::future_status::timeout) {
            spdlog::error("Send command(sync) '{}'(sync id: {}) timeout. payload: {}", command.command, command.sync_id,
                          command.to_json().dump());
            throw std::runtime_error("Command execution timeout");
        }

        return future.get();
    }

    void BotAdapter::send_message(const Sender &sender, const MessageChainPtrList &message_chain,
                                  std::optional<std::string_view> sync_id_option,
                                  std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = std::string(sync_id_option.value_or(generate_send_message_sync_id(sender)));
        spdlog::info("Send message to sender: {}, sync id: {}", to_string(sender), sync_id);
        // const auto message_json = to_json(message_chain);

        send_command(
            AdapterCommand(sync_id, "sendFriendMessage",
                           std::make_shared<bot_adapter::SendMsgContent>(sender.id, message_chain)),
            [this, sender, message_chain = message_chain, out_message_id_option](const nlohmann::json &res) mutable {
                const auto message_id_opt = get_optional<uint64_t>(res, "messageId");

                if (!message_id_opt.has_value()) {
                    spdlog::warn("sendFriendMessage response does not contain messageId. Response: {}", res.dump());
                    return;
                }
                auto message_id = *message_id_opt;

                MessageStorageEntry entry{.message_id = message_id,
                                          .sender_name = bot_profile.name,
                                          .sender_id = bot_profile.id,
                                          .send_time = std::chrono::system_clock::now(),
                                          .message_chain_list =
                                              std::make_shared<MessageChainPtrList>(std::move(message_chain))};
                g_person_message_storage.add_message(sender.id, message_id, entry);
                g_bot_send_group_message_storage.add_message(sender.id, message_id, std::move(entry));

                if (out_message_id_option) {
                    (*out_message_id_option)(message_id);
                }
            });
    }

    void
    BotAdapter::send_group_message(const Group &group, const MessageChainPtrList &message_chain,
                                   std::optional<std::string_view> sync_id_option,
                                   std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = std::string(sync_id_option.value_or(generate_send_group_message_sync_id(group)));
        spdlog::info("Send message to group: {}, sync id: {}", to_string(group), sync_id);
        // const auto message_json = to_json(message_chain);

        send_command(
            AdapterCommand(sync_id, "sendGroupMessage",
                           std::make_shared<bot_adapter::SendMsgContent>(group.id, message_chain)),
            [this, group, message_chain = message_chain, out_message_id_option](const nlohmann::json &res) mutable {
                const auto message_id_opt = get_optional<uint64_t>(res, "messageId");

                if (!message_id_opt.has_value()) {
                    spdlog::warn("sendGroupMessage response does not contain messageId. Response: {}", res.dump());
                    return;
                }
                auto message_id = *message_id_opt;

                MessageStorageEntry entry{.message_id = message_id,
                                          .sender_name = bot_profile.name,
                                          .sender_id = bot_profile.id,
                                          .send_time = std::chrono::system_clock::now(),
                                          .message_chain_list =
                                              std::make_shared<MessageChainPtrList>(std::move(message_chain))};
                g_group_message_storage.add_message(group.id, message_id, entry);
                g_bot_send_group_message_storage.add_message(group.id, message_id, std::move(entry));
                if (out_message_id_option) {
                    (*out_message_id_option)(message_id);
                }
            });
    }

    void
    BotAdapter::send_replay_msg(const Sender &sender, const MessageChainPtrList &message_chain, bool at_target,
                                std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = generate_send_replay_sync_id(sender);
        spdlog::info("Send replay message to {}({}), sync id: {}", sender.name, sender.id, sync_id);

        if (const auto &group_sender = try_group_sender(sender)) {
            MessageChainPtrList msg_chain_list;
            if (at_target) {
                msg_chain_list = make_message_chain_list(AtTargetMessage(sender.id), PlainTextMessage(" "));
                msg_chain_list.insert(msg_chain_list.cend(), message_chain.cbegin(), message_chain.cend());
            } else {
                msg_chain_list = message_chain;
            }

            send_group_message(group_sender->get().group, msg_chain_list);
        } else {
            send_message(sender, message_chain);
        }
    }

    /**
     * @brief 对markdown节点进行合并，连续同类型合并，单块不超过max_lines。
     * @param nodes 原始markdown节点
     * @param max_lines 单块最大行数
     * @param font_size rich_text时用于word-break的字体大小
     * @param body_width rich_text时用于word-break的最大宽度
     * @return 合并后的节点
     */
    static std::vector<wheel::MarkdownNode> merge_markdown_nodes(const std::vector<wheel::MarkdownNode> &nodes,
                                                                 size_t max_lines = 500, float font_size = 15,
                                                                 int body_width = 350) {
        using wheel::MarkdownNode;
        std::vector<MarkdownNode> merged;
        auto count_lines = [](const std::string &text) -> size_t {
            return std::count(text.begin(), text.end(), '\n') + 1;
        };
        auto word_break_for_rich_text = [font_size, body_width](const std::string &text) -> std::string {
            size_t max_chars_per_line = static_cast<size_t>(body_width / font_size);
            std::string result;
            size_t char_count = 0;
            for (size_t i = 0; i < text.size(); ++i) {
                char c = text[i];
                if (c == '\n') {
                    result += c;
                    char_count = 0;
                } else {
                    result += c;
                    ++char_count;
                    if (char_count >= max_chars_per_line) {
                        result += '\n';
                        char_count = 0;
                    }
                }
            }
            return result;
        };
        auto try_merge = [&](MarkdownNode &current, const MarkdownNode &next) -> bool {
            // rich_text
            if (current.rich_text && next.rich_text)
                return true;
            if (current.code_text && next.code_text && current.code_language == next.code_language)
                return true;
            if (current.table_text && next.table_text)
                return true;
            if (current.latex_text && next.latex_text)
                return true;
            return false;
        };
        MarkdownNode current;
        size_t current_lines = 0;
        bool has_current = false;
        for (const auto &node : nodes) {
            std::string node_text = node.text;
            // rich_text word-break
            if (node.rich_text) {
                node_text = word_break_for_rich_text(node_text);
            }
            size_t node_lines = count_lines(node_text);
            if (!has_current) {
                current = node;
                current.text = node_text;
                current_lines = node_lines;
                has_current = true;
                continue;
            }
            if (try_merge(current, node) && current_lines + node_lines <= max_lines) {
                // 合并
                if (current.rich_text && node.rich_text) {
                    if (current.rich_text && node.rich_text) {
                        *current.rich_text += "\n" + *node.rich_text;
                    }
                }
                if (current.code_text && node.code_text) {
                    if (current.code_text && node.code_text) {
                        *current.code_text += "\n" + *node.code_text;
                    }
                }
                if (current.table_text && node.table_text) {
                    if (current.table_text && node.table_text) {
                        *current.table_text += "\n" + *node.table_text;
                    }
                }
                if (current.latex_text && node.latex_text) {
                    if (current.latex_text && node.latex_text) {
                        *current.latex_text += "\n" + *node.latex_text;
                    }
                }
                current.text += "\n" + node_text;
                current_lines += node_lines;
            } else {
                merged.push_back(current);
                current = node;
                current.text = node_text;
                current_lines = node_lines;
            }
            // 拆分超长
            while (current_lines > max_lines) {
                // 拆分text
                size_t line_cnt = 0, split_pos = 0;
                for (size_t i = 0; i < current.text.size(); ++i) {
                    if (current.text[i] == '\n')
                        ++line_cnt;
                    if (line_cnt == max_lines) {
                        split_pos = i;
                        break;
                    }
                }
                MarkdownNode part = current;
                part.text = current.text.substr(0, split_pos);
                if (current.rich_text)
                    part.rich_text = part.text;
                if (current.code_text)
                    part.code_text = part.text;
                if (current.table_text)
                    part.table_text = part.text;
                if (current.latex_text)
                    part.latex_text = part.text;
                merged.push_back(part);
                // 剩余部分
                current.text = current.text.substr(split_pos + 1);
                if (current.rich_text)
                    current.rich_text = current.text;
                if (current.code_text)
                    current.code_text = current.text;
                if (current.table_text)
                    current.table_text = current.text;
                if (current.latex_text)
                    current.latex_text = current.text;
                current_lines = count_lines(current.text);
            }
        }
        if (has_current && !current.text.empty()) {
            merged.push_back(current);
        }
        return merged;
    }

    void BotAdapter::send_long_plain_text_reply(const Sender &sender, const std::string &text, bool at_target,
                                                uint64_t msg_length_limit,
                                                std::optional<std::function<void(uint64_t &)>> out_message_id_option,
                                                std::optional<database::UserPreference> user_preference_option) {
        const auto sync_id_base = generate_send_replay_sync_id(sender);

        // Parse llm reply content
        auto markdown_node = wheel::parse_markdown(text);

        float font_size = 15;
        int body_width = 350;

        // Check if it's simple text (not markdown blocks)
        if (markdown_node.empty() || (markdown_node.size() == 1 && !markdown_node[0].render_html_text.has_value())) {
            spdlog::info("Markdown text is short and no render HTML.");
            send_replay_msg(sender, make_message_chain_list(PlainTextMessage(text)), true, out_message_id_option);
            return;
        }

        std::function<void(const std::string_view sync_id, const MessageChainPtrList &msg_chain)> send_func;
        if (const auto group_sender = try_group_sender(sender)) {
            send_func = [this, group_sender, out_message_id_option](const std::string_view sync_id,
                                                                    const MessageChainPtrList &msg_chain) {
                send_group_message(group_sender->get().group, msg_chain, sync_id, out_message_id_option);
            };
            if (at_target) {
                spdlog::info("输出长文信息: @target");
                send_func(fmt::format("{}_at", sync_id_base), make_message_chain_list(AtTargetMessage(sender.id)));
            }
        } else {
            send_func = [this, sender, out_message_id_option](const std::string_view sync_id,
                                                              const MessageChainPtrList &msg_chain) {
                send_message(sender, msg_chain, sync_id, out_message_id_option);
            };
        }

        bool should_render_markdown =
            user_preference_option.has_value() ? user_preference_option->render_markdown_output : true;
        bool should_output_text = user_preference_option.has_value() ? user_preference_option->text_output : false;

        size_t render_html_count = 0;
        size_t index = 0;
        std::vector<ForwardMessageNode> forward_nodes;
        const auto &config = Config::instance();

        for (const auto &node : markdown_node) {
            // Handle markdown rendering if user preference allows it
            if (node.render_html_text.has_value() || node.code_text.has_value()) {
                std::string file_name = fmt::format("{}{}_markdown_render_block_{}", config.temp_res_path, sync_id_base,
                                                    render_html_count++);
                float font_size = 15;
                // Calculate HTML content width and height based on content
                float max_size = 20.f;
                size_t line_count = 1;
                for (auto line : SplitString(node.text, '\n')) {
                    if (line.length() > max_size) {
                        max_size = line.length();
                    }
                    ++line_count;
                }
                if (should_render_markdown || node.code_text.has_value()) {
                    nlohmann::json send_json = nlohmann::json{{
                                                                  "html",
                                                                  *node.render_html_text,
                                                              },
                                                              {"save_path", file_name + ".png"},
                                                              {"font_size", font_size}};
                    if (node.rich_text.has_value() && !node.code_text.has_value() && !node.table_text.has_value() &&
                        !node.latex_text.has_value()) {
                        send_json["html"] = replace_str(*node.render_html_text, "\n", "<br/>");
                        send_json.push_back({"body_width", 350});

                    } else if (node.code_text.has_value()) {

                        float width = std::max(20.f, max_size * font_size / 2.f) + 40.f +
                                      font_size; // Estimate width from content length
                        float height = std::max(20.f, (line_count + 2) * font_size * 1.5f) +
                                       20.f; // Estimate height based on number of lines
                        send_json.push_back({"width", (int64_t)width});
                        send_json.push_back({"height", (int64_t)height});
                        send_json.push_back({"body_width", (int64_t)width});
                    }
                    cpr::Response response = cpr::Post(
                        cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, "html_to_image")},
                        cpr::Body{std::string(send_json.dump())}, cpr::Header{{"Content-Type", "application/json"}});
                    spdlog::info("Render HTML text to image: {}.png", file_name);
                    // 发送图片消息
                    spdlog::info("长文块: {}, 发送图片消息: {}.png", index++, file_name);
                }

                // code display   node
                // Use planc's code display & run project (https://github.com/hubenchang0515)
                if (node.code_text.has_value() && line_count > 3 && line_count <= 100) {
                    std::string code_text_param = wheel::url_encode(*node.code_text);
                    code_text_param = base64::to_base64(code_text_param);

                    forward_nodes.push_back(ForwardMessageNode(
                        bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                        make_message_chain_list(
                            LocalImageMessage{file_name + ".png"},
                            PlainTextMessage(fmt::format(
                                " 你可以在这个链接下查看并运行代码哦: https://xplanc.org/shift/?lang={}&code={}",
                                *node.code_language, code_text_param))),
                        std::nullopt, std::nullopt));

                } else {

                    // Short line length, Markdown rener picture and line.
                    if (node.text.length() <= MAX_OUTPUT_LENGTH) {
                        MessageChainPtrList msg_chain;
                        if (should_render_markdown) {
                            msg_chain.push_back(
                                std::make_shared<LocalImageMessage>(LocalImageMessage{file_name + ".png"}));
                        }
                        if (should_output_text) {
                            msg_chain.push_back(std::make_shared<PlainTextMessage>(PlainTextMessage(node.text)));
                        }
                        forward_nodes.push_back(ForwardMessageNode(bot_profile.id, std::chrono::system_clock::now(),
                                                                   bot_profile.name, std::move(msg_chain), std::nullopt,
                                                                   std::nullopt));
                        continue;
                    }

                    // One Markdown render picture only, and need to split_output like normal plain text(text too long)
                    if (should_render_markdown) {
                        forward_nodes.push_back(
                            ForwardMessageNode(bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                                               make_message_chain_list(LocalImageMessage{file_name + ".png"}),
                                               std::nullopt, std::nullopt));
                    }
                }
            }

            // Output text if user preference allows it, regardless of whether markdown was rendered
            if (should_output_text || node.code_text.has_value() || !node.render_html_text.has_value()) {
                const auto split_output = Utf8Splitter(node.text, msg_length_limit);
                for (auto chunk : split_output) {
                    spdlog::info("长文块: {}, {}", index, chunk);
                    forward_nodes.push_back(ForwardMessageNode(
                        bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                        make_message_chain_list(PlainTextMessage(chunk)), std::nullopt, std::nullopt));
                    ++index;
                }
            }
        }

        const auto bot_profile = get_bot_profile();

        if (forward_nodes.size() == 1) {
            spdlog::info("仅有单块内容，直接发送该消息链");
            send_func(fmt::format("{}_img", sync_id_base), forward_nodes.front().message_chain);
            return;
        }

        for (size_t i = 0; i < forward_nodes.size(); ++i) {
            forward_nodes[forward_nodes.size() - i].time -= std::chrono::seconds(i);
        }
        spdlog::info("输出长文信息: long text");
        const auto forward_msg = ForwardMessage(forward_nodes, std::nullopt);
        spdlog::debug("forward message: {}", forward_msg.to_json().dump());
        send_func(fmt::format("{}_forward", sync_id_base), make_message_chain_list(forward_msg));
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

    std::vector<GroupInfo> BotAdapter::fetch_bot_group_list_info_sync() {
        auto res = send_command_sync(
            AdapterCommand(fmt::format("fetch_bot_group_info_{}", get_current_time_formatted()), "groupList"));
        std::vector<GroupInfo> group_info_list;

        if (const auto &data = get_optional(res, "data"); data.has_value()) {
            for (const auto &group : *data) {
                group_info_list.emplace_back(
                    get_optional<qq_id_t>(group, "id").value_or(-1),
                    get_optional<std::string_view>(group, "name").value_or(UNKNOWN_VALUE),
                    get_group_permission(get_optional<std::string_view>(group, "permission").value_or("UNKNOWN")));
            }
        }
        return std::move(group_info_list);
    }

    void BotAdapter::update_group_info_sync() {
        spdlog::info("Start update group info list");
        auto group_info_list = fetch_bot_group_list_info_sync();
        spdlog::info("Fetch group info list: groups count: {}", group_info_list.size());

        std::unordered_map<qq_id_t, GroupWrapper> group_wrapper_map;
        for (auto &group_info : group_info_list) {
            spdlog::info("Group info: {}({}), bot perm: {}", group_info.name, group_info.group_id,
                         to_string(group_info.bot_in_group_permission));
            GroupWrapper group_wrapper = GroupWrapper{group_info};

            // Fetch member info
            spdlog::info("Fetch members info for group: {}({})", group_info.name, group_info.group_id);
            *group_wrapper.member_info_list = group_by(fetch_group_member_list_sync(group_info),
                                                       [](const GroupMemberInfo &member) { return member.id; });
            spdlog::info("Fetch members info for group: {}({}): member count: {}", group_info.name, group_info.group_id,
                         group_wrapper.member_info_list->size());

            std::vector<std::string> member_name_list;
            for (const auto &member : group_wrapper.member_info_list->iter()) {
                member_name_list.push_back(member.second.member_name);
            }

            group_wrapper_map.insert(std::make_pair(group_info.group_id, std::move(group_wrapper)));

// #ifdef __USE_LIBTORCH__
//             spdlog::info("计算member list的member name embedding matrix");
//             std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
//             auto member_name_embedding_matrix =
//                 neural_network::get_model_set().text_embedding_model->embed(member_name_list);
//             std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
//             spdlog::info("计算member name embedding matrix cost: {}ms",
//                          std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
//             group_member_name_embedding_map.get_or_emplace_value(group_info.group_id,
//                                                                  std::move(member_name_embedding_matrix));
// #endif // __USE_LIBTORCH__
        }
        spdlog::info("Fetch group info list successed.");
        group_info_map = std::move(group_wrapper_map);
    }

    std::optional<Profile> BotAdapter::fetch_group_member_profile_sync(qq_id_t group_id, qq_id_t id) {
        if (auto result_json = send_command_sync(
                AdapterCommand(fmt::format("get memberProfile_{}", get_current_time_formatted()), "memberProfile",
                               std::make_shared<GetGroupMemberProfileContent>(group_id, id)));
            result_json.has_value()) {
            const auto &res = *result_json;
            const auto name = get_optional<std::string>(res, "nickname").value_or(EMPTY_JSON_STR_VALUE);
            const auto email = get_optional<std::string>(res, "email").value_or(EMPTY_JSON_STR_VALUE);
            const auto age = get_optional<uint32_t>(res, "age").value_or(0);
            const auto level = get_optional<uint32_t>(res, "level").value_or(0);
            const ProfileSex sex =
                from_string(get_optional<std::string_view>(res, "sex").value_or(EMPTY_JSON_STR_VALUE));
            return Profile{id, std::move(name), std::move(email), age, level, sex};
        }
        return std::nullopt;
    }

    std::vector<GroupMemberInfo> BotAdapter::fetch_group_member_list_sync(const GroupInfo &group_info) {
        auto json_res = send_command_sync(
            AdapterCommand{fmt::format("fetch_group_member_info_{}", get_current_time_formatted()), "memberList",
                           std::make_shared<CommandJsonContent>(nlohmann::json{{"target", group_info.group_id}})});
        std::vector<GroupMemberInfo> ret;
        if (!json_res.has_value()) {
            return ret;
        }
        const auto &res = get_optional(json_res, "data");
        if (!res.has_value()) {
            return ret;
        }
        for (auto &member : *res) {
            const qq_id_t id = get_optional(member, "id").value_or(UNKNOWN_ID);
            const std::string member_name = get_optional(member, "memberName").value_or(EMPTY_JSON_STR_VALUE);
            spdlog::debug("Group {}({}) Fetch member info: {}({})", group_info.name, group_info.group_id, member_name,
                          id);
            GroupMemberInfo member_info{
                id,
                group_info.group_id,
                std::move(member_name),
                get_optional(member, "specialTitle"),
                get_group_permission(get_optional<std::string>(member, "permission").value_or(UNKNOWN_VALUE)),
                map_optional(get_optional<uint64_t>(member, "joinTimestamp"),
                             [](auto val) { return timestamp_to_timepoint(val); }),
                map_optional(get_optional<uint64_t>(member, "lastSpeakTimestamp"),
                             [](auto val) { return timestamp_to_timepoint(val); }),
                get_optional<float>(member, "muteTimeRemaining").value_or(0.f)};
            ret.push_back(std::move(member_info));
        }
        return std::move(ret);
    }

    std::vector<GroupInfo> BotAdapter::get_bot_all_group_info() const {
        std::vector<GroupInfo> ret;

        for (const auto &group : group_info_map.iter()) {
            ret.push_back(group.second.group_info);
        }

        return ret;
    }

} // namespace bot_adapter
