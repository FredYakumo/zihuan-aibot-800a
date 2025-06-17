#include "bot_adapter.h"
#include "adapter_cmd.hpp"
#include "adapter_event.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "config.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "easywsclient.hpp"
#include "get_optional.hpp"
#include "nlohmann/json_fwd.hpp"
#include "time_utils.h"
#include "utils.h"
#include <chrono>
#include <cpr/cpr.h>
#include <cstdint>
#include <fstream>
#include <future>
#include <general-wheel-cpp/markdown_utils.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bot_adapter {

    using namespace wheel;

    BotAdapter::~BotAdapter() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    int BotAdapter::start() {
        is_running = true;

        std::thread update_group_info_thread([this]() {
            const auto &config = Config::instance();
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
        update_group_info_thread.join();

        // TODO: Handle reconnection logic

        return 0;
    }

    struct ParseMessageChainResult {
        MessageChainPtrList message_chain;
        message_id_t message_id;
        std::chrono::system_clock::time_point send_time;
    };

    ParseMessageChainResult parse_message_chain(const nlohmann::json &msg_chain) {
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
                if (id && group_id) {
                    // TODO: 获取原始message
                }
                ret.push_back(std::make_shared<QuoteMessage>(quote_text, id.value_or(0)));
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
        const auto parse_result = parse_message_chain(*msg_chain_json);

        auto process_event = [&](auto sender_ptr, auto create_event) {
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
            process_event(group_sender_ptr, [](auto sender, const auto &parse_result) {
                return GroupMessageEvent(parse_result.message_id, sender, parse_result.message_chain,
                                         parse_result.send_time);
            });
        } else if (type == "FriendMessage") {
            auto sender_ptr = std::make_shared<Sender>(*sender_json);
            process_event(sender_ptr, [](auto sender, const auto &parse_result) {
                return FriendMessageEvent(parse_result.message_id, sender, parse_result.message_chain,
                                          parse_result.send_time);
            });
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

        send_command(AdapterCommand(sync_id, "sendFriendMessage",
                                    std::make_shared<bot_adapter::SendMsgContent>(sender.id, message_chain)));
    }

    void
    BotAdapter::send_group_message(const Group &group, const MessageChainPtrList &message_chain,
                                   std::optional<std::string_view> sync_id_option,
                                   std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option) {
        const auto sync_id = std::string(sync_id_option.value_or(generate_send_group_message_sync_id(group)));
        spdlog::info("Send message to group: {}, sync id: {}", to_string(group), sync_id);
        // const auto message_json = to_json(message_chain);

        send_command(AdapterCommand(sync_id, "sendGroupMessage",
                                    std::make_shared<bot_adapter::SendMsgContent>(group.id, message_chain)));
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

    void BotAdapter::send_long_plain_text_reply(const Sender &sender, std::string text, bool at_target,
                                                uint64_t msg_length_limit) {
        const auto sync_id_base = generate_send_replay_sync_id(sender);

        // Parse llm reply content
        auto markdown_node = wheel::parse_markdown(std::move(text));
        if (markdown_node.size() == 1 && !markdown_node[0].render_html_text.has_value() &&
            markdown_node[0].text.length() < msg_length_limit) {
            spdlog::info("Markdown text is short and no render HTML.");
            send_replay_msg(sender, make_message_chain_list(PlainTextMessage(text)), true);
            return;
        }

        std::function<void(const std::string_view sync_id, const MessageChainPtrList &msg_chain)> send_func;
        if (const auto group_sender = try_group_sender(sender)) {
            send_func = [this, group_sender](const std::string_view sync_id, const MessageChainPtrList &msg_chain) {
                send_group_message(group_sender->get().group, msg_chain, sync_id);
            };
            if (at_target) {
                spdlog::info("输出长文信息: @target");
                send_func(fmt::format("{}_at", sync_id_base), make_message_chain_list(AtTargetMessage(sender.id)));
            }
        } else {
            send_func = [this, sender](const std::string_view sync_id, const MessageChainPtrList &msg_chain) {
                send_message(sender, msg_chain, sync_id);
            };
        }

        size_t render_html_count = 0;
        size_t index = 0;
        std::vector<ForwardMessageNode> forward_nodes;
        const auto &config = Config::instance();
        for (const auto &node : markdown_node) {
            if (node.render_html_text.has_value()) {
                std::string file_name = fmt::format("{}{}_markdown_render_block_{}", config.temp_res_path, sync_id_base,
                                                    render_html_count++);

                // ofs << "<html>\n" << node.render_html_text.value() <<"</html>\n";
                // spdlog::info("Render HTML text to path: {}.html", file_name);
                float font_size = 15;
                std::string render_html = *node.render_html_text;
                nlohmann::json send_json = nlohmann::json{{
                                                              "html",
                                                              std::move(render_html),
                                                          },
                                                          {"save_path", file_name + ".png"},
                                                          {"font_size", font_size}};
                if (node.rich_text.has_value() && !node.code_text.has_value() && !node.table_text.has_value()) {
                    render_html = replace_str(render_html, "\n", "<br/>");
                    send_json.push_back({"body_width", 350});

                } else if (node.code_text.has_value()) {
                    // Calculate HTML content width and height based on content
                    float max_size = 20.f;
                    size_t line_count = 1;
                    for (auto line : SplitString(node.text, '\n')) {
                        if (line.length() > max_size) {
                            max_size = line.length();
                        }
                        ++line_count;
                    }
                    float width = std::max(20.f, max_size * font_size / 2.f) + 30.f +
                                  font_size; // Estimate width from content length
                    float height = std::max(20.f, line_count * font_size * 1.5f) +
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

                if (node.text.length() <= MAX_OUTPUT_LENGTH) {
                    forward_nodes.push_back(ForwardMessageNode(
                        bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                        make_message_chain_list(LocalImageMessage{file_name + ".png"}, PlainTextMessage(node.text)),
                        std::nullopt, std::nullopt));
                    continue;
                }
                forward_nodes.push_back(ForwardMessageNode(
                    bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                    make_message_chain_list(LocalImageMessage{file_name + ".png"}), std::nullopt, std::nullopt));
            }
            const auto split_output = Utf8Splitter(node.text, msg_length_limit);
            for (auto chunk : split_output) {
                spdlog::info("长文块: {}, {}", index, chunk);
                forward_nodes.push_back(
                    ForwardMessageNode(bot_profile.id, std::chrono::system_clock::now(), bot_profile.name,
                                       make_message_chain_list(PlainTextMessage(chunk)), std::nullopt, std::nullopt));
                ++index;
            }
        }

        const auto bot_profile = get_bot_profile();

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
            group_wrapper_map.insert(std::make_pair(group_info.group_id, std::move(group_wrapper)));
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
                get_group_permission(get_optional<std::string>("member", "permission").value_or(UNKNOWN_VALUE)),
                map_optional(get_optional<uint64_t>(member, "joinTimestamp"),
                             [](auto val) { return timestamp_to_timepoint(val); }),
                map_optional(get_optional<uint64_t>(member, "lastSpeakTimestamp"),
                             [](auto val) { return timestamp_to_timepoint(val); }),
                get_optional<float>(member, "muteTimeRemaining").value_or(0.f)};
            ret.push_back(std::move(member_info));
        }
        return std::move(ret);
    }

    // void BotAdapter::send_message_async(const Sender &sender, const MessageChainPtrList &message_chain,
    //                                     size_t max_retry_count, std::chrono::milliseconds timeout) {
    //     std::promise<nlohmann::json> promise;
    //     auto future = promise.get_future();
    //     auto sync_id = fmt::format("send_message_async_to_{}_{}", sender.id, get_current_time_formatted());
    //     send_message(sender, message_chain, sync_id, [&promise](const nlohmann::json &res) { promise.set_value(res);
    //     });

    //     if (future.wait_for(timeout) == std::future_status::timeout) {
    //         spdlog::error("Send command(sync) '{}'(sync id: {}) timeout. payload: {}", command.command,
    //         command.sync_id,
    //                       command.to_json().dump());
    //         throw std::runtime_error("Command execution timeout");
    //     }

    //     return future.get();
    // }

} // namespace bot_adapter