#include "agent/simple_chat_action_agent.h"
#include "agent/llm.h" // for gen_common_prompt
#include "cli_handler.h"
#include "event.h" // try_to_replay_person/release_processing_replay_person
#include "global_data.h"
#include "rag.h"
#include "utils.h"
#include "vec_db/weaviate.h"
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>
#include <thread>

using wheel::join_str;
using wheel::ltrim;
using wheel::rtrim;

namespace agent {

    // local copy of helper from llm.cpp (consider centralizing later)
    inline std::string get_permission_chs(const std::string_view perm) {
        if (perm == "OWNER") {
            return "群主";
        } else if (perm == "ADMINISTRATOR") {
            return "管理员";
        }
        return "普通群友";
    }

    // helper conversions (duplicated from llm.cpp; consider refactor later)
    inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, ChatMessage msg) {
        msg_json.push_back(msg.to_json());
        return msg_json;
    }

    inline nlohmann::json msg_list_to_json(const std::deque<ChatMessage> &msg_list,
                                           const std::optional<std::string_view> system_prompt_option = std::nullopt) {
        nlohmann::json msg_json = nlohmann::json::array();
        if (system_prompt_option.has_value()) {
            msg_json.push_back(nlohmann::json{{"role", "system"}, {"content", system_prompt_option.value()}});
        }
        for (const auto &msg : msg_list) {
            nlohmann::json msg_entry = msg.to_json();
            spdlog::info("msg_entry: {}", msg_entry.dump());
            msg_json.push_back(std::move(msg_entry));
        }
        return msg_json;
    }

    inline nlohmann::json get_msg_json(const qq_id_t id, std::string name,
                                       const std::optional<std::string> &system_prompt_option = std::nullopt) {
        auto session = g_chat_session_map.get_or_create_value(
            id, [name = std::move(name)]() mutable { return ChatSession(std::move(name)); });
        return msg_list_to_json(session->message_list, system_prompt_option);
    }

    void insert_tool_call_record_async(const std::string &sender_name, qq_id_t sender_id,
                                       const nlohmann::json &msg_json, const std::string &func_name,
                                       const std::string &func_arguments, const std::string &tool_content) {
        std::thread([=] {
            set_thread_name("insert tool call record");
            spdlog::info("Start insert tool call record thread.");
            database::get_global_db_connection().insert_tool_calls_record(
                sender_name, sender_id, msg_json.dump(), std::chrono::system_clock::now(),
                fmt::format("{}({})", func_name, func_arguments), tool_content);
        }).detach();
    }

    void SimpleChatActionAgent::process_llm(const bot_cmd::CommandContext &context,
                                            const std::optional<std::string> &additional_system_prompt_option,
                                            const std::optional<database::UserPreference> &user_preference_option,
                                            const std::optional<nlohmann::json> &function_tools_opt) {
        spdlog::info("[SimpleChatActionAgent] 开始处理LLM信息");

        if (!try_to_replay_person(context.event->sender_ptr->id)) {
            spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.event->sender_ptr->id);
            adapter->send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("我还在思考中...你别急")));
            return;
        }

        spdlog::debug("Event type: {}, Sender json: {}", context.event->get_typename(),
                      context.event->sender_ptr->to_json().dump());

        std::string llm_content{};
        if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
            llm_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                           *context.msg_prop.ref_msg_content));
        }
        if (!context.msg_prop.at_id_set.empty() &&
            !(context.msg_prop.at_id_set.size() == 1 &&
              *context.msg_prop.at_id_set.begin() == adapter->get_bot_profile().id)) {
            llm_content.append("本消息提到了：");
            bool first = true;
            for (const auto &at_id : context.msg_prop.at_id_set) {
                if (!first) {
                    llm_content.append("、");
                }
                llm_content.append(fmt::format("{}", at_id));
                first = false;
            }
            llm_content.append("。");
        }
        std::string speak_content = EMPTY_MSG_TAG;
        if (context.msg_prop.plain_content != nullptr &&
            !wheel::ltrim(wheel::rtrim(*context.msg_prop.plain_content)).empty()) {
            speak_content = *context.msg_prop.plain_content;
        }

        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            if (auto group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr);
                group_sender.has_value()) {
                llm_content.append(fmt::format("{}\"{}\"({})[{}]对你说: \"{}\"",
                                               get_permission_chs(group_sender->get().permission),
                                               group_sender->get().name, context.event->sender_ptr->id,
                                               get_current_time_formatted(), speak_content));
            } else {
                llm_content.append(fmt::format("\"{}\"({})[{}]对你说: \"{}\"", context.event->sender_ptr->name,
                                               context.event->sender_ptr->id, get_current_time_formatted(),
                                               speak_content));
            }
        }
        std::string mixed_input_content;
        if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
            mixed_input_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                                   *context.msg_prop.ref_msg_content));
        }
        if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
            mixed_input_content.append(fmt::format("{}", *context.msg_prop.plain_content));
        }
        auto session_knowledge_opt = rag::query_knowledge(mixed_input_content, false, context.event->sender_ptr->id,
                                                          context.event->sender_ptr->name);
        std::string system_prompt = gen_common_prompt(adapter->get_bot_profile(), *adapter, *context.event->sender_ptr,
                                                      context.is_deep_think, additional_system_prompt_option);

        // Add session knowledge to system prompt if available
        if (session_knowledge_opt.has_value()) {
            system_prompt += "\n" + session_knowledge_opt.value();
        }

        spdlog::info("作为用户输入给llm的content: {}", llm_content);

        auto llm_thread = std::thread([this, context, llm_content, system_prompt, user_preference_option,
                                         function_tools_opt] {
            on_llm_thread(context, llm_content, system_prompt, user_preference_option, function_tools_opt);
        });

        llm_thread.detach();
    }

    std::string get_target_group_chat_history(const bot_adapter::BotAdapter &adapter, const qq_id_t group_id,
                                              qq_id_t target_id) {
        std::string target_name = std::to_string(target_id); // Default name is ID
        const auto &member_list = adapter.fetch_group_member_info(group_id)->get().member_info_list;
        if (auto member_info = member_list->find(target_id); member_info.has_value()) {
            target_name = member_info->get().member_name;
        }

        // Fetch recent messages from the group and filter by target user
        const auto &group_msg_list = g_group_message_storage.get_individual_last_msg_list(group_id, 1000);

        std::vector<std::string> target_msgs;
        target_msgs.reserve(100);
        size_t size_count = 0;
        for (auto it = group_msg_list.crbegin(); it != group_msg_list.crend(); ++it) {
            const auto &msg = *it;
            if (msg.sender_id == target_id) {
                auto text = bot_adapter::get_text_from_message_chain(*msg.message_chain_list);
                size_count += text.size();
                if (size_count > 3000) {
                    break;
                }
                target_msgs.insert(target_msgs.begin(),
                                   fmt::format("[{}]'{}': '{}'", time_point_to_db_str(msg.send_time), msg.sender_name,
                                               std::move(text)));
            }
        }

        if (target_msgs.empty()) {
            return fmt::format("在'{}'群里没有找到 '{}' 的聊天记录", group_id, target_name);
        } else {
            return fmt::format("'{}'群内 '{}' 的最近消息:\n{}", group_id, target_name,
                               join_str(target_msgs.cbegin(), target_msgs.cend(), "\n"));
        }
    }

    void SimpleChatActionAgent::process_tool_calls(const bot_cmd::CommandContext &context, nlohmann::json &msg_json,
                                                   std::vector<ChatMessage> &one_chat_session,
                                                   const std::optional<nlohmann::json> &function_tools_opt) {
        // loop check if have function call
        while (one_chat_session.rbegin()->tool_calls) {
            const auto &llm_res = *one_chat_session.rbegin();

            // llm request tool call
            add_to_msg_json(msg_json, llm_res);
            std::vector<ChatMessage> append_tool_calls;
            for (const auto &func_calls : *llm_res.tool_calls) {
                spdlog::info("Tool calls: {}()", func_calls.name, func_calls.arguments);
                std::optional<ChatMessage> tool_call_msg = std::nullopt;
                if (func_calls.name == "search_info") {
                    const auto arguments = nlohmann::json::parse(func_calls.arguments);
                    const std::optional<std::string> &query = get_optional(arguments, "query");
                    bool include_date = get_optional(arguments, "includeDate").value_or(false);
                    spdlog::info("Function call id {}: search_info(query={}, include_date={})", func_calls.id,
                                 query.value_or(EMPTY_JSON_STR_VALUE), include_date);
                    if (!query.has_value() || query->empty())
                        spdlog::warn("Function call id {}: search_info(query={}, include_date={}), query is null",
                                     func_calls.id, query.value_or(EMPTY_JSON_STR_VALUE), include_date);

                    std::string content;
                    const auto knowledge_list = vec_db::query_knowledge_from_vec_db(*query, 0.7f);
                    for (const auto &knowledge : knowledge_list)
                        content += fmt::format("{}\n", knowledge.content);

                    const auto net_search_list = rag::net_search_content(
                        include_date ? fmt::format("{} {}", get_current_time_formatted(), *query) : *query);
                    std::vector<bot_adapter::ForwardMessageNode> first_replay;
                    first_replay.emplace_back(context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                                              context.adapter.get_bot_profile().name,
                                              bot_adapter::make_message_chain_list(
                                                  bot_adapter::PlainTextMessage(fmt::format("搜索: \"{}\"", *query))));
                    for (const auto &net_search : net_search_list) {
                        content += fmt::format("{}( {} ):{}\n", net_search.title, net_search.url, net_search.content);
                        first_replay.emplace_back(
                            context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                            context.adapter.get_bot_profile().name,
                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                fmt::format("关联度: {:.2f}%\n{}( {} )", net_search.score * 100.0f, net_search.title,
                                            net_search.url))));
                    }
                    if (!first_replay.empty()) {
                        context.adapter.send_replay_msg(
                            *context.event->sender_ptr,
                            bot_adapter::make_message_chain_list(bot_adapter::ForwardMessage(
                                first_replay, bot_adapter::DisplayNode(std::string("联网搜索结果")))));
                    }
                    tool_call_msg = ChatMessage(ROLE_TOOL, content, func_calls.id);
                } else if (func_calls.name == "fetch_url_content") {
                    const auto arguments = nlohmann::json::parse(func_calls.arguments);
                    const std::vector<std::string> urls =
                        get_optional(arguments, "urls").value_or(std::vector<std::string>());
                    spdlog::info("Function call id {}: fetch_url_content(urls=[{}])", func_calls.id,
                                 join_str(std::cbegin(urls), std::cend(urls)));
                    if (urls.empty()) {
                        spdlog::info("Function call id {}: fetch_url_content(urls=[{}])", func_calls.id,
                                     join_str(std::cbegin(urls), std::cend(urls)));
                        context.adapter.send_long_plain_text_reply(*context.event->sender_ptr,
                                                                   "你发的啥,我看不到...再发一遍呢?", true);
                    } else {
                        context.adapter.send_long_plain_text_reply(
                            *context.event->sender_ptr,
                            ltrim(rtrim(llm_res.content)).empty() ? "等我看看这个链接哦..." : llm_res.content, true);
                    }

                    const auto url_search_res = rag::url_search_content(urls);
                    std::string content;

                    // Process successful results
                    for (const auto &[url, raw_content] : url_search_res.results) {
                        content += fmt::format("链接[{}]内容:\n{}\n\n", url, raw_content);
                    }

                    // Process failed results
                    if (!url_search_res.failed_reason.empty()) {
                        content += "以下链接获取失败:\n";
                        for (const auto &[url, error] : url_search_res.failed_reason) {
                            content += fmt::format("链接[{}]失败原因: {}\n", url, error);
                        }
                    }

                    if (url_search_res.results.empty()) {
                        spdlog::error("url_search: {} failed", join_str(std::cbegin(urls), std::cend(urls)));
                        tool_call_msg = ChatMessage(
                            ROLE_TOOL,
                            "抱歉，所有链接都获取失败了,可能是网络抽风了或者网站有反爬机制导致紫幻获取不到内容",
                            func_calls.id);
                    } else {
                        tool_call_msg = ChatMessage(ROLE_TOOL, content, func_calls.id);
                    }
                } else if (func_calls.name == "view_model_info") {
                    std::string model_info_str = "获取模型信息失败";
                    if (auto model_info = fetch_model_info(); model_info.has_value()) {
                        model_info_str = model_info->dump();
                    }
                    tool_call_msg = ChatMessage(ROLE_TOOL, model_info_str, func_calls.id);
                } else if (func_calls.name == "view_chat_history") {
                    const auto arguments = nlohmann::json::parse(func_calls.arguments);

                    // try with qq id
                    std::optional<qq_id_t> target_id_opt = get_optional<qq_id_t>(arguments, "targetId");
                    std::string content = "啥记录都没有";

                    if (const auto &group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr);
                        group_sender.has_value()) {
                        // Group Chat
                        if (target_id_opt.has_value()) {
                            // Target is specified in group chat
                            spdlog::info("查询群 '{}' 内目标 '{}'(qq号) 的聊天记录", group_sender->get().group.name,
                                         *target_id_opt);

                            content = get_target_group_chat_history(context.adapter, group_sender->get().group.id,
                                                                    *target_id_opt);

                        } else {
                            // Try with member name
                            auto target_name = get_optional<std::string>(arguments, "targetName");
                            if (target_name.has_value() && !target_name->empty()) {
                                spdlog::info("查询群 '{}' 内目标 '{}' 的聊天记录", group_sender->get().group.name,
                                             *target_name);
                                // query member name
                                std::vector<qq_id_t> target_id_list =
                                    context.adapter.group_member_name_embedding_map.find(group_sender->get().group.id)
                                        ->get()
                                        .get_similar_member_names(*target_name, 0.5f);
                                if (target_id_list.empty()) {
                                    content = fmt::format("在'{}'群里没有找到名为'{}'的群友",
                                                          group_sender->get().group.name, *target_name);

                                } else {
                                    content = "";
                                    for (const auto &target_id : target_id_list) {
                                        if (!content.empty()) {
                                            content += "\n";
                                        }
                                        spdlog::info("查询群 '{}' 内目标 '{}' 的聊天记录",
                                                     group_sender->get().group.name, target_id);
                                        content += get_target_group_chat_history(
                                            context.adapter, group_sender->get().group.id, target_id);
                                    }
                                }
                            } else {

                                // No target, get group's history
                                const auto &msg_list = g_group_message_storage.get_individual_last_msg_list(
                                    group_sender->get().group.id, 1000);
                                if (msg_list.empty()) {
                                    content = fmt::format("在'{}'群里还没有聊天记录哦", group_sender->get().group.name);
                                } else {
                                    std::string text;
                                    size_t total_length = 0;
                                    std::vector<std::string> msg_texts;
                                    for (auto it = msg_list.crbegin(); it != msg_list.crend(); ++it) {
                                        const auto &msg = *it;
                                        std::string msg_text = fmt::format(
                                            "[{}]'{}': '{}'", msg.sender_name, time_point_to_db_str(msg.send_time),
                                            bot_adapter::get_text_from_message_chain(*msg.message_chain_list));
                                        if (total_length + msg_text.length() > 3000) {
                                            break;
                                        }
                                        msg_texts.insert(msg_texts.begin(), msg_text);
                                        total_length += msg_text.length();
                                    }

                                    text = join_str(msg_texts.cbegin(), msg_texts.cend(), "\n");

                                    content =
                                        fmt::format("'{}'群的最近消息:\n{}", group_sender->get().group.name, text);
                                }
                            }
                        }
                    } else {
                        // Friend chat, ignore target
                        const auto &msg_list =
                            g_person_message_storage.get_individual_last_msg_list(context.event->sender_ptr->id, 1000);
                        if (msg_list.empty()) {
                            content = "我们之间还没有聊天记录哦";
                        } else {
                            std::string text;
                            size_t total_length = 0;
                            for (const auto &msg : msg_list) {
                                std::string msg_text =
                                    fmt::format("'{}': '{}'", msg.sender_name,
                                                bot_adapter::get_text_from_message_chain(*msg.message_chain_list));
                                if (total_length + msg_text.length() > 3000) {
                                    break;
                                }
                                if (!text.empty()) {
                                    text += "\n";
                                }
                                text += msg_text;
                                total_length += msg_text.length();
                            }

                            content = fmt::format("与'{}'的最近聊天记录:\n{}", context.event->sender_ptr->name, text);
                        }
                    }
                    tool_call_msg = ChatMessage(ROLE_TOOL, content, func_calls.id);
                } else if (func_calls.name == "query_group") {
                    const auto arguments = nlohmann::json::parse(func_calls.arguments);
                    const std::string item = get_optional(arguments, "item").value_or("");
                    spdlog::info("Function call id {}: query_group(item={})", func_calls.id, item);
                    auto group_sender = try_group_sender(*context.event->sender_ptr);
                    std::string content;
                    if (!group_sender.has_value()) {
                        content = "你得告诉我是哪个群";
                    } else {
                        if (item == "OWNER") {
                            const auto &member_list =
                                context.adapter.fetch_group_member_info(group_sender->get().group.id)
                                    ->get()
                                    .member_info_list;
                            for (const auto &[key, value] : member_list->iter()) {
                                if (value.permission == bot_adapter::GroupPermission::OWNER) {
                                    content += fmt::format("'{}'群的群主是: '{}'(QQ号为:{})",
                                                           group_sender->get().group.name, value.member_name, value.id);
                                    if (value.special_title.has_value()) {
                                        content += fmt::format("(特殊头衔: {})", value.special_title.value());
                                    }
                                    if (value.last_speak_time.has_value()) {
                                        content += fmt::format("(最后发言时间: {})",
                                                               system_clock_to_string(value.last_speak_time.value()));
                                    }
                                    break;
                                }
                            }
                        } else if (item == "ADMIN") {

                            const auto &member_list =
                                context.adapter.fetch_group_member_info(group_sender->get().group.id)
                                    ->get()
                                    .member_info_list;
                            std::vector<std::string> admin_info_list;
                            for (const auto &[key, value] : member_list->iter()) {
                                if (value.permission == bot_adapter::GroupPermission::ADMINISTRATOR) {
                                    std::string admin_info =
                                        fmt::format("'{}'(QQ号为:{})", value.member_name, value.id);
                                    if (value.special_title.has_value()) {
                                        admin_info += fmt::format("(特殊头衔: {})", value.special_title.value());
                                    }
                                    if (value.last_speak_time.has_value()) {
                                        admin_info +=
                                            fmt::format("(最后发言时间: {})",
                                                        system_clock_to_string(value.last_speak_time.value()));
                                    }
                                    admin_info_list.push_back(admin_info);
                                }
                            }
                            if (admin_info_list.empty()) {
                                content = fmt::format("'{}'群里没有管理员", group_sender->get().group.name);
                            } else {
                                content = fmt::format(
                                    "'{}'群的管理员有:\n{}", group_sender->get().group.name,
                                    join_str(std::cbegin(admin_info_list), std::cend(admin_info_list), "\n"));
                            }
                        } else if (item == "PROFILE") {
                            content = "暂未实现";
                        } else if (item == "NOTICE") {
                            auto announcements_opt =
                                context.adapter.get_group_announcement_sync(group_sender->get().group.id);
                            if (!announcements_opt.has_value()) {
                                content = "获取群公告失败,可能是网络波动或没有权限";
                            } else {
                                if (announcements_opt->empty()) {
                                    content = "本群没有群公告";
                                } else {
                                    const auto &member_list =
                                        context.adapter.fetch_group_member_info(group_sender->get().group.id)
                                            ->get()
                                            .member_info_list;
                                    std::vector<std::string> announcements_str_list;
                                    for (const auto &anno : *announcements_opt) {
                                        std::string sender_info;
                                        if (auto member_info_opt = member_list->find(anno.sender_id);
                                            member_info_opt.has_value()) {
                                            const auto &member_info = member_info_opt->get();
                                            sender_info = fmt::format("{}({})", member_info.member_name,
                                                                      get_permission_chs(member_info.permission));
                                        } else {
                                            sender_info = std::to_string(anno.sender_id);
                                        }
                                        std::string anno_str = fmt::format(
                                            "内容: {}\n发送者: {}\n发送时间: {}\n已确认人数: {}", anno.content,
                                            sender_info, system_clock_to_string(anno.publication_time),
                                            anno.confirmed_members_count);
                                        announcements_str_list.push_back(anno_str);
                                    }

                                    content = fmt::format("群'{}'的公告:\n", group_sender->get().group.name);
                                    content += join_str(std::cbegin(announcements_str_list),
                                                        std::cend(announcements_str_list), "\n---\n");
                                }
                            }
                            spdlog::info("获取群公告: {}", content);
                        } else {
                            content = "暂未实现";
                        }
                    }

                    tool_call_msg = ChatMessage(ROLE_TOOL, content, func_calls.id);
                } else if (func_calls.name == "get_function_list") {
                    tool_call_msg =
                        ChatMessage(ROLE_TOOL, fmt::format("可用功能/函数列表: {}", bot_cmd::get_available_commands()),
                                    func_calls.id);
                } else {
                    spdlog::error("Function {} is not impl.", func_calls.name);
                    tool_call_msg = ChatMessage(ROLE_TOOL,
                                                "主人还没有实现这个功能,快去github页面( "
                                                "https://github.com/FredYakumo/zihuan-aibot-800a )提issues吧",
                                                func_calls.id);
                }

                if (tool_call_msg.has_value()) {
                    insert_tool_call_record_async(context.event->sender_ptr->name, context.event->sender_ptr->id,
                                                  msg_json, func_calls.name, func_calls.arguments,
                                                  tool_call_msg->content);
                    append_tool_calls.emplace_back(std::move(*tool_call_msg));
                }
            }
            if (!append_tool_calls.empty()) {
                for (auto &append : append_tool_calls) {
                    add_to_msg_json(msg_json, append);
                    one_chat_session.emplace_back(std::move(append));
                }
                // one_chat_session.insert(std::end(one_chat_session),
                //                             std::make_move_iterator(std::begin(append_tool_calls)),
                //                             std::make_move_iterator(std::end(append_tool_calls)));
            }

            AgentInferenceParam inference_param;
            inference_param.think_mode = context.is_deep_think;
            inference_param.messages_json = msg_json;
            inference_param.function_tools_opt = function_tools_opt;
            if (auto llm_res = bind_output_llm_agent->inference(inference_param); llm_res.has_value()) {
                one_chat_session.push_back(std::move(*llm_res));
            } else {
                spdlog::warn("LLM did not response any chat message...");
                context.adapter.send_replay_msg(*context.event->sender_ptr, bot_adapter::make_message_chain_list(
                                                                                bot_adapter::PlainTextMessage("?")));
                release_processing_replay_person(context.event->sender_ptr->id);

                return;
            }
        }
    }

    void SimpleChatActionAgent::on_llm_thread(const bot_cmd::CommandContext &context, const std::string &llm_content,
                                              const std::string &system_prompt,
                                              const std::optional<database::UserPreference> &user_preference_option,
                                              const std::optional<nlohmann::json> &function_tools_opt) {
        spdlog::debug("Event type: {}, Sender json: {}", context.event->get_typename(),
                      context.event->sender_ptr->to_json().dump());

        spdlog::info("Start llm thread.");
        set_thread_name("AIBot LLM process");
        if (context.is_deep_think) {
            spdlog::info("开始深度思考");
        }
        const auto bot_profile = context.adapter.get_bot_profile();

        if (context.is_deep_think) {
            context.adapter.send_replay_msg(
                *context.event->sender_ptr,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage{"正在思思考中..."},
                                                     bot_adapter::ImageMessage{Config::instance().think_image_url}),
                false);
        }

        auto msg_json = get_msg_json(context.event->sender_ptr->id, context.event->sender_ptr->name, system_prompt);

        auto user_chat_msg = ChatMessage(ROLE_USER, llm_content);
        add_to_msg_json(msg_json, user_chat_msg);

        std::vector<ChatMessage> one_chat_session;
        AgentInferenceParam inference_param;
        inference_param.think_mode = context.is_deep_think;
        inference_param.messages_json = msg_json;
        inference_param.function_tools_opt = function_tools_opt;
        if (auto llm_res = g_llm_chat_agent->inference(inference_param); llm_res.has_value()) {
            one_chat_session.push_back(std::move(*llm_res));
        } else {
            spdlog::warn("LLM did not response any chat message...");
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("?")));
            release_processing_replay_person(context.event->sender_ptr->id);
            return;
        }

        process_tool_calls(context, msg_json, one_chat_session, function_tools_opt);

        std::string replay_content = one_chat_session.rbegin()->content;

        if (auto session = g_chat_session_map.find(context.event->sender_ptr->id); session.has_value()) {
            auto &message_list = session->get().message_list;

            // Add msg to global storage
            message_list.push_back(user_chat_msg);
            message_list.insert(std::end(message_list), std::make_move_iterator(std::begin(one_chat_session)),
                                std::make_move_iterator(std::end(one_chat_session)));

            // Remove messages that exceed the limit
            size_t total_len = 0;
            auto sess_it = message_list.rbegin();

            while (sess_it != message_list.rend() && total_len < USER_SESSION_MSG_LIMIT) {
                total_len += sess_it->content.length(); // UTF-8 length
                ++sess_it;
            }

            if (sess_it != message_list.rend()) {
                spdlog::info("{}({})的对话长度超过限制, 删除'{}'之前的上下文内容", context.event->sender_ptr->name,
                             context.event->sender_ptr->name, sess_it->content);

                // @Purpose: Collect erase message's contain tool_call_ids
                std::unordered_set<std::string> related_tool_call_ids;
                for (auto it = message_list.cbegin(); it != sess_it.base(); ++it) {
                    if (it->tool_calls.has_value()) {
                        for (const auto &tc : *it->tool_calls) {
                            related_tool_call_ids.insert(tc.id);
                        }
                    }
                }

                // Use below method to erase, purpose for purpose for Remove all message which should be removed
                // also message_list.erase(message_list.begin(), sess_it.base());

                // Construct a new message_list that contains remain message(purpose for Remove all message which
                // should be removed)
                std::deque<ChatMessage> new_message_list;
                for (auto it = sess_it.base(); it != message_list.end(); ++it) {
                    if (it->tool_call_id.has_value() && related_tool_call_ids.contains(*it->tool_call_id)) {
                        continue;
                    }
                    new_message_list.push_back(std::move(*it));
                }
                message_list = std::move(new_message_list);
            }
        }

        spdlog::info("Prepare to send msg response");
        context.adapter.send_long_plain_text_reply(
            *context.event->sender_ptr, replay_content, true, MAX_OUTPUT_LENGTH,
            [context, replay_content](uint64_t message_id) {
                std::string mixed_input_content;
                if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
                    mixed_input_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",",
                                                           context.event->sender_ptr->name,
                                                           *context.msg_prop.ref_msg_content));
                }
                if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
                    mixed_input_content.append(fmt::format("\n{}", *context.msg_prop.plain_content));
                }

                auto input_emb = neural_network::get_model_set().text_embedding_model->embed(mixed_input_content);
                auto llm_output_emb = neural_network::get_model_set().text_embedding_model->embed(replay_content);
                if (const auto &group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr)) {
                    database::get_global_db_connection().insert_message(
                        message_id, replay_content,
                        bot_adapter::GroupSender(
                            CLIHandler::get_bot_id(), context.adapter.get_bot_profile().name, std::nullopt,
                            to_string(context.adapter.get_group(group_sender->get().group.id)
                                          .group_info.bot_in_group_permission),
                            std::nullopt, std::chrono::system_clock::now(), group_sender->get().group),
                        std::chrono::system_clock::now(), std::set<uint64_t>{context.event->sender_ptr->id});

                } else {
                    database::get_global_db_connection().insert_message(
                        message_id, replay_content,
                        bot_adapter::Sender(CLIHandler::get_bot_id(), context.adapter.get_bot_profile().name,
                                            std::nullopt),
                        std::chrono::system_clock::now(), std::set<uint64_t>{context.event->sender_ptr->id});
                }
            },
            user_preference_option);

        release_processing_replay_person(context.event->sender_ptr->id);

        g_last_chat_message_time_map.insert_or_assign(context.event->sender_ptr->id, std::chrono::system_clock::now());
    }
} // namespace agent