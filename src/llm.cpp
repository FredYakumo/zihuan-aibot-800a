#include "llm.h"
#include "adapter_message.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "chat_session.hpp"
#include "config.h"
#include "constants.hpp"
#include "database.h"
#include "get_optional.hpp"
#include "global_data.h"
#include "rag.h"
#include "utils.h"
#include <_strings.h>
#include <chrono>
#include <cpr/cpr.h>
#include <cstdint>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

#include "llm_function_tools.hpp"

const Config &config = Config::instance();

inline std::string get_permission_chs(const std::string_view perm) {
    if (perm == "OWNER") {
        return "群主";
    } else if (perm == "ADMINISTRATOR") {
        return "管理员";
    }
    return "普通群友";
}

std::string gen_common_prompt(const bot_adapter::Profile &bot_profile, const bot_adapter::BotAdapter &adapter,
                              const bot_adapter::Sender &sender, bool is_deep_think) {
    const std::string &custom_prompt = (is_deep_think && config.custom_deep_think_system_prompt_option.has_value())
                                           ? *config.custom_deep_think_system_prompt_option
                                           : config.custom_system_prompt;
    if (const auto &group_sender = bot_adapter::try_group_sender(sender); group_sender.has_value()) {
        std::string permission = get_permission_chs(group_sender->get().permission);
        std::string bot_perm =
            get_permission_chs(adapter.get_group(group_sender->get().group.id).group_info.bot_in_group_permission);

        return fmt::format(
            "你是一个'{}'群里的{},你的名字叫{}(qq号{}),性别是:"
            "{}。{}。当前时间{}，当前跟你聊天的群友的名字叫\"{}\"(qq号{}),身份是{}。你只需要输出与该群友的聊天内容",
            group_sender->get().group.name, bot_perm, bot_profile.name, bot_profile.id,
            bot_adapter::to_chs_string(bot_profile.sex), custom_prompt, get_current_time_formatted(), sender.name,
            sender.id, permission);
    } else {
        return fmt::format("你的名字叫{}(qq号{}),性别是:"
                           "{}。{}。当前时间{}，当前跟你聊天的好友的名字叫\"{}\"(qq号{})。你只输出与该好友聊天的内容",
                           bot_profile.name, bot_profile.id, bot_adapter::to_chs_string(bot_profile.sex), custom_prompt,
                           get_current_time_formatted(), sender.name, sender.id);
    }
}

struct LLMResponse {
    std::optional<ChatMessage> chat_message_opt;
    std::optional<std::vector<ToolCall>> function_calls_opt;
};

std::optional<ChatMessage> get_llm_response(const nlohmann::json &msg_json, bool is_deep_think = false,
                                            const nlohmann::json &tools = DEFAULT_TOOLS) {
    nlohmann::json body = {{"model", config.llm_model_name},
                           {"messages", msg_json},
                           {"stream", false},
                           {"tools", tools},
                           {"is_deep_think", is_deep_think},
                           {"temperature", is_deep_think ? 0.0 : 1.3}};
    const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    spdlog::info("llm body: {}", json_str);
    cpr::Response response =
        cpr::Post(cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, LLM_API_SUFFIX)},
                  cpr::Body{json_str}, cpr::Header{{"Content-Type", "application/json"}});
    spdlog::info("Error msg: {}, status code: {}", response.error.message, response.status_code);

    try {
        spdlog::info("LLM response: {}", response.text);
        auto json = nlohmann::json::parse(response.text);
        auto &message = json["choices"][0]["message"];
        std::string result = std::string(ltrim(message["content"].get<std::string_view>()));
        std::string role = get_optional(message, "role").value_or(ROLE_ASSISTANT);
        remove_text_between_markers(result, "<think>", "</think>");
        ChatMessage ret{role, result};
        if (const auto &tool_calls = get_optional(message, "tool_calls"); tool_calls.has_value()) {
            spdlog::info("Tool calls");
            std::vector<ToolCall> function_calls;
            for (const nlohmann::json &tool_call : *tool_calls) {
                if (auto tc = try_get_chat_completeion_from_messag_tool_call(tool_call); tc.has_value())
                    function_calls.emplace_back(*tc);
                else
                    spdlog::error("解析tool call失败, 原始json为: {}", tool_call.dump());
            }
            ret.tool_calls = std::move(function_calls);
        }
        return ret;

    } catch (const std::exception &e) {
        spdlog::error("get_llm_response(): JSON 解析失败, {}", e.what());
    }
    return std::nullopt;
}

std::string query_chat_session_knowledge(const bot_cmd::CommandContext &context,
                                         const std::string_view &msg_content_str) {
    // Search knowledge for this chat message
    spdlog::info("Search knowledge for this chat message");
    auto msg_knowledge_list = rag::query_knowledge(msg_content_str);
    std::string chat_use_knowledge_str;
    {
        auto map = g_chat_session_knowledge_list_map.write();
        auto user_set_iter = map->find(context.event->sender_ptr->id);
        if (user_set_iter == map->cend()) {
            user_set_iter = map->insert(std::make_pair(context.event->sender_ptr->id, std::set<std::string>())).first;
        }
        for (const auto &knowledge : msg_knowledge_list) {
            if (knowledge.content.empty()) {
                continue;
            }
            // if (user_set_iter->second.size() >= MAX_KNOWLEDGE_LENGTH) {
            //     spdlog::info("{}({})的对话session知识数量超过限制, 删除最旧的知识", context.e->sender_ptr->name,
            //                  context.e->sender_ptr->id);
            //     user_set_iter->second.erase(user_set_iter->second.cbegin());
            // }
            if (knowledge.class_name_list.empty()) {
                user_set_iter->second.insert(fmt::format("{}", knowledge.content));

            } else {
                user_set_iter->second.insert(fmt::format(
                    "{}:{}",
                    join_str(std::cbegin(knowledge.class_name_list), std::cend(knowledge.class_name_list), "|"),
                    knowledge.content));
            }
        }
        size_t total_len = 0;
        auto it = user_set_iter->second.rbegin();
        while (it != user_set_iter->second.rend() && total_len < MAX_KNOWLEDGE_LENGTH) {
            total_len += it->length();
            ++it;
        }

        // Remove entries that exceed the limit
        if (it != user_set_iter->second.rend()) {
            spdlog::info("{}({})的对话session知识数量超过限制, 删除'{}'之前的知识内容", context.event->sender_ptr->name,
                         context.event->sender_ptr->id, *it);
            user_set_iter->second.erase(user_set_iter->second.begin(), it.base());
        }

        const auto user_chat_knowledge_list = map->find(context.event->sender_ptr->id);
        if (user_chat_knowledge_list == map->cend() || user_chat_knowledge_list->second.empty()) {
            spdlog::info("未查询到对话关联的知识");
        } else {
            chat_use_knowledge_str.append("相关的知识:\"");
            chat_use_knowledge_str.append(join_str(std::cbegin(user_chat_knowledge_list->second),
                                                   std::cend(user_chat_knowledge_list->second), "."));
            chat_use_knowledge_str.append("\"");
        }
    }

    // Search knowledge for username
    // spdlog::info("Search knowledge for username");
    // auto sender_name_knowledge_list = rag::query_knowledge(context.e->sender_ptr->name, true);
    // std::string sender_name_knowledge_str;
    // if (sender_name_knowledge_list.empty()) {
    //     spdlog::info("未查询到用户关联的知识");
    // } else {
    //     sender_name_knowledge_str = "与你聊天用户相关的信息:\"";
    //     sender_name_knowledge_str.append(join_str(std::cbegin(sender_name_knowledge_list),
    //                                               std::cend(sender_name_knowledge_list), "\n",
    //                                               [](const auto &knowledge) { return knowledge.content; }));
    //     sender_name_knowledge_str.append("\"");
    // }

    return chat_use_knowledge_str;
}

void insert_tool_call_record_async(const std::string &sender_name, qq_id_t sender_id, const nlohmann::json &msg_json,
                                   const std::string &func_name, const std::string &func_arguments,
                                   const std::string &tool_content) {
    std::thread([=] {
        set_thread_name("insert tool call record");
        spdlog::info("Start insert tool call record thread.");
        database::get_global_db_connection().insert_tool_calls_record(
            sender_name, sender_id, msg_json.dump(), std::chrono::system_clock::now(),
            fmt::format("{}({})", func_name, func_arguments), tool_content);
    }).detach();
}

void on_llm_thread(const bot_cmd::CommandContext &context, const std::string &msg_content_str,
                   const std::optional<std::string> &additional_system_prompt_option) {
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

    auto system_prompt = gen_common_prompt(bot_profile, context.adapter, *context.event->sender_ptr, context.is_deep_think);

    system_prompt += "\n" + query_chat_session_knowledge(context, msg_content_str);

    if (additional_system_prompt_option.has_value()) {
        system_prompt += additional_system_prompt_option.value();
    }

    auto msg_json = get_msg_json(system_prompt, context.event->sender_ptr->id, context.event->sender_ptr->name);

    auto user_chat_msg = ChatMessage(ROLE_USER, msg_content_str);
    add_to_msg_json(msg_json, user_chat_msg);

    std::vector<ChatMessage> one_chat_session;
    if (auto llm_res = get_llm_response(msg_json, context.is_deep_think); llm_res.has_value()) {
        one_chat_session.push_back(std::move(*llm_res));
    } else {
        spdlog::warn("LLM did not response any chat message...");
        context.adapter.send_replay_msg(*context.event->sender_ptr,
                                        bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("?")));
        release_processing_llm(context.event->sender_ptr->id);
        return;
    }

    // process function call
    // loop check if have function call
    while (one_chat_session.rbegin()->tool_calls) {
        spdlog::info("Tool calls");
        const auto &llm_res = *one_chat_session.rbegin();

        // llm request tool call
        add_to_msg_json(msg_json, llm_res);
        std::vector<ChatMessage> append_tool_calls;
        for (const auto &func_calls : *llm_res.tool_calls) {
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
                const auto knowledge_list = rag::query_knowledge(*query);
                for (const auto &knowledge : knowledge_list)
                    content += fmt::format("{}:{}\n",
                                           join_str(std::cbegin(knowledge.class_name_list),
                                                    std::cend(knowledge.class_name_list), "|"),
                                           knowledge.content) +
                               ".";

                const auto net_search_list = rag::net_search_content(
                    include_date ? fmt::format("{} {}", get_current_time_formatted(), *query) : *query);
                std::vector<bot_adapter::ForwardMessageNode> first_replay;
                first_replay.emplace_back(context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                                          context.adapter.get_bot_profile().name,
                                          bot_adapter::make_message_chain_list(
                                              bot_adapter::PlainTextMessage(fmt::format("搜索: \"{}\"", *query))));
                for (const auto &net_search : net_search_list) {
                    content += fmt::format("{}( {} ):{}\n", net_search.title, net_search.url, net_search.content);
                    first_replay.emplace_back(context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                                              context.adapter.get_bot_profile().name,
                                              bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
                                                  fmt::format("关联度: {:.2f}%\n{}( {} )", net_search.score * 100.0f,
                                                              net_search.title, net_search.url))));
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
                    context.adapter.send_long_plain_text_replay(*context.event->sender_ptr,
                                                                "你发的啥,我看不到...再发一遍呢?", true);
                } else {
                    context.adapter.send_long_plain_text_replay(*context.event->sender_ptr, "等我看看这个链接哦...", true);
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
                        ROLE_TOOL, "抱歉，所有链接都获取失败了,可能是网络抽风了或者网站有反爬机制导致紫幻获取不到内容",
                        func_calls.id);
                } else {
                    tool_call_msg = ChatMessage(ROLE_TOOL, content, func_calls.id);
                }
            } else {
                spdlog::error("Function {} is not impl.", func_calls.name);
                tool_call_msg = ChatMessage(ROLE_TOOL,
                                            "主人还没有实现这个功能,快去github页面( "
                                            "https://github.com/FredYakumo/zihuan-aibot-800a )提issues吧",
                                            func_calls.id);
            }

            if (tool_call_msg.has_value()) {
                insert_tool_call_record_async(context.event->sender_ptr->name, context.event->sender_ptr->id, msg_json,
                                              func_calls.name, func_calls.arguments, tool_call_msg->content);
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

        if (auto llm_res = get_llm_response(msg_json, context.is_deep_think); llm_res.has_value()) {
            one_chat_session.push_back(std::move(*llm_res));
        } else {
            spdlog::warn("LLM did not response any chat message...");
            context.adapter.send_replay_msg(*context.event->sender_ptr,
                                            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("?")));
            release_processing_llm(context.event->sender_ptr->id);
            return;
        }
    }

    // Add msg to global storage
    auto session_map = g_chat_session_map.write();
    auto &session = session_map->find(context.event->sender_ptr->id)->second;

    session.message_list.push_back(user_chat_msg);
    std::string replay_content = one_chat_session.rbegin()->content;
    session.message_list.insert(std::end(session.message_list), std::make_move_iterator(std::begin(one_chat_session)),
                                std::make_move_iterator(std::end(one_chat_session)));

    spdlog::info("Prepare to send msg response");

    context.adapter.send_long_plain_text_replay(*context.event->sender_ptr, replay_content);
    if (const auto &group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr)) {
        database::get_global_db_connection().insert_message(
            replay_content,
            bot_adapter::GroupSender(config.bot_id, context.adapter.get_bot_profile().name, std::nullopt, "",
                                     std::nullopt, std::chrono::system_clock::now(), group_sender->get().group),
            std::chrono::system_clock::now(), std::set<uint64_t>{context.event->sender_ptr->id});
    } else {
        database::get_global_db_connection().insert_message(
            replay_content, bot_adapter::Sender(config.bot_id, context.adapter.get_bot_profile().name, std::nullopt),
            std::chrono::system_clock::now(), std::set<uint64_t>{context.event->sender_ptr->id});
    }

    size_t total_len = 0;
    auto sess_it = session.message_list.rbegin();
    while (sess_it != session.message_list.rend() && total_len < USER_SESSION_MSG_LIMIT) {
        total_len += sess_it->content.length(); // UTF-8 length
        ++sess_it;
    }

    // Remove messages that exceed the limit
    if (sess_it != session.message_list.rend()) {
        spdlog::info("{}({})的对话长度超过限制, 删除'{}'之前的上下文内容", context.event->sender_ptr->name,
                     context.event->sender_ptr->name, sess_it->content);

        session.message_list.erase(session.message_list.begin(), sess_it.base());
    }

    // if (session.user_msg_count + 1 >= USER_SESSION_MSG_LIMIT) {
    //     session.message_list.pop_front();
    //     session.message_list.pop_front();
    // } else {
    //     ++session.user_msg_count;
    // }

    release_processing_llm(context.event->sender_ptr->id);
}

void process_llm(const bot_cmd::CommandContext &context,
                 const std::optional<std::string> &additional_system_prompt_option) {
    spdlog::info("开始处理LLM信息");

    if (!try_begin_processing_llm(context.event->sender_ptr->id)) {
        spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.event->sender_ptr->id);
        context.adapter.send_replay_msg(
            *context.event->sender_ptr,
            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("我还在思考中...你别急")));
        return;
    }

    spdlog::debug("Event type: {}, Sender json: {}", context.event->get_typename(),
                  context.event->sender_ptr->to_json().dump());

    std::string msg_content_str{};
    if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
        msg_content_str.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
                                           *context.msg_prop.ref_msg_content));
    }
    if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
        msg_content_str.append(
            fmt::format("\"{}\":\"{}\"", context.event->sender_ptr->name, *context.msg_prop.plain_content));
    }
    spdlog::info(msg_content_str);

    auto llm_thread = std::thread([context, msg_content_str, additional_system_prompt_option] {
        on_llm_thread(context, msg_content_str, additional_system_prompt_option);
    });

    llm_thread.detach();
}

std::optional<OptimMessageResult> optimize_message_query(const bot_adapter::Profile &bot_profile,
                                                         const std::string_view sender_name, qq_id_t sender_id,
                                                         const MessageProperties &message_props) {
    auto msg_list = get_message_list_from_chat_session(sender_name, sender_id);
    std::string current_message{join_str(std::cbegin(msg_list), std::cend(msg_list), "\n")};
    current_message += sender_name;
    current_message += ": \"";
    if (message_props.ref_msg_content != nullptr && !message_props.ref_msg_content->empty()) {
        current_message += "引用一条消息: " + (*message_props.ref_msg_content);
    }
    if (message_props.plain_content != nullptr && !message_props.plain_content->empty()) {
        current_message += "\n" + (*message_props.plain_content);
    }
    current_message += "\"\n";

    nlohmann::json msg_json;
    msg_json.push_back(
        {{"role", "system"},
         {"content", fmt::format(
                         R"(请执行下列任务
1. 分析用户提供的聊天记录（格式为 \"用户名\": \"内容\", \"用户名\": \"内容\"），按顺序排列，并整合整个对话历史的相关信息，但须以最下方（最新消息）为核心。
2. 用户信息如下：
- “你”的对象：名字“{}”，QQ号“{}”；
- “我”（用户）：名字“{}”，QQ号“{}”。
3. 将最新一条聊天内容转换为搜索查询，其中：
- 查询字符串需包含最新消息中需查询的信息，并整合整个对话历史中的相关细节；
- 如查询信息涉及时效性，例如新闻，版本号，训练数据中未出现过的库或者技术，设置queryDate的值为接进1.0，时效性越强越接近1.0，否则0.0。
4. 以最新消息为核心，分析总结现在用户对话中的意图并记录于 JSON 结果中的 \"summary\" 字段。例如：当用户输入 “一脚踢飞你” 时，由于上下文已知对象“紫幻”，则应转换为sumarry: "紫幻被一脚踢飞"；输入“掀裙子时”，则应转换为summary: "紫幻被掀裙子"
5. 分析聊天中你所缺乏的数据和信息，如何缺乏数据或者信息，存入\"fetchData\": [{{\"function\": \"获取信息的方式\", \"query\": \"查询字符串\"}}]，支持的获取信息的\"function\"如下
- 查询用户头像（查询字符串须为 QQ 号）
- 查询用户聊天记录（查询字符串须为 QQ 号）
- 查询用户资料（查询字符串须为 QQ 号）
- 联网搜索（主要是时效新闻，技术相关信息等，只有模型知识不覆盖才需要搜索）
- 查询配置信息（模型信息，运行硬件信息等，查询字符串为查询内容关键字）

\"fetchData\"中\"function\"必须等于功能名字的字符串，如果当前聊天不缺少任何信息，则\"fetchData\"为空列表[]。
6. 返回结果必须为一个 JSON 对象，格式如下：
{{
\"summary\": \"总结用户意图\",
\"queryDate\": 时效指数0.0-1.0,
\"fetchData\": [
{{
\"function\": \"获取信息的方式\",
\"query\": \"查询字符串\"
}},
...
]
}}
)",
                         bot_profile.name, bot_profile.id, sender_name, sender_id, get_today_date_str())}});

    msg_json.push_back({{"role", "user"}, {"content", current_message}});

    nlohmann::json body = {
        {"model", config.llm_model_name}, {"messages", msg_json}, {"stream", false}, {"temperature", 0.0}};
    const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    spdlog::info("llm body: {}", json_str);
    cpr::Response response =
        cpr::Post(cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, LLM_API_SUFFIX)},
                  cpr::Body{json_str}, cpr::Header{{"Content-Type", "application/json"}});

    try {
        spdlog::info(response.text);
        auto json = nlohmann::json::parse(response.text);
        std::string result = std::string(ltrim(json["choices"][0]["message"]["content"].get<std::string_view>()));
        remove_text_between_markers(result, "<think>", "</think>");
        nlohmann::json json_result = nlohmann::json::parse(result);
        auto summary = get_optional(json_result, "summary");
        auto query_date = get_optional(json_result, "queryDate");
        std::optional<nlohmann::json> fetch_data = get_optional(json_result, "fetchData");
        std::vector<FetchData> fetch_data_list;
        if (fetch_data.has_value()) {
            for (nlohmann::json &e : *fetch_data) {
                auto function = get_optional(e, "function");
                if (!function.has_value()) {
                    spdlog::warn(
                        "OptimMessageResult fetchData Node 解析失败, 没有function，跳过该fetchData。原始json为: {}",
                        e.dump());
                    continue;
                }
                auto query = get_optional(e, "query");
                if (!function.has_value()) {
                    spdlog::warn(
                        "OptimMessageResult fetchData Node 解析失败, 没有query，跳过该fetchData。原始json为: {}",
                        e.dump());
                    continue;
                }
                fetch_data_list.emplace_back(*function, *query);
            }
        }

        return OptimMessageResult(std::move(*summary), *query_date, std::move(fetch_data_list));
    } catch (const std::exception &e) {
        spdlog::error("JSON 解析失败: {}", e.what());
    }
    return std::nullopt;
}