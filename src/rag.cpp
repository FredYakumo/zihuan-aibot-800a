#include "rag.h"
#include "adapter_model.h"
#include "config.h"
#include "constants.hpp"
#include "db_knowledge.hpp"
#include "fmt/core.h"
#include "get_optional.hpp"
#include "msg_prop.h"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <config.h>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>

namespace rag {
    const Config &config = Config::instance();

    std::vector<std::pair<DBGroupMessage, double>> query_group_msg(const std::string_view query,
                                                                   std::optional<uint64_t> group_id_option) {
        std::vector<std::pair<DBGroupMessage, double>> result;

        std::string escaped_query = replace_str(query, "\"", "\\\"");

        std::string graphql_query;
        if (group_id_option) {
            graphql_query = "query {"
                            "  Get {"
                            "    Group_message("
                            "      nearText: { concepts: [\"" +
                            escaped_query +
                            "\"], certainty: 0.75 }"
                            "      where: {"
                            "        path: [\"group_id\"],"
                            "        operator: Equal,"
                            "        valueString: \"" +
                            std::to_string(group_id_option.value()) +
                            "\""
                            "      }"
                            "      sort: [{path: \"send_time\", order: desc}]"
                            "      limit: 5"
                            "    ) {"
                            "      sender_name"
                            "      send_time"
                            "      content"
                            "      _additional { certainty }"
                            "    }"
                            "  }"
                            "}";
        } else {
            graphql_query = "query {"
                            "  Get {"
                            "    Group_message("
                            "      nearText: { concepts: [\"" +
                            escaped_query +
                            "\"], certainty: 0.75 }"
                            "      sort: [{path: \"send_time\", order: desc}]"
                            "      limit: 5"
                            "    ) {"
                            "      sender_name"
                            "      send_time"
                            "      content"
                            "      _additional { certainty }"
                            "    }"
                            "  }"
                            "}";
        }

        nlohmann::json request_body;
        request_body["query"] = graphql_query;

        cpr::Response r = cpr::Post(cpr::Url{fmt::format("{}query_knowledge", config.vec_db_url)},
                                    cpr::Header{{"Content-Type", "application/json"}}, cpr::Body{request_body.dump()});

        if (r.status_code != 200) {
            return result;
        }

        try {
            auto response_json = nlohmann::json::parse(r.text);

            // 提取消息列表
            auto &messages = response_json["data"]["Get"]["Group_message"];
            for (auto &msg : messages) {
                DBGroupMessage db_msg;
                db_msg.sender_name = msg["sender_name"].get<std::string>();
                db_msg.send_time = msg["send_time"].get<std::string>();
                db_msg.content = msg["content"].get<std::string>();
                double certainty = msg["_additional"]["certainty"].get<float>();
                result.push_back(std::make_pair(std::move(db_msg), certainty));
            }
        } catch (const nlohmann::json::exception &e) {
        }

        return result;
    }

    std::vector<DBKnowledge> query_knowledge(const std::string_view query, bool exactly_match) {
        std::vector<DBKnowledge> result;

        nlohmann::json request_body{{"query", query}};

        cpr::Response r = cpr::Post(
            cpr::Url{fmt::format("{}{}", config.vec_db_url, exactly_match ? "find_keyword_match" : "query_knowledge")},
            cpr::Header{{"Content-Type", "application/json"}}, cpr::Body{request_body.dump()});

        if (r.status_code != 200) {
            spdlog::error("查询知识失败: {}", r.text);
            return result;
        }
        try {
            auto knowledge_json = nlohmann::json::parse(r.text);

            for (auto &json : knowledge_json) {
                DBKnowledge knowledge{
                    get_optional<std::string_view>(json, "content").value_or(""),
                    get_optional<std::string_view>(json, "creator_name").value_or(""),
                    get_optional<std::string_view>(json, "create_time").value_or(""),
                    get_optional<std::vector<std::string>>(json, "keyword").value_or(std::vector<std::string>()),
                    get_optional<float>(json, "certainty").value_or(0.0f)};
                spdlog::info("{}, 创建者: {}, 日期: {}, 置信度: {}, 关键字列表: {}", knowledge.content,
                             knowledge.creator_name, knowledge.create_dt, knowledge.certainty,
                             join_str(std::cbegin(knowledge.keywords), std::cend(knowledge.keywords), ","));
                result.push_back(knowledge);
            }
        } catch (const nlohmann::json::exception &e) {
            spdlog::error("查询知识出错: {}", e.what());
        } catch (const std::invalid_argument &e) {
            spdlog::error("查询知识时转换得分出错: {}", e.what());
        } catch (const std::out_of_range &e) {
            spdlog::error("查询知识时转换得分出错，数据超出范围: {}", e.what());
        }

        return result;
    }

    void insert_group_msg(uint64_t group_id, const std::string_view group_name, uint64_t sender_id,
                          const std::string_view sender_name, const std::string_view content) {
        spdlog::info("Insert msg to Group_message collection");
        std::string group_name_str(group_name);
        std::string sender_name_str(sender_name);
        std::string content_str(content);

        nlohmann::json request = {{"objects", nlohmann::json::array({{{"class", "Group_message"},
                                                                      {"properties",
                                                                       {{"sender_id", std::to_string(sender_id)},
                                                                        {"sender_name", sender_name_str},
                                                                        {"send_time", get_current_time_db()},
                                                                        {"group_id", std::to_string(group_id)},
                                                                        {"group_name", group_name_str},
                                                                        {"content", content_str}}}}})}};
        spdlog::info("{}", request.dump());

        cpr::Response response =
            cpr::Post(cpr::Url{fmt::format("{}/batch/objects", config.vec_db_url)}, cpr::Body{request.dump()},
                      cpr::Header{{"Content-Type", "application/json"}});

        if (response.status_code == 200) {
            // MiraiCP::Logger::logger.info(fmt::format("Insert msg db res: {}", response.text));
            spdlog::info("insert msg db successed");
        } else {
            spdlog::error("Failed to insert msg: {}", response.text);
        }
    }

    void insert_knowledge(const DBKnowledge &knowledge) {
        spdlog::info("Insert msg to AIBot_knowledge collection");

        nlohmann::json request = {{"objects", nlohmann::json::array({{{"class", "AIBot_knowledge"},
                                                                      {"properties",
                                                                       {{"creator_name", knowledge.creator_name},
                                                                        {"create_time", knowledge.create_dt},
                                                                        {"content", knowledge.content}}}}})}};
        spdlog::info("{}", request.dump());

        cpr::Response response =
            cpr::Post(cpr::Url{fmt::format("{}/batch/objects", config.vec_db_url)}, cpr::Body{request.dump()},
                      cpr::Header{{"Content-Type", "application/json"}});

        if (response.status_code == 200) {
            // MiraiCP::Logger::logger.info(fmt::format("Insert msg db res: {}", response.text));
            spdlog::info("Insert knowledge db successed");

        } else {
            spdlog::error("Failed to insert knowledge: {}", response.text);
        }
    }

    std::vector<NetSearchResult> net_search_content(const std::string_view query) {
        std::vector<NetSearchResult> results;
        nlohmann::json request_body{{"query", query}, {"include_images", false}, {"include_image_descriptions", false}};

        cpr::Response r =
            cpr::Post(cpr::Url{config.net_search_api_url},
                      cpr::Header{{"Content-Type", "application/json"}, {"Authorization", config.net_search_token}},

                      cpr::Body{request_body.dump()});

        if (r.status_code != 200) {
            spdlog::error("请求失败: {}", r.text);
        }
        try {

            auto j = nlohmann::json::parse(r.text);

            for (const auto &item : j["results"]) {
                NetSearchResult result;
                result.title = item.value("title", "");
                result.url = item.value("url", "");
                result.content = item.value("content", "");
                result.score = item.value("score", 0.0);
                results.push_back(result);
            }
        } catch (const nlohmann::json::exception &e) {
            spdlog::error("JSON解析失败: {}", e.what());
        }
        return results;
    }

    // std::vector<NetSearchImage> net_search_image(const std::string_view query) {

    // }

    std::optional<std::string> url_search_content(const std::vector<std::string> &url_list) {
        std::string results{"(以下引用了一些网页链接和它的内容，由于这个输入用户看不到，所以请在回答中列出概要或者详细"
                            "的结果[根据用户的指示]):\n"};
        nlohmann::json request_body{{"urls", url_list}, {"extract_depth", "advanced"}};

        cpr::Response r =
            cpr::Post(cpr::Url{config.url_search_api_url},
                      cpr::Header{{"Content-Type", "application/json"}, {"Authorization", config.url_search_token}},

                      cpr::Body{request_body.dump()});

        if (r.status_code != 200) {
            spdlog::error("请求失败: {}", r.text);
            return std::nullopt;
        }
        try {

            auto j = nlohmann::json::parse(r.text);
            auto res = j["results"];
            if (res.empty()) {
                return std::nullopt;
            }

            for (const auto &item : res) {
                results.append(item.value("url", ""));
                results += ',';
                results.append(item.value("raw_content", ""));
                results += '\n';
            }
        } catch (const nlohmann::json::exception &e) {
            spdlog::error("JSON解析失败: {}", e.what());
            return std::nullopt;
        }
        return results;
    }

    std::vector<std::string> get_message_list_from_chat_session(const std::string_view sender_name, qq_id_t sender_id) {
        std::vector<std::string> ret;

        return ret;
    }

    std::optional<OptimMessageResult> optimize_message_query(const bot_adapter::Profile &bot_profile,
                                              const std::string_view sender_name, qq_id_t sender_id,
                                              const MessageProperties &message_props) {
        auto msg_list = get_message_list_from_chat_session(sender_name, sender_id);
        std::string current_message {join_str(std::cbegin(msg_list), std::cend(msg_list), "\n")};
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
                            1. 分析用户提供的聊天记录（格式为 \"用户1\": \"内容\", \"用户2\": \"内容\"），按顺序排列，并整合整个对话历史的相关信息，但须以最下方（最新消息）为核心。\n
                            2. 用户信息如下：
                                - “你”的对象：名字“{}”，QQ号“{}”；
                                - “我”（用户）：名字“{}”，QQ号“{}”。
                            3. 将最新一条聊天内容转换为搜索查询，其中：
                                - 查询字符串需包含最新消息中需查询的信息，并整合整个对话历史中的相关细节；
                                - 如查询信息涉及时效性，例如新闻，版本号，训练数据中未出现过的库或者技术，设置queryDate的值为接进1.0，时效性越强越接近1.0，否则0.0。
                            4. 分析聊天记录中所涉及的功能，并记录于 JSON 结果中的 \"function\" 字段。支持的功能包括：
                                - 聊天\n
                                - 查询用户头像（查询字符串须为 QQ 号）
                                - 查询用户聊天记录（查询字符串须为 QQ 号）
                                - 查询用户资料（查询字符串须为 QQ 号）
                                - 查询知识库（查询字符串由用户最新信息和整体对话上下文整合形成。例如：当用户输入 “一脚踢飞” 时，由于上下文已知对象“紫幻”，则应转换为查询 “紫幻被一脚踢飞”；输入“掀裙子时”，则应转换查询为“紫幻被掀裙子”）

                            如聊天记录涉及其他功能，则将 \"function\" 字段设为 \"非法\"。
                            5. 如果对话历史中涉及多个功能或查询方向，则只返回最新一条消息对应的功能和查询字符串（注意：查询字符串应整合整体对话历史中的相关信息）。
                            6. 返回结果必须为一个 JSON 对象，格式如下：
                            {{
                            \"function\": \"功能\",
                            \"queryDate\": 时效指数0.0-1.0,
                            \"query\": \"查询字符串\"
                            }}
                            )",
                             bot_profile.name, bot_profile.id, sender_name, sender_id, get_today_date_str())}});
        
        msg_json.push_back({{"role", "user"}, {"content", current_message}});

        nlohmann::json body = {{"model", config.llm_model_name},
                               {"messages", msg_json},
                               {"stream", false},
                               {"temperature", 0.0}};
        const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        spdlog::info("llm body: {}", json_str);
        cpr::Response response =
            cpr::Post(cpr::Url{config.llm_api_url}, cpr::Body{json_str},
                      cpr::Header{{"Content-Type", "application/json"}, {"Authorization", config.llm_api_token}});

        try {
            spdlog::info(response.text);
            auto json = nlohmann::json::parse(response.text);
            std::string result = std::string(ltrim(json["choices"][0]["message"]["content"].get<std::string_view>()));
            remove_text_between_markers(result, "<think>", "</think>");
            nlohmann::json json_result = nlohmann::json::parse(result);
            auto function = get_optional(json_result, "function");
            auto query_date = get_optional(json_result, "queryDate");
            auto query_string = get_optional(json_result, "query");

            if (!function.has_value() || !query_date.has_value() || !query_string.has_value()) {
                spdlog::error("OptimMessageResult 解析失败");
            }
            
            return OptimMessageResult(*function, *query_date, *query_string);
        } catch (const std::exception &e) {
            spdlog::error("JSON 解析失败: {}", e.what());
        }
        return std::nullopt;
    }

} // namespace rag