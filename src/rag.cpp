#include "rag.h"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <config.h>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include "config.h"
#include <spdlog/spdlog.h>

namespace rag {
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

        cpr::Response r = cpr::Post(cpr::Url{fmt::format("{}/graphql", MSG_DB_URL)},
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

    std::vector<std::pair<DBKnowledge, double>> query_knowledge(const std::string_view query) {
        std::vector<std::pair<DBKnowledge, double>> result;

        std::string escaped_query = replace_str(query, "\"", "\\\"");

        std::string graphql_query;

        graphql_query = "query {"
                        "  Get {"
                        "    AIBot_knowledge("
                        "      nearText: { concepts: [\"" +
                        escaped_query +
                        "\"], certainty: 0.8 }"
                        "      limit: 5"
                        "    ) {"
                        "      creator_name"
                        "      create_time"
                        "      content"
                        "      _additional { certainty }"
                        "    }"
                        "  }"
                        "}";

        nlohmann::json request_body;
        request_body["query"] = graphql_query;

        cpr::Response r = cpr::Post(cpr::Url{fmt::format("{}/graphql", MSG_DB_URL)},
                                    cpr::Header{{"Content-Type", "application/json"}}, cpr::Body{request_body.dump()});

        if (r.status_code != 200) {

            return result;
        }
        try {
            auto response_json = nlohmann::json::parse(r.text);

            auto &messages = response_json["data"]["Get"]["AIBot_knowledge"];
            for (auto &msg : messages) {
                DBKnowledge db_msg;
                db_msg.creator_name = msg["creator_name"].get<std::string>();
                db_msg.create_dt = msg["create_time"].get<std::string>();
                db_msg.content = msg["content"].get<std::string>();
                // double score = std::stod(msg["_additional"]["certainty"].get<std::string>());
                double score = msg["_additional"]["certainty"].get<float>();
                result.push_back(std::make_pair(std::move(db_msg), score));
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
            cpr::Post(cpr::Url{fmt::format("{}/batch/objects", MSG_DB_URL)}, cpr::Body{request.dump()},
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
            cpr::Post(cpr::Url{fmt::format("{}/batch/objects", MSG_DB_URL)}, cpr::Body{request.dump()},
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

        cpr::Response r = cpr::Post(cpr::Url{NET_SEARCH_API_URL},
                                    cpr::Header{{"Content-Type", "application/json"},
                                                {"Authorization", NET_SEARCH_TOKEN}},

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

    std::string url_search_content(const std::vector<std::string> &url_list) {
        std::string results {"(以下引用了一些网页链接和它的内容，由于这个输入用户看不到，所以请在回答中列出概要或者详细的结果[根据用户的指示]):\n"};
        nlohmann::json request_body{{"urls", url_list}, {"extract_depth", "advanced"}};

        cpr::Response r = cpr::Post(cpr::Url{URL_SEARCH_API_URL},
                                    cpr::Header{{"Content-Type", "application/json"},
                                                {"Authorization", URL_SEARCH_TOKEN}},

                                    cpr::Body{request_body.dump()});

        if (r.status_code != 200) {
            spdlog::error("请求失败: {}", r.text);
        }
        try {

            auto j = nlohmann::json::parse(r.text);

            for (const auto &item : j["results"]) {
                results.append(item.value("url", ""));
                results += ',';
                results.append(item.value("raw_content", ""));
                results += '\n';
            }
        } catch (const nlohmann::json::exception &e) {
            spdlog::error("JSON解析失败: {}", e.what());
        }
        return results;
    }

} // namespace rag