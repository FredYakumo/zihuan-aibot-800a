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
#include "vec_db/models.h"
#include "vec_db/weaviate.h"
#include <config.h>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <iterator>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <general-wheel-cpp/string_utils.hpp>

namespace rag {
    const Config &config = Config::instance();

    using namespace wheel;
    using vec_db::DBGroupMessage;

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
        return vec_db::query_knowledge_from_vec_db(query, 0.7f, 5);
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
                                                                        {"key", knowledge.key},
                                                                        {"value", knowledge.value}}}}})}};
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
        nlohmann::json request_body{{"query", query}};

        cpr::Response r = cpr::Post(
            cpr::Url{fmt::format("{}:{}/{}", config.search_api_url, config.search_api_port, SEARCH_WEB_SUFFIX)},
            cpr::Header{{"Content-Type", "application/json"}},

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

    UrlSearchResult url_search_content(const std::vector<std::string> &url_list) {
        nlohmann::json request_body{{"urls", url_list}};

        cpr::Response r = cpr::Post(
            cpr::Url{fmt::format("{}:{}/{}", config.search_api_url, config.search_api_port, SEARCH_URL_SUFFIX)},
            cpr::Header{{"Content-Type", "application/json"}},

            cpr::Body{request_body.dump()});

        UrlSearchResult ret;
        if (r.status_code != 200) {
            spdlog::error("请求失败: {}", r.text);
            return ret;
        }

        try {
            auto j = nlohmann::json::parse(r.text);
            auto res = j["results"];

            for (const auto &item : res) {
                ret.results.push_back(std::make_pair(item.value("url", EMPTY_JSON_STR_VALUE),
                                                     item.value("raw_content", EMPTY_JSON_STR_VALUE)));
            }
            auto failed = j["failed_results"];
            for (const auto &item : failed) {
                ret.failed_reason.push_back(
                    std::make_pair(item.value("url", EMPTY_JSON_STR_VALUE), item.value("error", EMPTY_JSON_STR_VALUE)));
            }

            if (ret.results.empty()) {
                spdlog::error("Url search content failed: for [{}]",
                              join_str(std::cbegin(url_list), std::cend(url_list)));
            } else {
                spdlog::info("Url search {} succesed, {} failed.", ret.results.size(), ret.failed_reason.size());
            }

        } catch (const nlohmann::json::exception &e) {
            spdlog::error("JSON解析失败: {}", e.what());
        }
        return ret;
    }

} // namespace rag