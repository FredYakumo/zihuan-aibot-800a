#include "msg_db.h"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <MiraiCP.hpp>
#include <config.h>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

std::string escape_query(const std::string_view input) {
    std::string result(input);
    size_t pos = 0;
    while ((pos = result.find('"', pos)) != std::string::npos) {
        result.replace(pos, 1, "\\\"");
        pos += 2; // 跳过已转义的部分
    }
    return result;
}

std::vector<std::pair<DBGroupMessage, double>> query_group_msg(const std::string_view query,
                                                               std::optional<MiraiCP::QQID> group_id_option) {
    std::vector<std::pair<DBGroupMessage, double>> result;

    std::string escaped_query = escape_query(query);

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

    std::string escaped_query = escape_query(query);

    std::string graphql_query;

    graphql_query = "query {"
                    "  Get {"
                    "    AIBot_knowledge("
                    "      nearText: { concepts: [\"" + escaped_query +
                    "\"], certainty: 0.75 }"
                    "      sort: [{path: \"create_time\", order: desc}]"
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
            double certainty = msg["_additional"]["certainty"].get<float>();
            result.push_back(std::make_pair(std::move(db_msg), certainty));
        }
    } catch (const nlohmann::json::exception &e) {

    }

    return result;
}

void insert_group_msg(MiraiCP::QQID group_id, const std::string_view group_name, MiraiCP::QQID sender_id,
                      const std::string_view sender_name, const std::string_view content) {
    MiraiCP::Logger::logger.info("Insert msg to Group_message collection");

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
    MiraiCP::Logger::logger.info(request.dump());

    cpr::Response response = cpr::Post(cpr::Url{fmt::format("{}/batch/objects", MSG_DB_URL)}, cpr::Body{request.dump()},
                                       cpr::Header{{"Content-Type", "application/json"}});

    if (response.status_code == 200) {
        // MiraiCP::Logger::logger.info(fmt::format("Insert msg db res: {}", response.text));
        MiraiCP::Logger::logger.info("Insert msg db successed");

    } else {
        MiraiCP::Logger::logger.error(fmt::format("Failed to insert msg: {}", response.text));
    }
}

void insert_knowledge(const DBKnowledge &knowledge) {
    MiraiCP::Logger::logger.info("Insert msg to AIBot_knowledge collection");

    nlohmann::json request = {{"objects", nlohmann::json::array({{{"class", "AIBot_knowledge"},
                                                                  {"properties",
                                                                   {{"creator_name", knowledge.creator_name},
                                                                    {"create_time", knowledge.create_dt},
                                                                    {"content", knowledge.content}}}}})}};
    MiraiCP::Logger::logger.info(request.dump());

    cpr::Response response = cpr::Post(cpr::Url{fmt::format("{}/batch/objects", MSG_DB_URL)}, cpr::Body{request.dump()},
                                       cpr::Header{{"Content-Type", "application/json"}});

    if (response.status_code == 200) {
        // MiraiCP::Logger::logger.info(fmt::format("Insert msg db res: {}", response.text));
        MiraiCP::Logger::logger.info("Insert knowledge db successed");

    } else {
        MiraiCP::Logger::logger.error(fmt::format("Failed to insert knowledge: {}", response.text));
    }
}
