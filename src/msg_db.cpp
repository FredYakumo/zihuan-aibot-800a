#include "msg_db.h"
#include "nlohmann/json_fwd.hpp"
#include "utils.h"
#include <MiraiCP.hpp>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <string>
#include <string_view>

void insert_group_msg(MiraiCP::QQID group_id, const std::string_view group_name, MiraiCP::QQID sender_id,
                      const std::string_view sender_name, const std::string_view content) {
    MiraiCP::Logger::logger.info("Insert msg to Group_message collection");
    // Convert std::string_view to std::string
    std::string group_name_str(group_name);
    std::string sender_name_str(sender_name);
    std::string content_str(content);
    // Construct the JSON request
    nlohmann::json request = {
        {"objects", nlohmann::json::array({
            {
                {"class", "Group_message"},
                {"properties", {
                    {"sender_id", std::to_string(sender_id)},
                    {"sender_name", sender_name_str},
                    {"send_time", get_current_time_db()},
                    {"group_id", std::to_string(group_id)},
                    {"group_name", group_name_str},
                    {"content", content_str}
                }}
            }
        })}
    };
    MiraiCP::Logger::logger.info(request.dump());
    // Send the HTTP POST request
    cpr::Response response = cpr::Post(cpr::Url{fmt::format("{}/batch/objects", MSG_DB_URL)}, cpr::Body{request.dump()},
                                       cpr::Header{{"Content-Type", "application/json"}});
    // Check the response status
    if (response.status_code == 200) {
        // MiraiCP::Logger::logger.info(fmt::format("Insert msg db res: {}", response.text));
        MiraiCP::Logger::logger.info("Insert msg db successed");

    } else {
        MiraiCP::Logger::logger.error(fmt::format("Failed to insert msg: {}", response.text));
    }
}
