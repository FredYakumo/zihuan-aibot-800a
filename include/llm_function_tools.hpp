#pragma once
#include <nlohmann/json.hpp>
#include <string>

// Helper functions to create JSON objects
inline nlohmann::json make_tool_function(const std::string &name, const std::string &description,
                                         const nlohmann::json &parameters) {
    return {{"type", "function"},
            {"function", {{"name", name}, {"description", description}, {"parameters", parameters}}}};
}

inline nlohmann::json make_object_params(const std::vector<std::pair<std::string, nlohmann::json>> &properties,
                                         const std::vector<std::string> &required = {}) {
    nlohmann::json props;
    for (const auto &p : properties) {
        props[p.first] = p.second;
    }
    return {{"type", "object"}, {"properties", props}, {"required", required}};
}

// Define the tools
const nlohmann::json DEFAULT_TOOLS = nlohmann::json::array(
    {// Search info tool
     make_tool_function(
         "search_info",
         "查询信息.可以根据查询不同的信息拆分成多次调用.知识库里没出现过的信息必须要进行查询,"
         "如评价Rust语言和MIZ语言的区别,则多次调用分别查询MIZ语言的发展,MIZ语言的语法,MIZ语言的生态等",
         make_object_params({{"query", {{"type", "string"}, {"description", "查询内容的关键字"}}},
                             {"includeDate",
                              {{"type", "boolean"},
                               {"description", "是否包含日期.只有需要查询时效性的信息时才需要为true,如最近发生了什么,"
                                               "微软新出了什么操作系统,小米最近股价,原神的最新版本..."}}}},
                            {"query"})),

     // Interact tool
     make_tool_function(
         "interact", "如果用户的消息中与某个对象互动.如揍'张三'则是与'张三'互动, @1232142,则是@1232142这个QQ号",
         make_object_params(
             {{"target",
               {{"type", "string"}, {"description", "对象.如揍'张三'则为'张三',如果未提及对象,默认是你('紫幻')"}}},
              {"idOption", {{"type", "string"}, {"description", "target的QQ号(可空)"}}},
              {"action",
               {{"type", "string"}, {"description", "互动内容.如揍'张三'则为'揍'或者'打一顿', 如@1232142则为@"}}}})),

     // query user tool
     make_tool_function(
         "query_user", "如果用户的消息中涉及查看某个群友或者用户的资料,调用此函数",
         make_object_params(
             {{"target", {{"type", "string"}, {"description", "对象.只能为QQ号或者名字"}}},
              {"item",
               {{"type", "string"},
                {"description",
                 "查询内容,仅支持AVATAR(头像),SUMMARY(印象或者评价),MESSAGES(最近的消息).除此以外则是OTHER"}}}})),
     // query group tool
     make_tool_function(
         "query_group",
         "如果用户的消息中涉及查看群的信息,调用此函数",
         make_object_params(
             {{"item",
               {{"type", "string"},
                {"description", "查询内容,仅支持OWNER(群主),ADMIN(管理员),MESSAGES(最近的消息).除此以外则是OTHER"}}}})),

     // Fetch URL content tool
     make_tool_function("fetch_url_content", "你可以使用这个函数来查看网页链接里的内容",
                        make_object_params({{"url", {{"type", "string"}, {"description", "网页链接"}}}}, {"url"}))});