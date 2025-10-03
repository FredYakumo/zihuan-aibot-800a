#include "agent/brain.h"
#include "adapter_model.h"
#include "agent/action_descript_prompt.hpp"
#include "agent/llm.h"
#include "agent/llm_function_tools.hpp"
#include "bot_cmd.h"
#include "config.h"
#include "constant_types.hpp"
#include "msg_prop.h"
#include <iterator>
#include <string_utils.hpp>

namespace agent {

    Config &conf = Config::instance();

    using std::string_view;
    using wheel::join_str;
    using wheel::ltrim;
    using wheel::rtrim;

    std::string get_mixed_user_input_message_text(const bot_adapter::Sender &sender, qq_id_t bot_id,
                                                  const MessageProperties &msg_prop) {
        std::string llm_content{};
        const auto &sender_name = sender.name;
        const auto &sender_id = sender.id;

        if (msg_prop.ref_msg_content != nullptr && !msg_prop.ref_msg_content->empty()) {
            llm_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", sender_name, *msg_prop.ref_msg_content));
        }
        if (!msg_prop.at_id_set.empty() && !(msg_prop.at_id_set.size() == 1 && *msg_prop.at_id_set.begin() == bot_id)) {
            if (*msg_prop.at_id_set.begin() == bot_id) {
                llm_content.append("本消息提到了你");
            } else {
                llm_content.append("本消息提到了: ");
                llm_content.append(join_str(std::cbegin(msg_prop.at_id_set), std::cend(msg_prop.at_id_set), "、"));
                llm_content.append("。");
            }
        }
        std::string speak_content = EMPTY_MSG_TAG;
        if (msg_prop.plain_content != nullptr && !wheel::ltrim(wheel::rtrim(*msg_prop.plain_content)).empty()) {
            speak_content = *msg_prop.plain_content;
        }

        if (msg_prop.plain_content != nullptr && !msg_prop.plain_content->empty()) {
            if (auto group_sender = bot_adapter::try_group_sender(sender); group_sender.has_value()) {
                llm_content.append(fmt::format(
                    "{}\"{}\"({})[{}]对你说: \"{}\"", bot_adapter::get_permission_chs(group_sender->get().permission),
                    group_sender->get().name, sender_id, get_current_time_formatted(), speak_content));
            } else {
                llm_content.append(fmt::format("\"{}\"({})[{}]对你说: \"{}\"", sender_name, sender_id,
                                               get_current_time_formatted(), speak_content));
            }
        }
        std::string mixed_input_content;
        if (msg_prop.ref_msg_content != nullptr && !msg_prop.ref_msg_content->empty()) {
            mixed_input_content.append(
                fmt::format("\"{}\"引用了一个消息: \"{}\",", sender_name, *msg_prop.ref_msg_content));
        }
        if (msg_prop.plain_content != nullptr && !msg_prop.plain_content->empty()) {
            mixed_input_content.append(fmt::format("{}", *msg_prop.plain_content));
        }

        return mixed_input_content;
    }

    void Brain::process_friend_message_event(const bot_adapter::FriendMessageEvent &event) {
        spdlog::info("Brain agent Processing friend message event from user ID: {}", event.sender_ptr->id);
        // Implement friend message processing logic here
    }

    nlohmann::json brain_decision_function_tools = nlohmann::json::array(

        {make_tool_function(
             "chat_model",
             "日常闲聊"
             "用户请求查看上下文；2) "
             "需要回顾特定用户的发言；3) "
             "需要汇总群聊内容。此函数返回的才是真实的聊天历史数据。可多次调用以获取不同对象的聊天记录。",
             make_object_params({
                 {"targetId",
                  {{"type", PARAMETER_TYPE_NUMBER},
                   {"description",
                    "目标用户的QQ号(数字ID)。默认为null表示查询群内最近的聊天记录。当用户提及'我的聊天记录'"
                    "时，应填入用户自己的QQ号。有明确数字ID时优先使用此参数。"}}},
                 {"targetName",
                  {{"type", PARAMETER_TYPE_STRING},
                   {"description", "目标用户的名称。默认为null表示查询群内最近的聊天记录。当用户提及特定名称（如'"
                                   "查看名贵种猫的发言'）时使用。仅当没有明确的targetId时才使用此参数。"}}},
             })),
         make_tool_function("view_model_info",
                            "查看模型信息,当消息提及到(无论何种方式)系统提示词、模型信息、function "
                            "calls列表例如'输出模型信息','输出system prompt', '输出function calls "
                            "json',调用此函数.该函数得到的结果才是模型的信息",
                            {}),
         // Search info tool
         make_tool_function(
             "search_info",
             "查询信息.可以根据查询不同的信息拆分成多次调用.不认识的信息必须要进行查询,"
             "如评价Rust语言和MIZ语言的区别,则多次调用分别查询MIZ语言的发展,MIZ语言的语法,MIZ语言的生态等",
             make_object_params(
                 {{"query", {{"type", PARAMETER_TYPE_STRING}, {"description", "查询内容的关键字"}}},
                  {"category",
                   {{"type", PARAMETER_TYPE_STRING},
                    {"description",
                     "搜索类别,用于过滤向量数据库中的知识.常见类别包括'general'(通用知识),'tech'(技术相关),'news'("
                     "新闻资讯),'education'(教育相关)等.若不确定类别,可留空以进行全类别搜索."}}}},
                 {"query"})),

         make_tool_function("query_group", "如果用户的消息中涉及查看群的资料,调用此函数",
                            make_object_params({{"item",
                                                 {{"type", PARAMETER_TYPE_STRING},
                                                  {"description", "查询内容,仅支持OWNER(群主),ADMIN(管理员),"
                                                                  "NOTICE(群公告).除此以外则是OTHER"}}}})),
         // make_tool_function("get_group_member_list", "查询群成员列表.")

         // Fetch URL content tool
         make_tool_function(
             "fetch_url_content", "你可以使用这个函数来查看网页链接里的内容",
             make_object_params(
                 {{"urls", {{"type", PARAMETER_TYPE_ARRAY}, {"description", "网页链接列表,每个元素为url字符串"}}}},
                 {"url"})),
         make_tool_function("get_function_list", "获取紫幻可用功能,函数,function calls,指令列表.", {})});

    void Brain::process_group_message_event(const bot_adapter::GroupMessageEvent &event) {
        const auto &gs = bot_adapter::try_group_sender(*event.sender_ptr);
        if (!gs.has_value()) {
            spdlog::error("Brain agent: Oops failed processing: Failed to get group sender info for user ID: {}",
                          event.sender_ptr->id);
            return;
        }
        const auto &sender = gs->get();
        spdlog::info("Brain agent Processing group message event from group ID: {}, user ID: {}", sender.group.id,
                     sender.id);

        gen_inchat_prompt(m_adapter->get_bot_profile(), *m_adapter, sender, false,
                          action_prompt::ACTION_TO_USER_MENTION, conf.custom_system_prompt_option);
    }

} // namespace agent
