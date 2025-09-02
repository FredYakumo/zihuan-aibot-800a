#include "agent/llm.h"
#include "agent/simple_chat_action_agent.h"
#include <chrono>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <cpr/cpr.h>
#include <general-wheel-cpp/markdown_utils.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <nlohmann/json.hpp>

#include "adapter_message.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include "chat_session.hpp"
#include "config.h"
#include "constant_types.hpp"
#include "constants.hpp"
#include "database.h"
#include "event.h"
#include "get_optional.hpp"
#include "global_data.h"
#include "neural_network/model_set.h"
#include "rag.h"
#include "user_protait.h"
#include "utils.h"
#include "vec_db/weaviate.h"

const Config &config = Config::instance();

using namespace wheel;

inline std::string get_permission_chs(const std::string_view perm) {
    if (perm == "OWNER") {
        return "群主";
    } else if (perm == "ADMINISTRATOR") {
        return "管理员";
    }
    return "普通群友";
}



namespace agent {
    using std::string_view;

    std::string gen_common_prompt(const bot_adapter::Profile &bot_profile, const bot_adapter::BotAdapter &adapter,
                                  const bot_adapter::Sender &sender, bool is_deep_think,
                                  string_view action_description,
                                  const std::optional<std::string> additional_system_prompt_option) {
        const std::optional<std::string> &custom_prompt_option =
            (is_deep_think && config.custom_deep_think_system_prompt_option.has_value())
                ? config.custom_deep_think_system_prompt_option
                : config.custom_system_prompt_option;
        std::string ret;
        if (const auto &group_sender = bot_adapter::try_group_sender(sender); group_sender.has_value()) {
            std::string permission = get_permission_chs(group_sender->get().permission);
            std::string bot_perm =
                get_permission_chs(adapter.get_group(group_sender->get().group.id).group_info.bot_in_group_permission);
            if (custom_prompt_option.has_value()) {
                ret = fmt::format("{}你是一个'{}'群里的{},你的名字是'{}'(qq号{}),性别是:{}"
                                  "你正在群里聊天,你需要{},如: 你好.",
                                  custom_prompt_option.value(), group_sender->get().group.name, bot_perm,
                                  bot_profile.name, bot_profile.id, bot_adapter::to_chs_string(bot_profile.sex),
                                  action_description);
            } else {
                ret = fmt::format("你是一个'{}'群里的{},你的名字是'{}'(qq号{}),性别是:{}"
                                  "你正在群里聊天,你需要{},如: 你好",
                                  group_sender->get().group.name, bot_perm, bot_profile.name, bot_profile.id,
                                  bot_adapter::to_chs_string(bot_profile.sex), action_description);
            }
        } else {
            if (custom_prompt_option.has_value()) {
                ret = fmt::format("{}你是'{}'(qq号{}),性别是:{}"
                                  "你正在与好友聊天,你需要{},如: 你好",
                                  custom_prompt_option.value(), bot_profile.name, bot_profile.id,
                                  bot_adapter::to_chs_string(bot_profile.sex), action_description);
            } else {
                ret = fmt::format("你是'{}'(qq号{}),性别是:{}"
                                  "你正在与好友聊天,你需要{},如: 你好",
                                  bot_profile.name, bot_profile.id, bot_adapter::to_chs_string(bot_profile.sex), action_description);
            }
        }
        if (additional_system_prompt_option.has_value()) {
            ret += additional_system_prompt_option.value();
        }
        return ret;
    }

    struct LLMResponse {
        std::optional<ChatMessage> chat_message_opt;
        std::optional<std::vector<ToolCall>> function_calls_opt;
    };

    /**
     * @brief Converts a system prompt and a message list into a JSON format.
     *
     * This function converts a system prompt and a message list into a JSON array.
     * The first element of the array is the system prompt, and the subsequent elements
     * are the messages from the message list. Each message contains two fields: "role" and "content".
     *
     * @param msg_list The message list, of type std::deque<ChatMessage>.
     * @param system_prompt_option The system prompt content, of type std::optional<std::string_view>.
     * @return nlohmann::json Returns a JSON array containing the system prompt and the message list.
     */
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

    inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, ChatMessage msg) {
        msg_json.push_back(msg.to_json());
        return msg_json;
    }

    inline nlohmann::json get_msg_json(const qq_id_t id, std::string name,
                                       const std::optional<std::string> &system_prompt_option = std::nullopt) {
        auto session = g_chat_session_map.get_or_create_value(
            id, [name = std::move(name)]() mutable { return ChatSession(std::move(name)); });
        return msg_list_to_json(session->message_list, system_prompt_option);
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
- \"你\"的对象：名字"{}"，QQ号"{}"；
- \"我\"（用户）：名字"{}"，QQ号"{}"。
3. 将最新一条聊天内容转换为搜索查询，其中：
- 查询字符串需包含最新消息中需查询的信息，并整合整个对话历史中的相关细节；
- 如查询信息涉及时效性，例如新闻，版本号，训练数据中未出现过的库或者技术，设置queryDate的值为接进1.0，时效性越强越接近1.0，否则0.0。
4. 以最新消息为核心，分析总结现在用户对话中的意图并记录于 JSON 结果中的 \"summary\" 字段。例如：当用户输入 "一脚踢飞你" 时，由于上下文已知对象"紫幻"，则应转换为sumarry: "紫幻被一脚踢飞"；输入"掀裙子时"，则应转换为summary: "紫幻被掀裙子"
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
        cpr::Header headers{{"Content-Type", "application/json"}};
        if (!config.llm_api_key.empty()) {
            headers["Authorization"] = fmt::format("{}", config.llm_api_key);
        }
        cpr::Response response =
            cpr::Post(cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, LLM_API_SUFFIX)},
                      cpr::Body{json_str}, headers);

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
                        spdlog::warn("OptimMessageResult fetchData Node 解析失败, "
                                     "没有function，跳过该fetchData。原始json为: {}",
                                     e.dump());
                        continue;
                    }
                    auto query = get_optional(e, "query");
                    if (!function.has_value()) {
                        spdlog::warn("OptimMessageResult fetchData Node 解析失败, "
                                     "没有query，跳过该fetchData。原始json为: {}",
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

    std::optional<UserProtait>
    generate_user_protait(const bot_adapter::BotAdapter &adapter, const bot_adapter::Profile &profile,
                          const std::vector<bot_adapter::MessageChainPtrList> message_chain_list) {
        // 拼接用户历史消息
        std::string history;
        for (const auto &chain : message_chain_list) {
            std::string text = bot_adapter::get_text_from_message_chain(chain);
            if (!text.empty()) {
                history += text + "\n";
            }
        }

        // 获取bot_profile
        const auto &bot_profile = adapter.get_bot_profile();

        // 构造system prompt，包含bot_profile全部字段
        std::string system_prompt = fmt::format(
            R"(请根据以下信息生成一段简明的用户画像，作为你对该用户的印象信息。你将根据这段信息来理解和判断当前正在与你聊天的用户。请生成你对用户的喜好程度,并尽量提炼用户的性格、兴趣、行为习惯等特征，内容要简洁明了。
你的信息：
- 名字：{}  
- QQ号：{}  
- 性别：{}  
- 年龄：{}  
- 等级：{}  
- 邮箱：{}  
用户信息：
- 名字：{}  
- QQ号：{}  
- 性别：{}  
- 年龄：{}  
- 等级：{}  
- 邮箱：{}  
历史消息：
{}
你的输出格式为JSON: {{
"favorability": 你对这个用户的喜好程度(0.0-1.0的实数),
"portrait": "用户画像内容"
}}
)",
            bot_profile.name, bot_profile.id, bot_adapter::to_chs_string(bot_profile.sex), bot_profile.age,
            bot_profile.level, bot_profile.email, profile.name, profile.id, bot_adapter::to_chs_string(profile.sex),
            profile.age, profile.level, profile.email, history);

        nlohmann::json msg_json;
        msg_json.push_back({{"role", "system"}, {"content", system_prompt}});

        nlohmann::json body = {
            {"model", config.llm_model_name}, {"messages", msg_json}, {"stream", false}, {"temperature", 0.0}};
        const auto json_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        spdlog::info("llm body: {}", json_str);
        cpr::Header headers{{"Content-Type", "application/json"}};
        if (!config.llm_api_key.empty()) {
            headers["Authorization"] = fmt::format("{}", config.llm_api_key);
        }
        cpr::Response response =
            cpr::Post(cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, LLM_API_SUFFIX)},
                      cpr::Body{json_str}, headers);

        try {
            spdlog::info("生成用户画像: {}", response.text);
            if (response.status_code != 200) {
                spdlog::error("LLM API请求失败，状态码: {}", response.status_code);
                return std::nullopt;
            }
            auto json = nlohmann::json::parse(response.text);
            if (!json.contains("choices") || json["choices"].empty() || !json["choices"][0].contains("message") ||
                !json["choices"][0]["message"].contains("content")) {
                spdlog::error("LLM API返回格式错误: 缺少必要字段");
                return std::nullopt;
            }
            std::string content = json["choices"][0]["message"]["content"].get<std::string>();
            // 移除前后空白字符
            size_t start = content.find_first_not_of(" \t\n\r");
            size_t end = content.find_last_not_of(" \t\n\r");
            if (start == std::string::npos) {
                spdlog::error("LLM返回内容为空");
                return std::nullopt;
            }
            std::string result = content.substr(start, end - start + 1);
            // 验证JSON格式
            nlohmann::json result_json = nlohmann::json::parse(result);
            if (!result_json.contains("favorability") || !result_json.contains("portrait")) {
                spdlog::error("LLM返回JSON格式错误: 缺少favorability或portrait字段");
                return std::nullopt;
            }
            auto favorability_opt = get_optional<double>(result_json, "favorability");
            auto portrait_opt = get_optional<std::string>(result_json, "portrait");
            if (!favorability_opt.has_value() || !portrait_opt.has_value()) {
                spdlog::error("LLM返回JSON字段类型错误: favorability或portrait类型不正确");
                return std::nullopt;
            }
            return UserProtait{*portrait_opt, *favorability_opt, std::chrono::system_clock::now()};
        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("generate_user_protait(): JSON解析失败: {}，响应内容: {}", e.what(), response.text);
        } catch (const std::exception &e) {
            spdlog::error("generate_user_protait(): 发生异常: {}", e.what());
        }
        return std::nullopt;
    }

    std::optional<nlohmann::json> fetch_model_info() {
        spdlog::info("Fetch model info");
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        cpr::Header headers{{"Content-Type", "application/json"}};
        if (!config.llm_api_key.empty()) {
            headers["Authorization"] = fmt::format("{}", config.llm_api_key);
        }
        auto response = cpr::Get(
            cpr::Url{fmt::format("{}:{}/{}", config.llm_api_url, config.llm_api_port, LLM_MODEL_INFO_SUFFIX)}, headers);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (response.status_code != 200) {
            spdlog::error("Fetch model info failed, status code: {}, response: {}", response.status_code,
                          response.text);
            return std::nullopt;
        }
        try {
            nlohmann::json json_result = nlohmann::json::parse(response.text);
            spdlog::info("Fetch model info success, time taken: {} ms, response: {}", duration.count(),
                         json_result.dump(4));
            return json_result;
        } catch (const nlohmann::json::parse_error &e) {
            spdlog::error("Fetch model info JSON parse error: {}, response: {}", e.what(), response.text);
        } catch (const std::exception &e) {
            spdlog::error("Fetch model info exception: {}", e.what());
        }
        return std::nullopt;
    }

} // namespace agent