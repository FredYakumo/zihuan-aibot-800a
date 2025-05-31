#ifndef LLM_H
#define LLM_H
#include "bot_cmd.h"
#include "global_data.h"

std::string gen_common_prompt(const bot_adapter::Profile &bot_profile, const bot_adapter::BotAdapter &adapter,
                              const bot_adapter::Sender &sender, bool is_deep_think);

void process_llm(const bot_cmd::CommandContext &context,
                 const std::optional<std::string> &additional_system_prompt_option);

inline bool try_begin_processing_llm(uint64_t target_id) {
    std::lock_guard lock(g_chat_processing_map.first);
    if (auto it = g_chat_processing_map.second.find(target_id); it != std::cend(g_chat_processing_map.second)) {
        if (it->second) {
            return false;
        }
        it->second = true;
        return true;
    }
    g_chat_processing_map.second.emplace(target_id, true);
    return true;
}

/**
 * @brief Converts a system prompt and a message list into a JSON format.
 *
 * This function converts a system prompt and a message list into a JSON array.
 * The first element of the array is the system prompt, and the subsequent elements
 * are the messages from the message list. Each message contains two fields: "role" and "content".
 *
 * @param system_prompt The system prompt content, of type std::string_view.
 * @param msg_list The message list, of type std::deque<ChatMessage>.
 * @return nlohmann::json Returns a JSON array containing the system prompt and the message list.
 */
inline nlohmann::json msg_list_to_json(const std::string_view system_prompt, const std::deque<ChatMessage> &msg_list) {
    nlohmann::json msg_json = nlohmann::json::array();
    msg_json.push_back(nlohmann::json{{"role", "system"}, {"content", system_prompt}});
    for (const auto &msg : msg_list) {
        nlohmann::json msg_entry = msg.to_json();
        spdlog::info("msg_entry: {}", msg_entry.dump());
        msg_json.push_back(std::move(msg_entry));
    }
    return msg_json;
}

inline nlohmann::json &add_to_msg_json(nlohmann::json &msg_json, const ChatMessage &msg) {
    // nlohmann::json append_json {{"role", msg.role}, {"content", msg.content}};
    // if (msg.tool_calls) {
    //     append_json["tool_calls"] = *msg.tool_calls;
    // }
    // if (msg.tool_call_id) {
    //     append_json["tool_call_id"] = *msg.tool_call_id;
    // }
    // msg_json.push_back(std::move(append_json));
    msg_json.push_back(msg.to_json());
    return msg_json;
}

inline nlohmann::json get_msg_json(const std::string_view system_prompt, const uint64_t id,
                                   const std::string_view name) {
    {
        auto session = g_chat_session_map.read();
        if (auto iter = session->find(id); iter != session->cend()) {
            return msg_list_to_json(system_prompt, iter->second.message_list);
        }
    }
    auto session = g_chat_session_map.write()->insert({id, ChatSession(name)});
    return msg_list_to_json(system_prompt, session.first->second.message_list);
}

inline void release_processing_llm(uint64_t id) {
    std::lock_guard lock(g_chat_processing_map.first);
    g_chat_processing_map.second[id] = false;
}

/**
 * @brief Stores information needed for data fetching operations.
 */
struct FetchData {
    std::string function; ///< Name of the function to be executed.
    std::string query;    ///< Query string containing the information to be fetched.

    /**
     * @brief Constructor to initialize FetchData with function and query values.
     * @param function Name of the function to execute.
     * @param query Information to be fetched.
     */
    FetchData(std::string function, std::string query) : function(std::move(function)), query(std::move(query)) {}
};

/**
 * @brief Represents the result structure for optimized message processing.
 */
struct OptimMessageResult {
    std::string summary;               ///< Summary of the optimization result.
    float query_date;                  ///< Date associated with the query properbility (stored as floating-point).
    std::vector<FetchData> fetch_data; ///< Collection of FetchData objects for fetch-related details.

    /**
     * @brief Default constructor initializes member variables to default values.
     */
    OptimMessageResult() : summary(), query_date(0.0f), fetch_data() {}

    /**
     * @brief Parameterized constructor initializes all member variables.
     * @param s Summary of the result.
     * @param date Query date properbility.
     * @param data Fetch-related data collection.
     */
    OptimMessageResult(std::string s, float date, std::vector<FetchData> data)
        : summary(std::move(s)), query_date(date), fetch_data(std::move(data)) {}
};

/**
 * @brief Calls a model to optimize message records based on the provided parameters.
 *
 * @param bot_profile The bot's profile containing configuration and context information.
 * @param sender_name The name of the sender of the message.
 * @param sender_id The unique identifier for the sender (e.g., user ID).
 * @param message_props The properties of the message, such as content and metadata.
 * @return std::optional<OptimMessageResult> The optimized message result, if available.
 *         Returns an empty optional if the optimization fails or no result is produced.
 */
std::optional<OptimMessageResult> optimize_message_query(const bot_adapter::Profile &bot_profile,
                                                         const std::string_view sender_name, qq_id_t sender_id,
                                                         const MessageProperties &message_props);

#endif