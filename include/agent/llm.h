#pragma once


#include "bot_adapter.h"
#include "adapter_model.h"
#include "constant_types.hpp"
#include "msg_prop.h"
#include "user_protait.h"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace agent {

    std::string gen_common_prompt(const bot_adapter::Profile &bot_profile, const bot_adapter::BotAdapter &adapter,
                      const bot_adapter::Sender &sender, bool is_deep_think,
                      std::string_view action_description,
                      const std::optional<std::string> additional_system_prompt_option = std::nullopt);

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
            FetchData(std::string function, std::string query)
                : function(std::move(function)), query(std::move(query)) {}
        };

        /**
         * @brief Represents the result structure for optimized message processing.
         */
        struct OptimMessageResult {
            std::string summary; ///< Summary of the optimization result.
            float query_date;    ///< Date associated with the query properbility (stored as floating-point).
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

        struct ReplayContentNode {
            std::optional<std::string> normal;
            std::optional<std::string> rich_text;
        };

        std::vector<ReplayContentNode> optimize_reply_content(std::string content);

        /**
         * @brief Generates a user portrait based on the provided profile and message chain list.
         *
         * This function analyzes the user's profile and the message chain list to create a user portrait,
         * which is a summary of the user's characteristics and preferences. It utilizes the provided bot adapter
         * for any necessary interactions or data retrieval during the portrait generation process.
         *
         * @param adapter
         * @param profile The user's profile containing basic information.
         * @param message_chain_list A list of message chains representing the user's interactions.
         * @return UserProtait Returns a UserProtait object containing the generated portrait and creation time.
         */
        std::optional<UserProtait>
        generate_user_protait(const bot_adapter::BotAdapter &adapter, const bot_adapter::Profile &profile,
                              const std::vector<bot_adapter::MessageChainPtrList> &message_chain_list);

        /**
         * @brief Fetch model info from the model server.
         *
         * @return std::optional<nlohmann::json> Returns a JSON object containing model info, if available.
         *         Returns an empty optional if the fetch fails or no info is available.
         */
        std::optional<nlohmann::json> fetch_model_info();

} // namespace agent
