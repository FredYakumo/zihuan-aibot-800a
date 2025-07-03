#ifndef BOT_CMD_H
#define BOT_CMD_H

#include "adapter_event.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "msg_prop.h"
#include <utility>

namespace bot_cmd {
    /**
     * @brief Describe a bot command run context.
     * This class should be construct at command running phase
     * 
     */
    struct CommandContext {
        bot_adapter::BotAdapter &adapter;
        std::shared_ptr<bot_adapter::MessageEvent> event;
        std::string param;
        bool is_deep_think = false;
        MessageProperties msg_prop;

        CommandContext(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::MessageEvent> e,
                       std::string_view param, bool is_deep_think, MessageProperties msg_prop)
            : adapter(adapter), event(e), param(param), is_deep_think(is_deep_think),
              msg_prop(msg_prop) {}
    };

    /**
     * @brief Describe a bot command run result.
     * 
     */
    struct CommandRes {
        /// The command wants to break following cmd process in run loop.
        bool interrupt_following_commands;
        /// The command wants to skip the last process_llm() function call.
        bool skip_default_process_llm;
        /// The command wants to reply in deep think mode.
        bool is_deep_think = false;
        /// Func accepts MessageProps ref and purpose to modify it, or nullopts if don't need to modify msg.
        std::optional<std::function<void(const MessageProperties &)>> is_modify_msg;

        CommandRes(bool interrupt_following_commands = false, bool skip_default_process_llm = false,
                   bool is_deep_think = false,
                   std::optional<std::function<void(const MessageProperties &)>> is_modify_msg = std::nullopt)
            : interrupt_following_commands(interrupt_following_commands),
              skip_default_process_llm(skip_default_process_llm), is_deep_think(is_deep_think),
              is_modify_msg(is_modify_msg) {}
    };

    struct CommandProperty {
        bool is_need_admin;
        bool is_need_param;
        std::function<CommandRes(CommandContext)> runer;

        CommandProperty(bool is_need_admin, bool is_need_param, std::function<CommandRes(CommandContext)> runner)
            : is_need_admin(is_need_admin), is_need_param(is_need_param), runer(runner) {}
        CommandProperty(std::function<CommandRes(CommandContext)> runner) : CommandProperty(true, false, runner) {}
        CommandProperty(bool is_need_admin, std::function<CommandRes(CommandContext)> runner)
            : CommandProperty(is_need_admin, false, runner) {}
    };
    CommandRes clear_chat_session_command(CommandContext context);
    CommandRes deep_think_command(CommandContext context);
    CommandRes queto_command(CommandContext context);
    CommandRes query_knowledge_command(CommandContext context);
    CommandRes add_knowledge_command(CommandContext context);
    CommandRes checkin_knowledge_command(CommandContext context);
    CommandRes query_memory_command(CommandContext context);
    CommandRes query_add_knowledge_list_command(CommandContext context);

    /**
     * @brief Handles a network search command based on the provided context.
     *
     * This function processes a network search command by extracting the search query from the message context.
     * If no valid search query is found, it prompts the user to provide one. Otherwise, it initiates a
     * background thread to perform the network search and process the results.
     *
     * @param context The command context containing message properties and sender information.
     * @return bot_cmd::CommandRes Returns a bot_cmd::CommandRes object indicating the success of the command handling.
     */
    CommandRes net_search_command(bot_cmd::CommandContext context);

    /**
     * @brief Handles a network search command based on the provided context.
     *
     * This function processes a network search command by extracting the search query from the message context.
     * If no valid search query is found, it prompts the user to provide one. Otherwise, it initiates a
     * background thread to perform the network search and process the results.
     *
     * @param context The command context containing message properties and sender information.
     * @return bot_cmd::CommandRes Returns a bot_cmd::CommandRes object indicating the success of the command handling.
     */
    CommandRes url_search_command(bot_cmd::CommandContext context);

    /**
     * @brief Retrieves and processes user preference settings.
     *
     * This function handles the command to get user preferences, including settings like markdown rendering,
     * text output format, and auto chat session timing.
     *
     * @param context The command context containing message properties and sender information.
     * @return bot_cmd::CommandRes Returns a bot_cmd::CommandRes object indicating the success of the command handling.
     */

    CommandRes get_user_preference_command(bot_cmd::CommandContext context);

    CommandRes set_user_preference_command(bot_cmd::CommandContext context);

    extern std::vector<std::pair<std::string, bot_cmd::CommandProperty>> keyword_command_map;

    inline void init_command_map() {
        keyword_command_map.push_back(
            std::make_pair("#新对话", bot_cmd::CommandProperty{false, false, bot_cmd::clear_chat_session_command}));
        keyword_command_map.push_back(
            std::make_pair("#思考", bot_cmd::CommandProperty{false, false, bot_cmd::deep_think_command}));
        keyword_command_map.push_back(
            std::make_pair("#查询知识", bot_cmd::CommandProperty{true, true, bot_cmd::query_knowledge_command}));
        keyword_command_map.push_back(
            std::make_pair("#添加知识", bot_cmd::CommandProperty{false, true, bot_cmd::add_knowledge_command}));
        keyword_command_map.push_back(
            std::make_pair("#入库知识", bot_cmd::CommandProperty{true, true, bot_cmd::checkin_knowledge_command}));
        keyword_command_map.push_back(
            std::make_pair("#查看记忆", bot_cmd::CommandProperty{true, bot_cmd::query_memory_command}));
        keyword_command_map.push_back(
            std::make_pair("#待添加知识", bot_cmd::CommandProperty{true, bot_cmd::query_add_knowledge_list_command}));
        keyword_command_map.push_back(
            std::make_pair("#联网", bot_cmd::CommandProperty{false, false, bot_cmd::net_search_command}));
        keyword_command_map.push_back(
            std::make_pair("#url", bot_cmd::CommandProperty{false, true, bot_cmd::url_search_command}));
        keyword_command_map.push_back(
            std::make_pair("#查看设置", bot_cmd::CommandProperty{false, false, bot_cmd::get_user_preference_command}));
        keyword_command_map.push_back(
            std::make_pair("#设置", bot_cmd::CommandProperty{false, true, bot_cmd::set_user_preference_command}));
    }

} // namespace bot_cmd

#endif