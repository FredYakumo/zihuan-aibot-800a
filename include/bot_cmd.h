#ifndef BOT_CMD_H
#define BOT_CMD_H

#include "adapter_event.h"
#include "adapter_model.h"
#include "bot_adapter.h"
#include "msg_prop.h"
#include <utility>

namespace bot_cmd {
    struct CommandContext {
        bot_adapter::BotAdapter &adapter;
        std::shared_ptr<bot_adapter::MessageEvent> e;
        std::string param;
        bool is_no_param;
        bool is_deep_think = false;
        MessageProperties msg_prop;

        CommandContext(bot_adapter::BotAdapter &adapter, std::shared_ptr<bot_adapter::MessageEvent> e,
                       std::string_view param, bool is_command_only, bool is_deep_think, MessageProperties msg_prop)
            : adapter(adapter), e(e), param(param), is_no_param(is_command_only), is_deep_think(is_deep_think),
              msg_prop(msg_prop) {}
    };

    struct CommandRes {
        bool is_break_cmd_process;
        bool is_deep_think = false;
        std::optional<std::function<void(const MessageProperties &)>> is_modify_msg;

        CommandRes(bool is_break_cmd_process = false,
                   std::optional<std::function<void(const MessageProperties &)>> is_modify_msg = std::nullopt,
                   bool is_deep_think = false)
            : is_break_cmd_process(is_break_cmd_process), is_modify_msg(is_modify_msg), is_deep_think(is_deep_think) {}
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

    extern std::vector<std::pair<std::string, bot_cmd::CommandProperty>> keyword_command_map;

    inline void init_command_map() {
        // keyword_command_map.emplace("#语录", bot_cmd::CommandProperty{false, true, bot_cmd::queto_command});
        keyword_command_map.push_back(
            std::make_pair("#新对话", bot_cmd::CommandProperty{false, false, bot_cmd::deep_think_command}));
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
    }

} // namespace bot_cmd

#endif