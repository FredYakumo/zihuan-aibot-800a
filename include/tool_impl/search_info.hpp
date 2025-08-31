// search_info tool implementation
#pragma once

#include "bot_adapter.h"
#include "chat_session.hpp"
#include "get_optional.hpp"
#include "rag.h"
#include "vec_db/weaviate.h"
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace tool_impl {

    inline ChatMessage search_info(const bot_cmd::CommandContext &context, const ToolCall &tool_call,
                                   std::vector<bot_adapter::ForwardMessageNode> &out_first_replay_list) {
        const auto arguments = nlohmann::json::parse(tool_call.arguments);
        const std::optional<std::string> &query = get_optional(arguments, "query");
        std::string category = get_optional(arguments, "category").value_or("general");
        spdlog::info("Function call id {}: search_info(query={}, category={})", tool_call.id,
                     query.value_or(EMPTY_JSON_STR_VALUE), category);
        if (!query.has_value() || query->empty())
            spdlog::warn("Function call id {}: search_info(query={}, category={}), query is null", tool_call.id,
                         query.value_or(EMPTY_JSON_STR_VALUE), category);

        std::string content;
        const auto knowledge_list = vec_db::query_knowledge_from_vec_db(*query, 0.7f);
        if (!knowledge_list.empty()) {
            for (const auto &knowledge : knowledge_list)
                content += fmt::format("{}\n", knowledge.content);

            // extractly match knowledge, skip network search
            if (!knowledge_list.empty() && knowledge_list[0].certainty >= 0.87f) {
                return ChatMessage(ROLE_TOOL, content, tool_call.id);
            }
        }


        const auto net_search_list = rag::net_search_content(*query);
        out_first_replay_list.emplace_back(
            context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
            context.adapter.get_bot_profile().name,
            bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(fmt::format("搜索: \"{}\"", *query))));
        for (const auto &net_search : net_search_list) {
            content += fmt::format("{}( {} ):{}\n", net_search.title, net_search.url, net_search.content);
            out_first_replay_list.emplace_back(
                context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
                context.adapter.get_bot_profile().name,
                bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(fmt::format(
                    "关联度: {:.2f}%\n{}( {} )", net_search.score * 100.0f, net_search.title, net_search.url))));
        }
        return ChatMessage(ROLE_TOOL, content, tool_call.id);
    }

} // namespace tool_impl
