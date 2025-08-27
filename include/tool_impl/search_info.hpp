// search_info tool implementation
#pragma once

#include "bot_adapter.h"
#include "chat_session.hpp"
#include "get_optional.hpp"
#include "vec_db/weaviate.h"
#include "rag.h"
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace tool_impl {

inline ChatMessage search_info(const bot_cmd::CommandContext &context, const ToolCall &tool_call) {
	const auto arguments = nlohmann::json::parse(tool_call.arguments);
	const std::optional<std::string> &query = get_optional(arguments, "query");
	bool include_date = get_optional(arguments, "includeDate").value_or(false);
	spdlog::info("Function call id {}: search_info(query={}, include_date={})", tool_call.id,
				 query.value_or(EMPTY_JSON_STR_VALUE), include_date);
	if (!query.has_value() || query->empty())
		spdlog::warn("Function call id {}: search_info(query={}, include_date={}), query is null", tool_call.id,
					 query.value_or(EMPTY_JSON_STR_VALUE), include_date);

	std::string content;
	const auto knowledge_list = vec_db::query_knowledge_from_vec_db(*query, 0.7f);
	for (const auto &knowledge : knowledge_list)
		content += fmt::format("{}\n", knowledge.content);

	const auto net_search_list =
		rag::net_search_content(include_date ? fmt::format("{} {}", get_current_time_formatted(), *query) : *query);
	std::vector<bot_adapter::ForwardMessageNode> first_replay;
	first_replay.emplace_back(
		context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
		context.adapter.get_bot_profile().name,
		bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(fmt::format("搜索: \"{}\"", *query))));
	for (const auto &net_search : net_search_list) {
		content += fmt::format("{}( {} ):{}\n", net_search.title, net_search.url, net_search.content);
		first_replay.emplace_back(
			context.adapter.get_bot_profile().id, std::chrono::system_clock::now(),
			context.adapter.get_bot_profile().name,
			bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage(
				fmt::format("关联度: {:.2f}%\n{}( {} )", net_search.score * 100.0f, net_search.title,
							net_search.url))));
	}
	if (!first_replay.empty()) {
		context.adapter.send_replay_msg(
			*context.event->sender_ptr,
			bot_adapter::make_message_chain_list(
				bot_adapter::ForwardMessage(first_replay, bot_adapter::DisplayNode(std::string("联网搜索结果")))));
	}
	return ChatMessage(ROLE_TOOL, content, tool_call.id);
}

} // namespace tool_impl

