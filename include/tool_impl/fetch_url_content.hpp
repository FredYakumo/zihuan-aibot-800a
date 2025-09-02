// fetch_url_content tool implementation
#pragma once

#include "chat_session.hpp"
#include "get_optional.hpp"
#include "rag.h"
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>

namespace tool_impl {

using wheel::join_str;
using wheel::ltrim;
using wheel::rtrim;

inline ChatMessage fetch_url_content(const bot_cmd::CommandContext &context, const ToolCall &tool_call,
									 const ChatMessage &llm_res) {
	const auto arguments = nlohmann::json::parse(tool_call.arguments);
	const std::vector<std::string> urls = get_optional(arguments, "urls").value_or(std::vector<std::string>());
	spdlog::info("Function call id {}: fetch_url_content(urls=[{}])", tool_call.id,
				 join_str(std::cbegin(urls), std::cend(urls)));
	if (urls.empty()) {
		spdlog::info("Function call id {}: fetch_url_content(urls=[{}])", tool_call.id,
					 join_str(std::cbegin(urls), std::cend(urls)));
		context.adapter.send_long_plain_text_reply(*context.event->sender_ptr, "你发的啥,我看不到...再发一遍呢?", true);
	} else {
		context.adapter.send_long_plain_text_reply(
			*context.event->sender_ptr, ltrim(rtrim(llm_res.content)).empty() ? "等我看看这个链接哦..." : llm_res.content,
			true);
	}

	const auto url_search_res = rag::url_search_content(urls);
	std::string content;

	// Process successful results
	for (const auto &[url, raw_content] : url_search_res.results) {
		content += fmt::format("链接[{}]内容:\n{}\n\n", url, raw_content);
	}

	// Process failed results
	if (!url_search_res.failed_reason.empty()) {
		content += "以下链接获取失败:\n";
		for (const auto &[url, error] : url_search_res.failed_reason) {
			content += fmt::format("链接[{}]失败原因: {}\n", url, error);
		}
	}

	if (url_search_res.results.empty()) {
		spdlog::error("url_search: {} failed", join_str(std::cbegin(urls), std::cend(urls)));
		return ChatMessage(ROLE_TOOL, "抱歉，所有链接都获取失败了,可能是网络抽风了或者网站有反爬机制导致紫幻获取不到内容",
						   tool_call.id);
	} else {
		return ChatMessage(ROLE_TOOL, content, tool_call.id);
	}
}

} // namespace tool_impl

