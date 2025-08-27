// view_model_info tool implementation
#pragma once

#include "agent/llm.h"
#include "chat_session.hpp"
#include <spdlog/spdlog.h>

namespace tool_impl {

inline ChatMessage view_model_info(const ToolCall &tool_call) {
	std::string model_info_str = "获取模型信息失败";
	if (auto model_info = agent::fetch_model_info(); model_info.has_value()) {
		model_info_str = model_info->dump();
	}
	return ChatMessage(ROLE_TOOL, model_info_str, tool_call.id);
}

} // namespace tool_impl

