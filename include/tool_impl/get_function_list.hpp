// get_function_list tool implementation
#pragma once

#include "bot_cmd.h"
#include "chat_session.hpp"
#include <fmt/format.h>

namespace tool_impl {

inline ChatMessage get_function_list(const ToolCall &tool_call) {
	return ChatMessage(ROLE_TOOL, fmt::format("可用功能/函数列表: {}", bot_cmd::get_available_commands()),
					   tool_call.id);
}

} // namespace tool_impl

