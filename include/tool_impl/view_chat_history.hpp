// view_chat_history tool implementation
#pragma once

#include "adapter_model.h"
#include "bot_adapter.h"
#include "chat_session.hpp"
#include "global_data.h"
#include "tool_impl/common.hpp"
#include <fmt/format.h>

namespace tool_impl {

inline std::string get_target_group_chat_history(const bot_adapter::BotAdapter &adapter, const qq_id_t group_id,
												 qq_id_t target_id) {
	std::string target_name = std::to_string(target_id); // Default name is ID
	const auto &member_list = adapter.fetch_group_member_info(group_id)->get().member_info_list;
	if (auto member_info = member_list->find(target_id); member_info.has_value()) {
		target_name = member_info->get().member_name;
	}

	// Fetch recent messages from the group and filter by target user
	const auto &group_msg_list = g_group_message_storage.get_individual_last_msg_list(group_id, 1000);

	std::vector<std::string> target_msgs;
	target_msgs.reserve(100);
	size_t size_count = 0;
	for (auto it = group_msg_list.crbegin(); it != group_msg_list.crend(); ++it) {
		const auto &msg = *it;
		if (msg.sender_id == target_id) {
			auto text = bot_adapter::get_text_from_message_chain(*msg.message_chain_list);
			size_count += text.size();
			if (size_count > 3000) {
				break;
			}
			target_msgs.insert(target_msgs.begin(),
							   fmt::format("[{}]'{}': '{}'", time_point_to_db_str(msg.send_time), msg.sender_name,
										   std::move(text)));
		}
	}

	if (target_msgs.empty()) {
		return fmt::format("在'{}'群里没有找到 '{}' 的聊天记录", group_id, target_name);
	} else {
		return fmt::format("'{}'群内 '{}' 的最近消息:\n{}", group_id, target_name,
						   wheel::join_str(target_msgs.cbegin(), target_msgs.cend(), "\n"));
	}
}

using tool_impl::get_permission_chs;

inline ChatMessage view_chat_history(const bot_cmd::CommandContext &context, const ToolCall &tool_call) {
	const auto arguments = nlohmann::json::parse(tool_call.arguments);

	// try with qq id
	std::optional<qq_id_t> target_id_opt = get_optional<qq_id_t>(arguments, "targetId");
	std::string content = "啥记录都没有";

	if (const auto &group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr);
		group_sender.has_value()) {
		// Group Chat
		if (target_id_opt.has_value()) {
			// Target is specified in group chat
			spdlog::info("查询群 '{}' 内目标 '{}'(qq号) 的聊天记录", group_sender->get().group.name, *target_id_opt);

			content = get_target_group_chat_history(context.adapter, group_sender->get().group.id, *target_id_opt);

		} else {
			// Try with member name
			auto target_name = get_optional<std::string>(arguments, "targetName");
			if (target_name.has_value() && !target_name->empty()) {
				spdlog::info("查询群 '{}' 内目标 '{}' 的聊天记录", group_sender->get().group.name, *target_name);
				// query member name
				std::vector<qq_id_t> target_id_list =
					context.adapter.group_member_name_embedding_map.find(group_sender->get().group.id)
						->get()
						.get_similar_member_names(*target_name, 0.5f);
				if (target_id_list.empty()) {
					content = fmt::format("在'{}'群里没有找到名为'{}'的群友", group_sender->get().group.name,
										  *target_name);

				} else {
					content = "";
					for (const auto &target_id : target_id_list) {
						if (!content.empty()) {
							content += "\n";
						}
						spdlog::info("查询群 '{}' 内目标 '{}' 的聊天记录", group_sender->get().group.name, target_id);
						content += get_target_group_chat_history(context.adapter, group_sender->get().group.id,
																 target_id);
					}
				}
			} else {

				// No target, get group's history
				const auto &msg_list =
					g_group_message_storage.get_individual_last_msg_list(group_sender->get().group.id, 1000);
				if (msg_list.empty()) {
					content = fmt::format("在'{}'群里还没有聊天记录哦", group_sender->get().group.name);
				} else {
					std::string text;
					size_t total_length = 0;
					std::vector<std::string> msg_texts;
					for (auto it = msg_list.crbegin(); it != msg_list.crend(); ++it) {
						const auto &msg = *it;
						std::string msg_text = fmt::format(
							"[{}]'{}': '{}'", msg.sender_name, time_point_to_db_str(msg.send_time),
							bot_adapter::get_text_from_message_chain(*msg.message_chain_list));
						if (total_length + msg_text.length() > 3000) {
							break;
						}
						msg_texts.insert(msg_texts.begin(), msg_text);
						total_length += msg_text.length();
					}

					text = wheel::join_str(msg_texts.cbegin(), msg_texts.cend(), "\n");

					content = fmt::format("'{}'群的最近消息:\n{}", group_sender->get().group.name, text);
				}
			}
		}
	} else {
		// Friend chat, ignore target
		const auto &msg_list =
			g_person_message_storage.get_individual_last_msg_list(context.event->sender_ptr->id, 1000);
		if (msg_list.empty()) {
			content = "我们之间还没有聊天记录哦";
		} else {
			std::string text;
			size_t total_length = 0;
			for (const auto &msg : msg_list) {
				std::string msg_text = fmt::format("'{}': '{}'", msg.sender_name,
												   bot_adapter::get_text_from_message_chain(*msg.message_chain_list));
				if (total_length + msg_text.length() > 3000) {
					break;
				}
				if (!text.empty()) {
					text += "\n";
				}
				text += msg_text;
				total_length += msg_text.length();
			}

			content = fmt::format("与'{}'的最近聊天记录:\n{}", context.event->sender_ptr->name, text);
		}
	}
	return ChatMessage(ROLE_TOOL, content, tool_call.id);
}

} // namespace tool_impl

