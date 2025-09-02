// query_group tool implementation
#pragma once

#include "adapter_model.h"
#include "bot_adapter.h"
#include "bot_cmd.h"
#include <general-wheel-cpp/string_utils.hpp>
#include "chat_session.hpp"
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "tool_impl/common.hpp"

namespace tool_impl {

using tool_impl::get_permission_chs;
using wheel::join_str;

inline ChatMessage query_group(const bot_cmd::CommandContext &context, const ToolCall &tool_call) {
	const auto arguments = nlohmann::json::parse(tool_call.arguments);
	const std::string item = get_optional(arguments, "item").value_or("");
	spdlog::info("Function call id {}: query_group(item={})", tool_call.id, item);
	auto group_sender = try_group_sender(*context.event->sender_ptr);
	std::string content;
	if (!group_sender.has_value()) {
		content = "你得告诉我是哪个群";
	} else {
		if (item == "OWNER") {
			const auto &member_list =
				context.adapter.fetch_group_member_info(group_sender->get().group.id)->get().member_info_list;
			for (const auto &[key, value] : member_list->iter()) {
				if (value.permission == bot_adapter::GroupPermission::OWNER) {
					content += fmt::format("'{}'群的群主是: '{}'(QQ号为:{})", group_sender->get().group.name,
										   value.member_name, value.id);
					if (value.special_title.has_value()) {
						content += fmt::format("(特殊头衔: {})", value.special_title.value());
					}
					if (value.last_speak_time.has_value()) {
						content += fmt::format("(最后发言时间: {})", system_clock_to_string(value.last_speak_time.value()));
					}
					break;
				}
			}
		} else if (item == "ADMIN") {

			const auto &member_list =
				context.adapter.fetch_group_member_info(group_sender->get().group.id)->get().member_info_list;
			std::vector<std::string> admin_info_list;
			for (const auto &[key, value] : member_list->iter()) {
				if (value.permission == bot_adapter::GroupPermission::ADMINISTRATOR) {
					std::string admin_info = fmt::format("'{}'(QQ号为:{})", value.member_name, value.id);
					if (value.special_title.has_value()) {
						admin_info += fmt::format("(特殊头衔: {})", value.special_title.value());
					}
					if (value.last_speak_time.has_value()) {
						admin_info += fmt::format("(最后发言时间: {})", system_clock_to_string(value.last_speak_time.value()));
					}
					admin_info_list.push_back(admin_info);
				}
			}
			if (admin_info_list.empty()) {
				content = fmt::format("'{}'群里没有管理员", group_sender->get().group.name);
			} else {
				content = fmt::format(
					"'{}'群的管理员有:\n{}", group_sender->get().group.name,
					join_str(std::cbegin(admin_info_list), std::cend(admin_info_list), "\n"));
			}
		} else if (item == "PROFILE") {
			content = "暂未实现";
		} else if (item == "NOTICE") {
			auto announcements_opt = context.adapter.get_group_announcement_sync(group_sender->get().group.id);
			if (!announcements_opt.has_value()) {
				content = "获取群公告失败,可能是网络波动或没有权限";
			} else {
				if (announcements_opt->empty()) {
					content = "本群没有群公告";
				} else {
					const auto &member_list =
						context.adapter.fetch_group_member_info(group_sender->get().group.id)->get().member_info_list;
					std::vector<std::string> announcements_str_list;
					for (const auto &anno : *announcements_opt) {
						std::string sender_info;
						if (auto member_info_opt = member_list->find(anno.sender_id); member_info_opt.has_value()) {
							const auto &member_info = member_info_opt->get();
							sender_info = fmt::format("{}({})", member_info.member_name,
													  get_permission_chs(member_info.permission));
						} else {
							sender_info = std::to_string(anno.sender_id);
						}
						std::string anno_str = fmt::format("内容: {}\n发送者: {}\n发送时间: {}\n已确认人数: {}", anno.content,
															sender_info, system_clock_to_string(anno.publication_time),
															anno.confirmed_members_count);
						announcements_str_list.push_back(anno_str);
					}

					content = fmt::format("群'{}'的公告:\n", group_sender->get().group.name);
					content += join_str(std::cbegin(announcements_str_list), std::cend(announcements_str_list),
										"\n---\n");
				}
			}
			spdlog::info("获取群公告: {}", content);
		} else {
			content = "暂未实现";
		}
	}

	return ChatMessage(ROLE_TOOL, content, tool_call.id);
}

} // namespace tool_impl

