#include "agent/simple_chat_action_agent.h"
#include "agent/llm.h" // for gen_common_prompt
#include "rag.h"
#include "event.h" // try_to_replay_person/release_processing_replay_person
#include "global_data.h"
#include "utils.h"
#include <general-wheel-cpp/string_utils.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <thread>

namespace agent {

	// local copy of helper from llm.cpp (consider centralizing later)
	inline std::string get_permission_chs(const std::string_view perm) {
		if (perm == "OWNER") {
			return "群主";
		} else if (perm == "ADMINISTRATOR") {
			return "管理员";
		}
		return "普通群友";
	}

	// forward declare internal worker defined in llm.cpp
	void on_llm_thread(const bot_cmd::CommandContext &context, const std::string &llm_content,
					   const std::string &system_prompt,
					   const std::optional<database::UserPreference> &user_preference_option);

	void SimpleChatActionAgent::process_llm(const bot_cmd::CommandContext &context,
											const std::optional<std::string> &additional_system_prompt_option,
											const std::optional<database::UserPreference> &user_preference_option) {
		spdlog::info("(Class) 开始处理LLM信息");

		if (!try_to_replay_person(context.event->sender_ptr->id)) {
			spdlog::warn("User {} try to let bot answer, but bot is still thiking", context.event->sender_ptr->id);
			adapter->send_replay_msg(
				*context.event->sender_ptr,
				bot_adapter::make_message_chain_list(bot_adapter::PlainTextMessage("我还在思考中...你别急")));
			return;
		}

		spdlog::debug("Event type: {}, Sender json: {}", context.event->get_typename(),
					  context.event->sender_ptr->to_json().dump());

		std::string llm_content{};
		if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
			llm_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
										   *context.msg_prop.ref_msg_content));
		}
		if (!context.msg_prop.at_id_set.empty() &&
			!(context.msg_prop.at_id_set.size() == 1 &&
			  *context.msg_prop.at_id_set.begin() == adapter->get_bot_profile().id)) {
			llm_content.append("本消息提到了：");
			bool first = true;
			for (const auto &at_id : context.msg_prop.at_id_set) {
				if (!first) {
					llm_content.append("、");
				}
				llm_content.append(fmt::format("{}", at_id));
				first = false;
			}
			llm_content.append("。");
		}
		std::string speak_content = EMPTY_MSG_TAG;
		if (context.msg_prop.plain_content != nullptr &&
			!wheel::ltrim(wheel::rtrim(*context.msg_prop.plain_content)).empty()) {
			speak_content = *context.msg_prop.plain_content;
		}

		if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
			if (auto group_sender = bot_adapter::try_group_sender(*context.event->sender_ptr); group_sender.has_value()) {
				llm_content.append(fmt::format("{}\"{}\"({})[{}]对你说: \"{}\"",
							   get_permission_chs(group_sender->get().permission),
							   group_sender->get().name, context.event->sender_ptr->id,
							   get_current_time_formatted(), speak_content));
			} else {
				llm_content.append(fmt::format("\"{}\"({})[{}]对你说: \"{}\"", context.event->sender_ptr->name,
							   context.event->sender_ptr->id, get_current_time_formatted(),
							   speak_content));
			}
		}
		std::string mixed_input_content;
		if (context.msg_prop.ref_msg_content != nullptr && !context.msg_prop.ref_msg_content->empty()) {
			mixed_input_content.append(fmt::format("\"{}\"引用了一个消息: \"{}\",", context.event->sender_ptr->name,
												   *context.msg_prop.ref_msg_content));
		}
		if (context.msg_prop.plain_content != nullptr && !context.msg_prop.plain_content->empty()) {
			mixed_input_content.append(fmt::format("{}", *context.msg_prop.plain_content));
		}
		auto session_knowledge_opt = rag::query_knowledge(mixed_input_content, false, context.event->sender_ptr->id,
															  context.event->sender_ptr->name);
		std::string system_prompt =
			gen_common_prompt(adapter->get_bot_profile(), *adapter, *context.event->sender_ptr,
						  context.is_deep_think, additional_system_prompt_option);

		// Add session knowledge to system prompt if available
		if (session_knowledge_opt.has_value()) {
			system_prompt += "\n" + session_knowledge_opt.value();
		}

		spdlog::info("作为用户输入给llm的content: {}", llm_content);

		auto llm_thread = std::thread([context, llm_content, system_prompt, user_preference_option] {
			on_llm_thread(context, llm_content, system_prompt, user_preference_option);
		});

		llm_thread.detach();
	}
} // namespace agent