#include "MiraiCP.hpp"
#include "config.h"
#include "plugin.h"
#include "utils.h"
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <functional>
#include <global_data.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include "bot_cmd.h"
#include "msg_prop.h"
#include "llm.h"


void AIBot::onEnable() {
    // MiraiCP::Event::registerEvent<MiraiCP::BotOnlineEvent>([](MiraiCP::BotOnlineEvent e) {
    //     init_config();
    // });
    init_config();
    bot_cmd::init_command_map();

    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([](MiraiCP::GroupMessageEvent e) {
        // MiraiCP::Logger::logger.info("Recv message: " + e.message.toString());
        auto sender_id = e.sender.id();

        if (is_banned_id(sender_id)) {
            return;
        }

        auto bot_name = e.bot.nick();
        auto bot_id = e.bot.id();
        const auto msg_prop = get_msg_prop_from_event(e, bot_name);
        const auto group_id = e.group.id();
        const auto group_name = e.group.nickOrNameCard();
        const auto sender_name = e.sender.nickOrNameCard();
        const auto admin = is_admin(sender_id);

        // @TODO: need optim
        if (msg_prop.is_at_me) {
            MiraiCP::Logger::logger.info("开始处理指令信息");
            for (auto &cmd : bot_cmd::keyword_command_map) {
                if (cmd.second.is_need_admin && !admin) {
                    continue;
                }
                if (msg_prop.plain_content == nullptr || msg_prop.plain_content->empty()) {
                    continue;
                }
                bot_cmd::CommandRes res;
                if (cmd.second.is_need_param) {
                    if (auto param = extract_parentheses_content_after_keyword(*msg_prop.plain_content, cmd.first); !param.empty()) {
                        res = cmd.second.runer(bot_cmd::CommandContext{e, param,
                                                              is_strict_format(*msg_prop.plain_content, cmd.first),
                                                              sender_id, group_id, sender_name, bot_name, bot_id, msg_prop});
                    } else {
                        // auto msg_chain = MiraiCP::MessageChain{MiraiCP::At(sender_id)};
                        // msg_chain.add(MiraiCP::PlainText{fmt::format(" 错误。请指定参数, 用法 {} (...)",
                        // cmd.first)}); e.group.sendMessage(msg_chain); return;
                        continue;
                    }
                } else {
                    if (msg_prop.plain_content->find(cmd.first) != std::string::npos) {
                        res = cmd.second.runer(bot_cmd::CommandContext{e, "", true, sender_id, group_id, sender_name, bot_name, bot_id, msg_prop});
                    } else {
                        continue;
                    }
                }
                if (res.is_stop_command) {
                    return;
                }
                if (res.is_modify_msg) {
                    res.is_modify_msg.value()(msg_prop);
                }
                break;
            }
        }

        auto msg_storage_thread = std::thread([msg_prop, group_id, sender_id, sender_name, group_name] {
            set_thread_name("AIBot msg storage");
            MiraiCP::Logger::logger.info("Start message storage thread.");
            msg_storage(msg_prop, group_id, sender_id, sender_name, group_name);
        });
        msg_storage_thread.detach();

        if (!msg_prop.is_at_me) {
            return;
        }

        auto context = bot_cmd::CommandContext(e, "", false, 
            sender_id, group_id, sender_name, bot_name, bot_id, msg_prop);
        process_llm(context);
    });
}