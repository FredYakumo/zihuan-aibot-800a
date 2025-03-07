#include "MiraiCP.hpp"
#include "plugin.h"
#include <string>

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info("Recv message: " + e.message.toString());
        MiraiCP::internal::Message *at_me_msg = nullptr;
        MiraiCP::internal::Message *ref_msg = nullptr;
        MiraiCP::internal::Message *plain_msg = nullptr;

        for (auto msg : e.message) {
            MiraiCP::Logger::logger.info(std::string("Message Type: ") + std::to_string(msg.getType()) + ", Content: " + msg->content);

            if (msg.getType() == MiraiCP::SingleMessageType::At_t && msg->content == std::to_string(e.bot.id())) {
                at_me_msg = &msg;
            } else if (msg.getType() == MiraiCP::SingleMessageType::QuoteReply_t) {
                ref_msg = &msg;
            } else if (msg.getType() == MiraiCP::SingleMessageType::PlainText_t) {
                plain_msg = &msg;
            }
        }

        if (at_me_msg == nullptr) {
            return;
        }

        if (plain_msg->get()->content.find("AI总结") != std::string::npos || plain_msg->get()->content.find("ai总结") != std::string::npos) {
            auto msg_chain = MiraiCP::MessageChain{e.sender.at()};
            if (ref_msg == nullptr) {
                msg_chain.emplace_back(MiraiCP::PlainText{std::string{"error: 请引用一个消息."}});
            } else {
                msg_chain.emplace_back(MiraiCP::PlainText{std::string{"用户让我AI总结消息: \""} + 
                ref_msg->get()->content + std::string{"\", 但是现在model=DeepSeek-R1:0b"}});
            }

            e.group.sendMessage(msg_chain);
        }

    });
}