#include "MiraiCP.hpp"
#include "plugin.h"
#include <string>

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info("Recv message: " + e.message.toString());
        for (auto msg : e.message) {
            MiraiCP::Logger::logger.info(std::string("Message Type: ") + std::to_string(msg.getType()) + ", Content: " + msg->content);
        }

    });
}