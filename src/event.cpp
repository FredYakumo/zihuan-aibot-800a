#include "MiraiCP.hpp"
#include "plugin.h"

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info(":)");
        auto msg = e.message.toString();
        if (msg.find("@2496875785") != std::string::npos) {
            e.group.sendMessage("baka");
        }
    });
}