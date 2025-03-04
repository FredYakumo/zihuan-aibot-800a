#include "MiraiCP.hpp"
#include "plugin.h"

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info(":)");
        e.group.sendMessage("baka");
    });
}