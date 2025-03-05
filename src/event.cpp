#include "MiraiCP.hpp"
#include "plugin.h"

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info(":)");
        auto msg = e.message.toString();
        
        if (msg.find(MiraiCP::At(2496875785).content) != std::string::npos) {
            e.group.sendMessage(MiraiCP::At(e.sender.id()).content + "baka");
        }
    });
}