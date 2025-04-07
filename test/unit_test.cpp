#include "adapter_event.h"
#include "adapter_message.h"
#include "bot_adapter.h"
#include <gtest/gtest.h>

TEST(UnitTest, MsgPropTest) {
    bot_adapter::PlainTextMessage msg{"abc"};
    GTEST_LOG_(INFO) << msg.to_json();
}

TEST(UnitTest, MainTest) { GTEST_LOG_(INFO) << "Main unit test"; }

TEST(UnitTest, BotAdapterTest) {
    spdlog::set_level(spdlog::level::debug);
    bot_adapter::BotAdapter adapter{"ws://localhost:13378/all"};

    adapter.register_event<bot_adapter::GroupMessageEvent>([](const bot_adapter::GroupMessageEvent &e) {
        spdlog::info("接受到消息, 从sender: {}", e.sender.id);
    });

    adapter.start();
}