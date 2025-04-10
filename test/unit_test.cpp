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
    const auto test_sender_id = 3507578481;
    adapter.register_event<bot_adapter::GroupMessageEvent>([&adapter](const bot_adapter::GroupMessageEvent &e) {
        spdlog::info("接受到消息, 从sender: {}, group: {}", e.sender.id, e.group.name);

        if (e.sender.id == test_sender_id) {
            adapter.send_message(e.group,
                                bot_adapter::make_message_chain_list(
                                    bot_adapter::AtTargetMessage(e.sender.id),
                                    bot_adapter::PlainTextMessage(" test successed")
                                    ));
        }
    });

    adapter.start();
}