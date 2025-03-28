#include <gtest/gtest.h>
#include "adapter_message.h"
#include "bot_adapter.h"

TEST(UnitTest, MsgPropTest) {
    bot_adapter::PlainTextMessage msg{"abc"};
    GTEST_LOG_(INFO) << msg.to_json();
}

TEST(UnitTest, MainTest) {
    GTEST_LOG_(INFO) << "Main unit test";
}

TEST(UnitTest, BotAdapterTest) {
    bot_adapter::BotAdapter adapter {"ws://localhost:13378/all"};
    adapter.start();
}