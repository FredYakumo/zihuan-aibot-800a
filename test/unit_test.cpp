#include <gtest/gtest.h>
#include "adapter_message.h"

TEST(UnitTest, MsgPropTest) {
    bot_adapter::PlainTextMessage msg{"abc"};
    GTEST_LOG_(INFO) << msg.to_json();
}


TEST(UnitTest, MainTest) {
    GTEST_LOG_(INFO) << "Main unit test";
}