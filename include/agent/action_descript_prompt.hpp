#pragma once

#include <string_view>

namespace agent::action_prompt {
    const std::string_view ACTION_TO_USER_MENTION = "向这条消息做出反应,下列的function tools都是处理各种事情的专家,\
                你必须使用以下这些function tool做出回应";
}