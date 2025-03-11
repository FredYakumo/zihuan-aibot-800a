#include "utils.h"
#include "global_data.h"
#include <cstdint>
#include <fmt/format.h>
#include <string_view>
#include "config.h"

std::string gen_common_prompt(const std::string_view bot_name, const MiraiCP::QQID bot_id, const std::string_view user_name, const uint64_t user_id) {
    return fmt::format("你的名字叫{}(qq号{})，{}.你作为一个群友，只负责聊天。当前跟你聊天的群友的名字叫\"{}\"(qq号{})，当前时间是: {}", bot_name, bot_id, CUSTOM_SYSTEM_PROMPT, user_name, user_id, get_current_time_formatted());
}