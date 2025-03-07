#include "MiraiCP.hpp"
#include "nlohmann/detail/conversions/from_json.hpp"
#include "nlohmann/detail/meta/type_traits.hpp"
#include "nlohmann/detail/value_t.hpp"
#include "plugin.h"
#include <memory>
#include <string>
#include <cpr/cpr.h>
#include <string_view>

std::string simple_get_llm_response(const std::string_view prompt) {
    nlohmann::json body = {
    {"model", "DeepSeek-R1:7b"},
    {"prompt", "你是一个语文老师，请通俗易懂地总结以下消息(总结内容不能太长)：" + std::string(prompt)}
    };
    MiraiCP::Logger::logger.info("llm body: " + body.dump());
    cpr::Response response = cpr::Post(
        cpr::Url{"http://localhost:11434/api/generate"},
        cpr::Body{body},
        cpr::Header{{"Content-Type", "application/json"}}
    );
    return response.text;
}

void AIBot::onEnable() {
    MiraiCP::Event::registerEvent<MiraiCP::GroupMessageEvent>([] (MiraiCP::GroupMessageEvent e) {
        MiraiCP::Logger::logger.info("Recv message: " + e.message.toString());
        MiraiCP::internal::Message *at_me_msg = nullptr;
        std::unique_ptr<std::string> ref_msg_content = nullptr;
        MiraiCP::internal::Message *plain_msg = nullptr;

        for (auto msg : e.message) {
            MiraiCP::Logger::logger.info(std::string("Message Type: ") + std::to_string(msg.getType()) + ", Content: " + msg->content);

            if (msg.getType() == MiraiCP::SingleMessageType::At_t && msg->content == std::to_string(e.bot.id())) {
                at_me_msg = &msg;
            } else if (msg.getType() == MiraiCP::SingleMessageType::QuoteReply_t) {
                MiraiCP::Logger::logger.info(msg.get()->toJson()["source"]["originalMessage"]);
                ref_msg_content = std::make_unique<std::string>(msg.get()->toJson()["source"]["originalMessage"].dump());
            } else if (msg.getType() == MiraiCP::SingleMessageType::PlainText_t) {
                plain_msg = &msg;
            }
        }

        if (at_me_msg == nullptr) {
            return;
        }

        if (plain_msg->get()->content.find("AI总结") != std::string::npos || plain_msg->get()->content.find("ai总结") != std::string::npos) {
            auto msg_chain = MiraiCP::MessageChain{e.sender.at()};
            if (ref_msg_content == nullptr) {
                msg_chain.add(MiraiCP::PlainText{std::string{" error: 请引用一个消息."}});
            } else {
                msg_chain.add((MiraiCP::PlainText{simple_get_llm_response(*ref_msg_content)}));
            }

            e.group.sendMessage(msg_chain);
        }

    });
}