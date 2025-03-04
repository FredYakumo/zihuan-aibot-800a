#include "plugin.h"

const MiraiCP::PluginConfig MiraiCP::CPPPlugin::config{
    "AIBot-800a", // 插件id，如果和其他插件id重复将会被拒绝加载！
    "AIBot-800a",        // 插件名称
    "0.0.1",            // 插件版本
    "fredyakumo",        // 插件作者
    "Baka foo", // 可选：插件描述
    "1970-01-01"        // 可选：日期
};

// 创建当前插件实例。请不要进行其他操作，
// 初始化请在onEnable中进行
void MiraiCP::enrollPlugin() { MiraiCP::enrollPlugin<AIBot>(); }