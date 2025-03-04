#ifndef PLUGIN_H
#define PLUGIN_H

#include "MiraiCP.hpp"

class AIBot : public MiraiCP::CPPPlugin {
  // 配置插件信息
  AIBot() : CPPPlugin() {}
  ~AIBot() override = default; // override关键字是为了防止内存泄漏

  // 入口函数。插件初始化时会被调用一次，请在此处注册监听
  void onEnable() override { /*插件启动时执行一次*/ }

  // 退出函数。请在这里结束掉所有子线程，否则可能会导致程序崩溃
  void onDisable() override { /*插件结束前执行*/ }
};

#endif