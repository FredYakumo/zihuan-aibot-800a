# 机器人命令参考

本文档提供了`bot_cmd`命名空间的参考，该命名空间负责处理AIBot Zihuan中的用户命令。命令是触发特定预定义操作的特殊消息。

## 核心组件

### `CommandContext` 结构体

该结构体封装了命令执行所需的所有必要信息。它会传递给每个命令处理函数。

```cpp
struct CommandContext {
    bot_adapter::BotAdapter &adapter;
    std::shared_ptr<bot_adapter::MessageEvent> event;
    std::string param;
    bool is_deep_think = false;
    MessageProperties msg_prop;
};
```

- `adapter`: `BotAdapter`实例的引用，用于发送回复和其他机器人操作。
- `event`: 触发命令的`MessageEvent`的共享指针。包含发送者信息、消息内容等。
- `param`: 命令的参数字符串。这是消息中跟随命令关键字的部分。
- `is_deep_think`: 一个布尔标志，指示机器人是否处于"深度思考"模式。
- `msg_prop`: 包含消息解析属性的对象。