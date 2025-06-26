# Bot Adapter API 参考

本文档提供了 `bot_adapter` 库的参考，该库用于通过 WebSocket 与Mirai qqbot或者onebot(使用of)进行通信(以下简称qq bot)。其主要组件是 `BotAdapter` 类。

## 名词定义
- qq bot: Mirai机器人的实体或者one bot机器人实体, qq bot负责控制QQ账号的运行生命周期
- zihuan-aibot-800a: 本项目,使用大语言模型及各种模型,以及数据库等的智能体。通过与qq bot交互来实现紫幻的能力。以下简称aibot zihuan。

## `BotAdapter` 类

`BotAdapter` 类是AIBot zihuan与qq bot交互的主要入口点。它管理 WebSocket 连接、处理传入事件，并提供发送消息和检索信息的方法。

### 生命周期

#### 构造函数

```cpp
BotAdapter::BotAdapter(const std::string_view url, std::optional<uint64_t> bot_id_option = std::nullopt);
```

创建一个新的 `BotAdapter` 实例。

-   `url`:  WebSocket URL。
-   `bot_id_option`: qq bot的可选 qq号。存储在bot的配置文件中。

#### 析构函数

```cpp
BotAdapter::~BotAdapter();
```

清理资源，包括 WebSocket 连接。

#### `start`

```cpp
int BotAdapter::start();
```

启动AIBot zihuan。此方法进入一个阻塞循环，轮询来自 WebSocket 服务器的传入消息并进行分发。它还会启动一个后台线程来定期更新群组信息。该函数将一直运行直到 WebSocket 连接关闭。

### 事件处理

#### `register_event`

```cpp
template <typename EventFuncT>
inline void BotAdapter::register_event(std::function<void(std::shared_ptr<EventFuncT> e)> func);
```

为特定事件类型注册一个处理函数。这是响应传入消息等事件的主要机制。

-   `EventFuncT`: 要监听的事件类型。必须是派生自 `Event` 的类。例如 `GroupMessageEvent` 和 `FriendMessageEvent`。
-   `func`: 一个接收 `std::shared_ptr` 事件对象的函数（或 lambda）。

**示例:**

```cpp
bot.register_event<GroupMessageEvent>([](std::shared_ptr<GroupMessageEvent> e) {
    spdlog::info("收到群消息: {}", e->to_json().dump());
    // ... 处理群消息
});
```

### 消息发送

这些函数用于向好友或群组发送消息。

#### `send_message`

```cpp
void BotAdapter::send_message(
    const Sender &sender,
    const MessageChainPtrList &message_chain,
    std::optional<std::string_view> sync_id_option = std::nullopt,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

向好友发送消息。这是发送好友消息的基本函数。

-   `sender`: 要发送消息的好友。
-   `message_chain`: 构成消息的消息组件列表（文本、图片等）。
-   `sync_id_option`: 用于跟踪命令的可选自定义同步 ID。
-   `out_message_id_option`: 用于接收已发送消息 ID 的可选回调。

#### `send_group_message`

```cpp
void BotAdapter::send_group_message(
    const Group &group,
    const MessageChainPtrList &message_chain,
    std::optional<std::string_view> sync_id_option = std::nullopt,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

向群组发送消息。这是发送群组消息的基本函数。

-   `group`: 目标群组。
-   `message_chain`: 消息内容。
-   `sync_id_option`: 可选的自定义同步 ID。
-   `out_message_id_option`: 用于接收已发送消息 ID 的可选回调。

#### `send_replay_msg`

```cpp
void BotAdapter::send_replay_msg(
    const Sender &sender,
    const MessageChainPtrList &message_chain,
    bool at_target = true,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

一个用于回复已收消息的便捷函数。它会自动处理原始消息是来自群组还是好友。

-   `sender`: 要回复的人。
-   `message_chain`: 回复内容。
-   `at_target`: 如果为 `true` 并且消息在群组中，它将 `@` 这个人。

#### `send_long_plain_text_reply`

```cpp
void BotAdapter::send_long_plain_text_reply(
    const Sender &sender,
    std::string text,
    bool at_target = true,
    uint64_t msg_length_limit = MAX_OUTPUT_LENGTH);
```

发送具有高级格式化功能的长文本回复。该函数会将输入的 `text` 解析为 Markdown。

-   **Markdown 解析**: 它可以处理复杂的 Markdown 结构。
-   **图片渲染**: 对于代码块、表格或富文本等元素，它会调用外部服务将它们渲染为图片，然后发送。这避免了纯文本消息中的字符限制和格式问题。
-   **消息拆分**: 如果文本太长，它将被拆分为多个部分。
-   **合并转发消息**: 最终的输出通常作为单条"合并转发消息"发送，以保持内容聚合在一起。

此函数非常适合用于发送来自 LLM 等服务的复杂、格式化的响应。

### 信息检索

这些方法用于获取有关机器人、用户和群组的数据。

#### `update_bot_profile`

```cpp
void BotAdapter::update_bot_profile();
```

从服务器异步获取qq bot自身(QQ账号)的配置文件信息（如昵称、电子邮件等）并更新本地的 `bot_profile` 对象。

#### `get_bot_profile`

```cpp
const Profile &BotAdapter::get_bot_profile() const;
```

返回对本地缓存的AIBot zihuan(QQ账号)信息的常量引用。

#### `update_group_info_sync`

```cpp
void BotAdapter::update_group_info_sync();
```

同步获取qq bot所属的所有群组的列表，并为每个群组获取其完整的成员列表。这是一个潜在的长时间运行的阻塞操作，会填充内部缓存。`start()` 方法会在后台线程中定期运行此操作。

#### `fetch_group_member_info`

```cpp
inline std::optional<std::reference_wrapper<const GroupWrapper>>
BotAdapter::fetch_group_member_info(qq_id_t group_id) const;
```

从本地缓存中获取特定群组的信息（包括其成员列表）。该缓存由 `update_group_info_sync()` 填充。

-   如果缓存中找到该群组，则返回一个 `GroupWrapper` 对象，否则返回 `std::nullopt`。

#### `get_group`

```cpp
inline const GroupWrapper &BotAdapter::get_group(qq_id_t group_id) const;
```

与 `fetch_group_member_info` 类似，但直接返回对 `GroupWrapper` 的引用。如果缓存中未找到该群组，则会抛出异常。

#### `get_message_id`

```cpp
inline void BotAdapter::get_message_id(
    uint64_t message_id,
    uint64_t target_id,
    CommandResHandleFunc out_func);
```

通过其 ID 异步获取消息。

-   `message_id`: 要获取的消息的 ID。
-   `target_id`: 消息所在的好友或群组的 ID。
-   `out_func`: 将使用结果调用的回调函数。

#### `get_group_announcement`

```cpp
inline void BotAdapter::get_group_announcement(
    qq_id_t group_id,
    GroupAnnouncementResHandleFunc out_func,
    int offset = 0,
    int size = 10);
```

异步获取给定群组的公告列表。

-   `group_id`: 目标群组的 ID。
-   `out_func`: 用于处理 `GroupAnnouncement` 列表的回调。
-   `offset`: 用于分页的起始偏移量。
-   `size`: 要获取的公告数量。

#### `get_group_announcement_sync`

```cpp
inline std::optional<std::vector<GroupAnnouncement>>
BotAdapter::get_group_announcement_sync(
    qq_id_t group_id,
    int offset = 0,
    int size = 10,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(20000));
```

同步获取群组公告。此函数会阻塞直到收到响应或超时。

-   成功时返回 `GroupAnnouncement` 的向量，失败或超时则返回 `std::nullopt`。 