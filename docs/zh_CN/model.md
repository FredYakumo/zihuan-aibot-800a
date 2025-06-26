# 数据模型参考

本文档详细介绍了 `bot_adapter` 库中使用的数据模型，这些模型用于表示用户、群组、消息以及其他与 QQ Bot 交互时所需的数据结构。

## 核心模型 (`adapter_model.h`)

这些是表示用户和群组基本信息的核心结构。

### `Sender`

代表一个消息发送者。

```cpp
struct Sender {
    qq_id_t id;
    std::string name;
    std::optional<std::string> remark;
};
```

-   `id`: 发送者的 QQ 号。
-   `name`: 发送者的显示名称（群名片或昵称）。
-   `remark`: 对发送者的备注信息。

### `Group`

代表一个聊天群组。

```cpp
struct Group {
    uint64_t id;
    std::string name;
    std::string permission;
};
```

-   `id`: 群组的 ID。
-   `name`: 群组的名称。
-   `permission`: Bot 在该群的默认权限。

### `GroupSender`

代表一个群消息发送者，继承自 `Sender` 并包含额外的群相关信息。

```cpp
struct GroupSender : public Sender {
    std::string permission;
    std::optional<std::chrono::system_clock::time_point> join_time;
    std::chrono::system_clock::time_point last_speak_time;
    Group group;
};
```

-   `permission`: 成员在群中的权限（如 `"MEMBER"`, `"ADMINISTRATOR"`）。
-   `join_time`: 成员的入群时间。
-   `last_speak_time`: 成员的最后发言时间。
-   `group`: 该成员所在的 `Group` 对象。

### `Profile`

代表一个用户的个人资料。

```cpp
struct Profile {
    uint64_t id;
    std::string name;
    std::string email;
    uint32_t age;
    uint32_t level;
    ProfileSex sex;
};
```

-   `id`: 用户的 QQ 号。
-   `name`: 用户的昵称。
-   `email`: 用户的邮箱地址。
-   `age`: 用户的年龄。
-   `level`: 用户的等级。
-   `sex`: 用户的性别 (`ProfileSex` 枚举：`UNKNOWN`, `MALE`, `FEMALE`)。

### `GroupInfo`

包含群组的基本信息以及 Bot 在群内的权限。

```cpp
struct GroupInfo {
    qq_id_t group_id;
    std::string name;
    GroupPermission bot_in_group_permission;
};
```

- `group_id`: 群号。
- `name`: 群名称。
- `bot_in_group_permission`: Bot 在此群的权限 (`GroupPermission` 枚举)。

### `GroupMemberInfo`

包含群成员的详细信息。

```cpp
struct GroupMemberInfo {
    qq_id_t id;
    qq_id_t group_id;
    std::string member_name;
    std::optional<std::string> special_title;
    GroupPermission permission;
    std::optional<std::chrono::system_clock::time_point> join_time;
    std::optional<std::chrono::system_clock::time_point> last_speak_time;
    float mute_time_remaining;
};
```

- `id`: 成员的 QQ 号。
- `group_id`: 所在群的群号。
- `member_name`: 成员的群名片。
- `special_title`: 成员的特殊头衔。
- `permission`: 成员的权限。
- `join_time`: 入群时间。
- `last_speak_time`: 最后发言时间。
- `mute_time_remaining`: 剩余禁言时间（秒）。

### `GroupAnnouncement`

代表一条群公告。

```cpp
struct GroupAnnouncement {
    bool all_confirmed;
    std::string content;
    std::string fid;
    qq_id_t group_id;
    std::chrono::system_clock::time_point publication_time;
    qq_id_t sender_id;
    bool is_top;
};
```

- `all_confirmed`: 是否所有人都已确认。
- `content`: 公告内容。
- `fid`: 公告的唯一标识符。
- `group_id`: 所在群的群号。
- `publication_time`: 发布时间。
- `sender_id`: 发布者的 QQ 号。
- `is_top`: 是否为置顶公告。

## 消息模型 (`adapter_message.h`)

这些模型用于表示不同类型的消息。它们都继承自 `MessageBase` 虚基类。

### `MessageBase`

所有消息类型的抽象基类。

```cpp
struct MessageBase {
    virtual std::string_view get_type() const = 0;
    virtual nlohmann::json to_json() const = 0;
    virtual const std::string &display_text() const = 0;
};
```
`MessageChainPtrList` 被定义为 `std::vector<std::shared_ptr<MessageBase>>`，用于组成消息链。

### `PlainTextMessage`

纯文本消息。

```cpp
struct PlainTextMessage : public MessageBase {
    std::string text;
};
```

### `AtTargetMessage`

`@`某人的消息。

```cpp
struct AtTargetMessage : public MessageBase {
    uint64_t target;
};
```
- `target`: 被`@`用户的 QQ 号。

### `ImageMessage` / `LocalImageMessage`

图片消息。`ImageMessage` 使用 URL，而 `LocalImageMessage` 使用本地文件路径。

```cpp
struct ImageMessage : public MessageBase {
    std::string url;
    std::optional<std::string> describe_text;
};

struct LocalImageMessage : public MessageBase {
    std::string path;
    std::optional<std::string> describe_text;
};
```

### `QuoteMessage`

引用回复消息。

```cpp
struct QuoteMessage : public MessageBase {
    std::string text;
    message_id_t ref_msg_id;
    std::optional<qq_id_t> ref_group_id_opt;
    std::optional<qq_id_t> ref_friend_id_opt;
};
```
- `text`: 引用消息的文本预览。
- `ref_msg_id`: 被引用的消息 ID。
- `ref_group_id_opt`: 如果是群消息，则为被引用消息所在的群号。
- `ref_friend_id_opt`: 如果是好友消息，则为被引用消息所在的好友QQ号。

### `ForwardMessage`

合并转发消息。

```cpp
struct ForwardMessage : public MessageBase {
    std::vector<ForwardMessageNode> node_list;
    std::optional<DisplayNode> display;
};
```
- `node_list`: 一个 `ForwardMessageNode` 的列表，每个节点代表一条被转发的消息。
- `display`: 转发消息的自定义显示样式，包含标题和摘要。

#### `ForwardMessageNode`

合并转发消息中的单个节点。

```cpp
struct ForwardMessageNode {
    uint64_t sender_id;
    std::chrono::system_clock::time_point time;
    std::string sender_name;
    MessageChainPtrList message_chain;
    std::optional<uint64_t> message_id;
    std::optional<uint64_t> message_ref;
};
```
包含了单条被转发消息的发送者、时间、内容等信息。

## 消息属性模型 (`msg_prop.h`)

这个模型用于从原始消息事件中提取出对大语言模型（LLM）有用的结构化信息。

### `MessageProperties`

代表一条消息中提取出的关键属性，方便后续处理。

```cpp
struct MessageProperties {
    bool is_at_me;
    std::shared_ptr<std::string> ref_msg_content;
    std::shared_ptr<std::string> plain_content;
    std::set<uint64_t> at_id_set;
};
```

-   `is_at_me`: 一个布尔值，表示这条消息是否`@`了aibot zihuan本身。
-   `ref_msg_content`: 一个指向字符串的共享指针，包含了被引用的消息内容。如果消息没有引用其他消息，则为 `nullptr`。
-   `plain_content`: 一个指向字符串的共享指针，包含了消息中的纯文本内容（已经过清理，例如去除了`@`信息）。
-   `at_id_set`: 一个集合，包含了消息中所有被`@`用户的 QQ 号。
