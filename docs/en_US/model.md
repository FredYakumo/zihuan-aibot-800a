# Data Model Reference

This document details the data models used in the `bot_adapter` library. These models represent users, groups, messages, and other data structures required for interacting with a QQ Bot.

## Core Models (`adapter_model.h`)

These are the core structures for representing basic user and group information.

### `Sender`

Represents a message sender.

```cpp
struct Sender {
    qq_id_t id;
    std::string name;
    std::optional<std::string> remark;
};
```

-   `id`: The sender's QQ ID (QQ号).
-   `name`: The sender's display name (group nickname or regular nickname).
-   `remark`: A remark or note about the sender.

### `Group`

Represents a chat group.

```cpp
struct Group {
    uint64_t id;
    std::string name;
    std::string permission;
};
```

-   `id`: The group's ID.
-   `name`: The group's name.
-   `permission`: The Bot's default permission level in this group.

### `GroupSender`

Represents a group message sender, inheriting from `Sender` and containing additional group-related information.

```cpp
struct GroupSender : public Sender {
    std::string permission;
    std::optional<std::chrono::system_clock::time_point> join_time;
    std::chrono::system_clock::time_point last_speak_time;
    Group group;
};
```

-   `permission`: The member's permission level in the group (e.g., `"MEMBER"`, `"ADMINISTRATOR"`).
-   `join_time`: The time the member joined the group.
-   `last_speak_time`: The last time the member spoke.
-   `group`: The `Group` object to which this member belongs.

### `Profile`

Represents a user's profile.

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

-   `id`: The user's QQ ID (QQ号).
-   `name`: The user's nickname.
-   `email`: The user's email address.
-   `age`: The user's age.
-   `level`: The user's level.
-   `sex`: The user's gender (enum `ProfileSex`: `UNKNOWN`, `MALE`, `FEMALE`).

### `GroupInfo`

Contains basic information about a group and the Bot's permissions within it.

```cpp
struct GroupInfo {
    qq_id_t group_id;
    std::string name;
    GroupPermission bot_in_group_permission;
};
```

- `group_id`: The group ID.
- `name`: The group name.
- `bot_in_group_permission`: The Bot's permission in this group (enum `GroupPermission`).

### `GroupMemberInfo`

Contains detailed information about a group member.

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

- `id`: The member's QQ ID (QQ号).
- `group_id`: The ID of the group they belong to.
- `member_name`: The member's group nickname.
- `special_title`: The member's special title.
- `permission`: The member's permission level.
- `join_time`: The time they joined the group.
- `last_speak_time`: The last time they spoke.
- `mute_time_remaining`: Remaining mute time in seconds.

### `GroupAnnouncement`

Represents a group announcement.

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

- `all_confirmed`: Whether everyone has confirmed the announcement.
- `content`: The announcement's content.
- `fid`: The unique identifier for the announcement.
- `group_id`: The ID of the group where the announcement was posted.
- `publication_time`: The time of publication.
- `sender_id`: The QQ ID (QQ号) of the sender.
- `is_top`: Whether the announcement is pinned.

## Message Models (`adapter_message.h`)

These models represent different types of messages. They all inherit from the `MessageBase` virtual base class.

### `MessageBase`

The abstract base class for all message types.

```cpp
struct MessageBase {
    virtual std::string_view get_type() const = 0;
    virtual nlohmann::json to_json() const = 0;
    virtual const std::string &display_text() const = 0;
};
```
`MessageChainPtrList` is defined as `std::vector<std::shared_ptr<MessageBase>>` and is used to form message chains.

### `PlainTextMessage`

A plain text message.

```cpp
struct PlainTextMessage : public MessageBase {
    std::string text;
};
```

### `AtTargetMessage`

A message that `@`'s a user.

```cpp
struct AtTargetMessage : public MessageBase {
    uint64_t target;
};
```
- `target`: The QQ ID (QQ号) of the user being `@`'d.

### `ImageMessage` / `LocalImageMessage`

An image message. `ImageMessage` uses a URL, while `LocalImageMessage` uses a local file path.

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

A quote-reply message.

```cpp
struct QuoteMessage : public MessageBase {
    std::string text;
    message_id_t ref_msg_id;
    std::optional<qq_id_t> ref_group_id_opt;
    std::optional<qq_id_t> ref_friend_id_opt;
};
```
- `text`: A text preview of the quoted message.
- `ref_msg_id`: The ID of the message being quoted.
- `ref_group_id_opt`: If it's a group message, the group ID where the quoted message is located.
- `ref_friend_id_opt`: If it's a friend message, the friend's QQ ID (QQ号) where the quoted message is located.

### `ForwardMessage`

A forwarded message aggregate.

```cpp
struct ForwardMessage : public MessageBase {
    std::vector<ForwardMessageNode> node_list;
    std::optional<DisplayNode> display;
};
```
- `node_list`: A list of `ForwardMessageNode`, where each node represents one forwarded message.
- `display`: A custom display style for the forwarded message, including a title and summary.

#### `ForwardMessageNode`

A single node within a forwarded message.

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
Contains information about a single forwarded message, including its sender, time, and content.

## Message Properties Model (`msg_prop.h`)

This model is used to extract structured information useful for Large Language Models (LLMs) from raw message events.

### `MessageProperties`

Represents key properties extracted from a message for subsequent processing.

```cpp
struct MessageProperties {
    bool is_at_me;
    std::shared_ptr<std::string> ref_msg_content;
    std::shared_ptr<std::string> plain_content;
    std::set<uint64_t> at_id_set;
};
```

-   `is_at_me`: A boolean indicating if the message `@`'s aibot zihuan itself.
-   `ref_msg_content`: A shared pointer to a string containing the content of a quoted message. It is `nullptr` if no message was quoted.
-   `plain_content`: A shared pointer to a string containing the plain text content of the message (cleaned of information like `@` mentions).
-   `at_id_set`: A set containing the QQ IDs (QQ号) of all users `@`'d in the message. 

## ChatMessage Model (`chat_session.hpp`)

This model represents a single message within a chat session, typically between a user and the llm. It's designed to be compatible with the message format expected by large language models.

### `ChatMessage`

Represents a single message in a chat history.

```cpp
struct ChatMessage {
    std::string role{};
    std::string content{};
    std::optional<std::string> tool_call_id = std::nullopt;
    std::chrono::system_clock::time_point timestamp{std::chrono::system_clock::now()};
    std::optional<std::vector<ToolCall>> tool_calls;
};
```

- `role`: The role of the message sender (e.g., `"user"`, `"assistant"`, `"system"`).
- `content`: The text content of the message.
- `tool_call_id`: If this message is a response from a tool, this field holds the ID of the tool call it corresponds to.
- `timestamp`: The time the message was created.
- `tool_calls`: An optional list of tool calls requested by the model.

### `ToolCall`

Represents a function call requested by the language model.

```cpp
struct ToolCall {
    std::string id;
    std::string arguments;
    std::string name;
    std::string type;
    int64_t index;
};
```

- `id`: A unique identifier for the tool call.
- `arguments`: A JSON string containing the arguments for the function.
- `name`: The name of the function to be called.
- `type`: The type of the tool, typically `"function"`.
- `index`: The index of the tool call.
