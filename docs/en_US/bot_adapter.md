# Bot Adapter API Reference

This document provides a reference for the `bot_adapter` library, which facilitates communication with a QQ Bot platform (like Mirai or OneBot) over WebSockets. The primary component is the `BotAdapter` class.

## Definitions
- **QQ Bot**: The entity of the Mirai or OneBot instance. The QQ Bot is responsible for controlling the lifecycle of a QQ account.
- **zihuan-aibot-800a**: This project, an intelligent agent that uses Large Language Models (LLMs), other models, and databases. It realizes its capabilities by interacting with a `QQ Bot`. Hereinafter referred to as AIBot Zihuan.

## `BotAdapter` Class

The `BotAdapter` class is the main entry point for interaction between AIBot Zihuan and a QQ Bot. It manages the WebSocket connection, handles incoming events, and provides methods for sending messages and retrieving information.

### Lifecycle

#### Constructor

```cpp
BotAdapter::BotAdapter(const std::string_view url, std::optional<uint64_t> bot_id_option = std::nullopt);
```

Creates a new `BotAdapter` instance.

-   `url`: The WebSocket URL of the QQ Bot server.
-   `bot_id_option`: An optional QQ ID (QQ号) for the bot. If provided, it is stored in the bot's profile.

#### Destructor

```cpp
BotAdapter::~BotAdapter();
```

Cleans up resources, including the WebSocket connection.

#### `start`

```cpp
int BotAdapter::start();
```

Starts the AIBot Zihuan adapter. This method enters a blocking loop that polls for incoming messages from the WebSocket server and dispatches them. It also initiates a background thread to periodically update group information. This function will run until the WebSocket connection is closed.

### Event Handling

#### `register_event`

```cpp
template <typename EventFuncT>
inline void BotAdapter::register_event(std::function<void(std::shared_ptr<EventFuncT> e)> func);
```

Registers a handler function for a specific event type. This is the primary mechanism for reacting to events like incoming messages.

-   `EventFuncT`: The type of event to listen for. Must be a class derived from `Event`. Examples include `GroupMessageEvent` and `FriendMessageEvent`.
-   `func`: A function (or lambda) that takes a `std::shared_ptr` to the event object.

**Example:**

```cpp
bot.register_event<GroupMessageEvent>([](std::shared_ptr<GroupMessageEvent> e) {
    spdlog::info("Received group message: {}", e->to_json().dump());
    // ... handle the group message
});
```

### Message Sending

These functions are used to send messages to friends or groups.

#### `send_message`

```cpp
void BotAdapter::send_message(
    const Sender &sender,
    const MessageChainPtrList &message_chain,
    std::optional<std::string_view> sync_id_option = std::nullopt,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

Sends a message to a friend. This is the fundamental function for sending friend messages.

-   `sender`: The friend to send the message to.
-   `message_chain`: A list of message components (text, images, etc.) that form the message.
-   `sync_id_option`: Optional custom sync ID for tracking the command.
-   `out_message_id_option`: Optional callback to receive the ID of the sent message.

#### `send_group_message`

```cpp
void BotAdapter::send_group_message(
    const Group &group,
    const MessageChainPtrList &message_chain,
    std::optional<std::string_view> sync_id_option = std::nullopt,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

Sends a message to a group. This is the fundamental function for sending group messages.

-   `group`: The target group.
-   `message_chain`: The message content.
-   `sync_id_option`: Optional custom sync ID.
-   `out_message_id_option`: Optional callback to receive the ID of the sent message.

#### `send_replay_msg`

```cpp
void BotAdapter::send_replay_msg(
    const Sender &sender,
    const MessageChainPtrList &message_chain,
    bool at_target = true,
    std::optional<std::function<void(uint64_t &out_message_id)>> out_message_id_option = std::nullopt);
```

A convenience function to reply to a received message. It automatically handles whether the original message was from a group or a friend.

-   `sender`: The person to reply to.
-   `message_chain`: The reply content.
-   `at_target`: If `true` and the message is in a group, it will `@` the person.

#### `send_long_plain_text_reply`

```cpp
void BotAdapter::send_long_plain_text_reply(
    const Sender &sender,
    std::string text,
    bool at_target = true,
    uint64_t msg_length_limit = MAX_OUTPUT_LENGTH);
```

Sends a long text reply, with advanced formatting capabilities. The function will parse the input `text` as Markdown.

-   **Markdown Parsing**: It can handle complex markdown structures.
-   **Image Rendering**: For elements like code blocks, tables, or rich text, it calls an external service to render them as images, which are then sent. This avoids character limits and formatting issues in plain text messages.
-   **Message Splitting**: If the text is too long, it is split into multiple parts.
-   **Forwarded Message**: The final output is typically sent as a single "forwarded message" to keep the content grouped together.

This function is ideal for sending complex, formatted responses from services like LLMs.

### Information Retrieval

These methods are for fetching data about the bot, users, and groups.

#### `update_bot_profile`

```cpp
void BotAdapter::update_bot_profile();
```

Asynchronously fetches the QQ Bot's own profile information (i.e., the QQ account's profile, like nickname, email, etc.) from the server and updates the local `bot_profile` object.

#### `get_bot_profile`

```cpp
const Profile &BotAdapter::get_bot_profile() const;
```

Returns a constant reference to the locally cached bot profile information (of the QQ account).

#### `update_group_info_sync`

```cpp
void BotAdapter::update_group_info_sync();
```

Synchronously fetches a list of all groups the QQ Bot is a member of, and for each group, fetches its complete member list. This is a potentially long-running, blocking operation that populates an internal cache. The `start()` method runs this periodically in a background thread.

#### `fetch_group_member_info`

```cpp
inline std::optional<std::reference_wrapper<const GroupWrapper>>
BotAdapter::fetch_group_member_info(qq_id_t group_id) const;
```

Fetches information about a specific group (including its member list) from the local cache. The cache is populated by `update_group_info_sync()`.

-   Returns a `GroupWrapper` object if the group is found in the cache, otherwise `std::nullopt`.

#### `get_group`

```cpp
inline const GroupWrapper &BotAdapter::get_group(qq_id_t group_id) const;
```

Similar to `fetch_group_member_info`, but returns a direct reference to the `GroupWrapper`. Throws an exception if the group is not found in the cache.

#### `get_message_id`

```cpp
inline void BotAdapter::get_message_id(
    uint64_t message_id,
    uint64_t target_id,
    CommandResHandleFunc out_func);
```

Asynchronously fetches a message by its ID.

-   `message_id`: The ID of the message to fetch.
-   `target_id`: The ID of the friend or group where the message is located.
-   `out_func`: A callback function that will be invoked with the result.

#### `get_group_announcement`

```cpp
inline void BotAdapter::get_group_announcement(
    qq_id_t group_id,
    GroupAnnouncementResHandleFunc out_func,
    int offset = 0,
    int size = 10);
```

Asynchronously fetches a list of announcements for a given group.

-   `group_id`: The target group's ID.
-   `out_func`: A callback to handle the list of `GroupAnnouncement`.
-   `offset`: The starting offset for pagination.
-   `size`: The number of announcements to fetch.

#### `get_group_announcement_sync`

```cpp
inline std::optional<std::vector<GroupAnnouncement>>
BotAdapter::get_group_announcement_sync(
    qq_id_t group_id,
    int offset = 0,
    int size = 10,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(20000));
```

Synchronously fetches group announcements. This function blocks until a response is received or a timeout occurs.

-   Returns a vector of `GroupAnnouncement` on success, or `std::nullopt` on failure or timeout.
