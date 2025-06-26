# Bot Command Reference

This document provides a reference for the `bot_cmd` namespace, which is responsible for handling user-issued commands in AIBot Zihuan. Commands are special messages that trigger specific, predefined actions.

## Overview

The command system allows users to interact with the bot's specialized functions, such as managing conversation history, searching its knowledge base, or performing real-time web searches. Commands are typically identified by a keyword, often prefixed with `#` (e.g., `#新对话`).

The core logic resides in the `bot_cmd` namespace and is initialized by `init_command_map()`. When a message is received, the system checks if it matches any registered command keywords. If a match is found, the corresponding command handler function is executed.

## Core Components

These are the fundamental data structures used by the command system.

### `CommandContext` Struct

This struct encapsulates all the necessary information for a command's execution. It is passed to every command handler function.

```cpp
struct CommandContext {
    bot_adapter::BotAdapter &adapter;
    std::shared_ptr<bot_adapter::MessageEvent> event;
    std::string param;
    bool is_deep_think = false;
    MessageProperties msg_prop;
};
```
-   `adapter`: A reference to the `BotAdapter` instance, used for sending replies and other bot actions.
-   `event`: A shared pointer to the `MessageEvent` that triggered the command. Contains sender information, message content, etc.
-   `param`: The parameter string for the command. This is the part of the message that follows the command keyword.
-   `is_deep_think`: A boolean flag indicating if the bot is in "deep think" mode.
-   `msg_prop`: An object containing parsed properties of the message.

### `CommandRes` Struct

This struct represents the result of a command's execution, signaling to the main message processing loop how to proceed.

```cpp
struct CommandRes {
    bool interrupt_following_commands;
    bool skip_default_process_llm;
    bool is_deep_think = false;
    std::optional<std::function<void(const MessageProperties &)>> is_modify_msg;
};
```
-   `interrupt_following_commands`: If `true`, no other commands will be processed for this message.
-   `skip_default_process_llm`: If `true`, the default LLM processing step that normally follows command execution will be skipped.
-   `is_deep_think`: If set to `true`, it tells the system to engage a more intensive "deep think" mode for its response.
-   `is_modify_msg`: An optional function to modify message properties after command execution.

### `CommandProperty` Struct

This struct defines the properties of a single command.

```cpp
struct CommandProperty {
    bool is_need_admin;
    bool is_need_param;
    std::function<CommandRes(CommandContext)> runer;
};
```
-   `is_need_admin`: If `true`, the command can only be executed by a user with admin privileges.
-   `is_need_param`: If `true`, the command requires a parameter.
-   `runer`: The function that implements the command's logic. It takes a `CommandContext` and returns a `CommandRes`.

## Command Registration

Commands are registered in a global map.

-   `keyword_command_map`: A `std::vector<std::pair<std::string, bot_cmd::CommandProperty>>` that maps command keywords (e.g., `"#新对话"`) to their `CommandProperty`.
-   `init_command_map()`: This function populates the `keyword_command_map`. It is called once at application startup to register all available commands.

## Available Commands

Here is a list of default commands and their functions.

#### `#新对话` (`clear_chat_session_command`)
Clears the conversation history for the current user. This effectively starts a new session with the bot, making it forget the previous context of the chat. If the command is sent with additional text, the bot will clear the context and then respond to the new text.
-   **Admin Required**: No
-   **Parameter Required**: No

#### `#思考` (`deep_think_command`)
Triggers "deep think" mode for the current request. This can result in a more detailed or comprehensive response from the LLM, potentially at the cost of longer processing time.
-   **Admin Required**: No
-   **Parameter Required**: No

#### `#查询知识` (`query_knowledge_command`)
Searches the bot's long-term knowledge base for information matching the provided query parameter.
-   **Admin Required**: Yes
-   **Parameter Required**: Yes
-   **Usage**: `#查询知识 <query_text>`

#### `#添加知识` (`add_knowledge_command`)
Adds a piece of text to a temporary waiting list to be added to the knowledge base later. This allows users to contribute knowledge, which an admin can review and approve.
-   **Admin Required**: No
-   **Parameter Required**: Yes
-   **Usage**: `#添加知识 <text_to_add>`
> **Note**: The implementation for this command is currently disabled in `src/bot_cmd.cpp`.

#### `#入库知识` (`checkin_knowledge_command`)
Approves and moves a specific piece of knowledge from the waiting list into the permanent knowledge base. The item is identified by its index in the waiting list.
-   **Admin Required**: Yes
-   **Parameter Required**: Yes
-   **Usage**: `#入库知识 <index>`

#### `#查看记忆` (`query_memory_command`)
Displays the bot's current short-term memory, showing recent conversation histories for all active users. This is useful for debugging and understanding the bot's current state.
-   **Admin Required**: Yes
-   **Parameter Required**: No

#### `#待添加知识` (`query_add_knowledge_list_command`)
Shows the list of knowledge items that are currently in the waiting list and have not yet been approved.
-   **Admin Required**: Yes
-   **Parameter Required**: No

#### `#联网` (`net_search_command`)
Performs a real-time internet search based on the provided text or the content of a replied-to message. The search results are used as additional context for the LLM to generate a response.
-   **Admin Required**: No
-   **Parameter Required**: No (uses message content)

#### `#url` (`url_search_command`)
Fetches the content from one or more specified URLs. The extracted text content is then used as context for the LLM's response. Multiple URLs can be provided, separated by commas.
-   **Admin Required**: No
-   **Parameter Required**: Yes
-   **Usage**: `#url (<url1>,<url2>,...)`
