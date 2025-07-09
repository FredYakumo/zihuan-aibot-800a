# Message Processing

This document describes the message processing workflow in AIBot Zihuan, including event handling, data processing, and response generation.

## Overview

The message processing pipeline consists of three main stages:
1. Event reception and routing
2. Message content extraction and validation
3. LLM-based response generation
4. Response formatting and delivery

## Event Handling

### Event Types

Two primary message event types are processed by the system:

- `GroupMessageEvent`: Triggered when a message is received in a QQ group
- `FriendMessageEvent`: Triggered when a private message is received from a friend

Both events inherit from the base `MessageEvent` class and contain:
- Sender information (ID, nickname, permissions)
- Message content (text, images, and other media)
- Timestamp and metadata

### Event Processing Functions

Event handling is implemented in `event.cpp` with the following functions:

#### `on_group_msg_event`
```cpp
void on_group_msg_event(std::shared_ptr<GroupMessageEvent> event);
```
Processes incoming group messages. Validates permissions, extracts message content, and initiates response generation.

#### `on_friend_msg_event`
```cpp
void on_friend_msg_event(std::shared_ptr<FriendMessageEvent> event);
```
Handles private messages from friends. Similar to group message processing but with simplified permission checks.

## Data Processing Flow

1. **Message Extraction**: Raw message data is extracted from the event object
2. **Content Validation**: Checks for empty messages, invalid content, or command prefixes
3. **Context Retrieval**: Fetches relevant conversation history and user profile from the database
4. **Input Preparation**: Formats the message and context for LLM processing
5. **LLM Inference**: Sends prepared input to the LLM model
6. **Response Handling**: Processes LLM output and generates appropriate message chain
7. **Delivery**: Sends the formatted response back to the source (group or friend)

## LLM Integration

Core message processing logic is implemented in `llm.cpp`, which provides the following key functions:

### `process_message`
```cpp
std::string process_message(const std::string& user_message, const std::string& context);
```
Main function for processing user messages with the LLM. Takes the user message and conversation context as input and returns the generated response.

### `generate_response`
```cpp
MessageChainPtrList generate_response(const std::string& llm_output);
```
Converts raw LLM output into a formatted `MessageChainPtrList` suitable for sending via the bot adapter.

## Error Handling

All message processing includes robust error handling with spdlog logging:
- Invalid message formats
- LLM inference failures
- Network errors during response delivery
- Database connection issues

## Performance Considerations

- Message processing is asynchronous to avoid blocking the main event loop
- Long-running LLM inference operations are offloaded to worker threads
- Conversation context is cached to reduce database load

## Example Workflow

1. User sends message in a group: "Hello, what's the weather today?"
2. `GroupMessageEvent` is triggered and routed to `on_group_msg_event`
3. Message text is extracted and validated
4. Relevant context (previous conversation) is retrieved
5. `llm.cpp::process_message` is called with message and context
6. LLM generates weather response
7. `llm.cpp::generate_response` converts text to message chain
8. `BotAdapter::send_group_message` delivers the response