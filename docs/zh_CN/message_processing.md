# 消息处理

本文档描述了AIBot Zihuan中的消息处理工作流程，包括事件处理、数据处理和响应生成。

## 概述

消息处理管道包括三个主要阶段：
1. 事件接收和路由
2. 消息内容提取和验证
3. 基于LLM的响应生成
4. 响应格式化和交付

## 事件处理

### 事件类型

系统处理两种主要的消息事件类型：

- `GroupMessageEvent`：在QQ群中收到消息时触发
- `FriendMessageEvent`：收到好友私信时触发

这两种事件都继承自基类`MessageEvent`，并包含：
- 发送者信息（ID、昵称、权限）
- 消息内容（文本、图片和其他媒体）
- 时间戳和元数据

### 事件处理函数

事件处理在`event.cpp`中实现，包含以下函数：

#### `on_group_msg_event`
```cpp
void on_group_msg_event(std::shared_ptr<GroupMessageEvent> event);
```
处理传入的群消息。验证权限、提取消息内容并启动响应生成。

#### `on_friend_msg_event`
```cpp
void on_friend_msg_event(std::shared_ptr<FriendMessageEvent> event);
```
处理来自好友的私信。与群消息处理类似，但权限检查更简单。

## 数据处理流程

1. **消息提取**：从事件对象中提取原始消息数据
2. **内容验证**：检查空消息、无效内容或命令前缀
3. **上下文检索**：从数据库中获取相关对话历史和用户资料
4. **输入准备**：格式化消息和上下文以进行LLM处理
5. **LLM推理**：将准备好的输入发送到LLM模型
6. **响应处理**：处理LLM输出并生成适当的消息链
7. **交付**：将格式化的响应发送回源（群或好友）

## LLM集成

核心消息处理逻辑在`llm.cpp`中实现，提供以下关键函数：

### `process_message`
```cpp
std::string process_message(const std::string& user_message, const std::string& context);
```
使用LLM处理用户消息的主函数。接收用户消息和对话上下文作为输入，并返回生成的响应。

### `generate_response`
```cpp
MessageChainPtrList generate_response(const std::string& llm_output);
```
将原始LLM输出转换为适合通过机器人适配器发送的格式化`MessageChainPtrList`。

## 错误处理

所有消息处理都包含强大的错误处理和spdlog日志记录：
- 无效的消息格式
- LLM推理失败
- 响应传递过程中的网络错误
- 数据库连接问题

## 性能考虑

- 消息处理是异步的，以避免阻塞主事件循环
- 长时间运行的LLM推理操作被卸载到工作线程
- 对话上下文被缓存以减少数据库负载

## 示例工作流程

1. 用户在群中发送消息："你好，今天天气怎么样？"
2. `GroupMessageEvent`被触发并路由到`on_group_msg_event`
3. 提取并验证消息文本
4. 检索相关上下文（之前的对话）
5. 调用`llm.cpp::process_message`处理消息和上下文
6. LLM生成天气响应
7. `llm.cpp::generate_response`将文本转换为消息链
8. `BotAdapter::send_group_message`交付响应