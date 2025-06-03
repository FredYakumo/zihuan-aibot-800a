#ifndef GLOBAL_DATA_H
#define GLOBAL_DATA_H

#include "msg_prop.h"
#include "mutex_data.hpp"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mutex_data.hpp"
#include "chat_session.hpp"
#include "db_knowledge.hpp"
#include <shared_mutex>

constexpr size_t USER_SESSION_MSG_LIMIT = 60000;
constexpr size_t MAX_KNOWLEDGE_LENGTH = 4096;

extern MutexData<std::unordered_map<uint64_t, ChatSession>> g_chat_session_map;
extern MutexData<std::unordered_map<uint64_t, std::set<std::string>>> g_chat_session_knowledge_list_map;
extern std::pair<std::mutex, std::unordered_map<uint64_t, bool>> g_chat_processing_map;
extern MutexData<std::vector<DBKnowledge>> g_wait_add_knowledge_list;


/**
 * @class IndividualMessageIdStorage
 * @brief 存储对象消息ID和属性的线程安全容器 / Thread-safe container for storing individual message IDs and properties
 */
 class IndividualMessageIdStorage {
    private:
        mutable std::shared_mutex mutex_;  // 读写锁，用于线程安全 / Read-write lock for thread safety
        // 嵌套的unordered_map结构: [群组ID -> [消息ID -> 消息属性]] / Nested unordered_map: [group ID -> [message ID -> message properties]]
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, MessageProperties>> message_storage;
    
    public:
        /**
         * @brief 插入消息到存储中 / Insert message into storage
         * @param group_id 群组ID / Group ID
         * @param message_id 消息ID / Message ID
         * @param msg_prop 消息属性 / Message properties
         */
        void insert_message(uint64_t group_id, uint64_t message_id, MessageProperties msg_prop) {
            std::unique_lock lock(mutex_);  // 获取独占锁 / Acquire exclusive lock
            message_storage[group_id][message_id] = std::move(msg_prop);
        }
    
        /**
         * @brief 获取指定消息 / Get specific message
         * @param individual_id 对象ID / Individual ID
         * @param message_id 消息ID / Message ID
         * @return 消息属性(如果存在) / Message properties (if exists)
         */
        std::optional<MessageProperties> get_message(uint64_t individual_id, uint64_t message_id) const {
            std::shared_lock lock(mutex_);  // 获取共享锁 / Acquire shared lock
            auto group_it = message_storage.find(individual_id);
            if (group_it == message_storage.end()) {
                return std::nullopt;
            }
            auto msg_it = group_it->second.find(message_id);
            if (msg_it == group_it->second.end()) {
                return std::nullopt;
            }
            return msg_it->second;
        }
    
        /**
         * @brief 删除指定消息 / Remove specific message
         * @param group_id 群组ID / Group ID
         * @param message_id 消息ID / Message ID
         * @return 是否成功删除 / Whether removal was successful
         */
        bool remove_message(uint64_t group_id, uint64_t message_id) {
            std::unique_lock lock(mutex_);  // 获取独占锁 / Acquire exclusive lock
            auto group_it = message_storage.find(group_id);
            if (group_it == message_storage.end()) {
                return false;
            }
            return group_it->second.erase(message_id) > 0;
        }
    
        /**
         * @brief 获取对象所有消息的指针 / Get pointer to all messages of an individual
         * @param individual_id 对象ID / Individual ID
         * @return 消息映射的指针(如果存在) / Pointer to message map (if exists)
         */
        std::optional<const std::unordered_map<uint64_t, MessageProperties>*> get_individual_messages(uint64_t individual_id) const {
            std::shared_lock lock(mutex_);  // 获取共享锁 / Acquire shared lock
            auto it = message_storage.find(individual_id);
            if (it == message_storage.cend()) {
                return std::nullopt;
            }
            return &(it->second);
        }
    
        /**
         * @brief 清除对象所有消息 / Clear all messages of an individual
         * @param individual_id individual ID / Individual ID
         * @return 是否成功清除 / Whether clearance was successful
         */
        bool clear_individual_messages(uint64_t individual_id) {
            std::unique_lock lock(mutex_);  // 获取独占锁 / Acquire exclusive lock
            return message_storage.erase(individual_id) > 0;
        }
    };


void insert_group_message(uint64_t group_id, uint64_t message_id, MessageProperties msg_prop);

std::optional<ChatSession> get_user_chat_session(uint64_t chat_session);

#endif