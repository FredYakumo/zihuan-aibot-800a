/**
 * @file concurrent_unordered_map.hpp
 * @brief A thread-safe unordered map
 */

#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

/**
 * @brief Thread-safe unordered map container with concurrent access support
 *
 * @tparam Key The key type of the map
 * @tparam Value The mapped type of the map
 * @tparam Hash The hash function object type (default: std::hash<Key>)
 * @tparam KeyEqual The key equality comparison function object type (default: std::equal_to<Key>)
 * @tparam Allocator The allocator type (default: std::allocator<std::pair<const Key, Value>>)
 *
 * This implementation provides basic thread safety guarantees for all operations
 * using reader-writer locks (shared_mutex). Read operations can execute concurrently,
 * while write operations require exclusive access.
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
class concurrent_unordered_map {
  public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<const Key, Value>;
    using size_type = typename std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>::size_type;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = Allocator;

    /**
     * @brief Constructs an empty container with default-constructed allocator
     */
    concurrent_unordered_map() = default;

    /**
     * @brief Constructs an empty container with specified allocator
     * @param alloc The allocator to use for all memory allocations
     */
    explicit concurrent_unordered_map(const Allocator &alloc) : m_map(alloc) {}

    /**
     * @brief Inserts or updates an element in the container
     * @param key The key of the element to insert/update
     * @param value The value to associate with the key
     * @return A reference wrapper to the inserted or updated value
     */
    std::reference_wrapper<Value> insert_or_assign(const Key &key, const Value &value) {
        std::unique_lock lock(m_mutex);
        auto [it, _] = m_map.insert_or_assign(key, value);
        return std::ref(it->second);
    }

    /**
     * @brief Attempts to insert an element if the key doesn't exist
     * @return true if insertion succeeded, false if key already exists
     */
    bool try_insert(const Key &key, const Value &value) {
        std::unique_lock lock(m_mutex);
        return m_map.try_emplace(key, value).second;
    }

    /**
     * @brief Finds an element with given key
     * @return std::optional containing a reference to the value if found, empty otherwise
     * @warning The returned reference is only valid while the caller maintains appropriate synchronization
     */
    std::optional<std::reference_wrapper<Value>> find(const Key &key) const {
        std::shared_lock lock(m_mutex);
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            return std::ref(it->second);
        }
        return std::nullopt;
    }

    // std::optional<std::reference_wrapper<Value>> try_update_value(const Key &key) const {

    // }

    /**
     * @brief Removes the element with given key
     * @return true if element was removed, false if key didn't exist
     */
    bool erase(const Key &key) {
        std::unique_lock lock(m_mutex);
        return m_map.erase(key) > 0;
    }

    /**
     * @brief Checks if container contains element with specific key
     */
    bool contains(const Key &key) const {
        std::shared_lock lock(m_mutex);
        return m_map.contains(key);
    }

    /**
     * @brief Returns the number of elements in the container
     */
    size_type size() const {
        std::shared_lock lock(m_mutex);
        return m_map.size();
    }

    /**
     * @brief Checks if the container is empty
     */
    bool empty() const {
        std::shared_lock lock(m_mutex);
        return m_map.empty();
    }

    /**
     * @brief Erases all elements from the container
     */
    void clear() {
        std::unique_lock lock(m_mutex);
        m_map.clear();
    }

    /**
     * @brief Applies given function to each element (read-only)
     * @tparam Func Callable type accepting (key, value) parameters
     * @param func The function to apply
     */
    template <typename Func> void for_each(Func func) const {
        std::shared_lock lock(m_mutex);
        for (const auto &[key, value] : m_map) {
            func(key, value);
        }
    }

    /**
     * @brief Applies given function to each element (allows modification)
     * @tparam Func Callable type accepting (key, value) parameters
     * @param func The function to apply
     */
    template <typename Func> void modify(Func func) {
        std::unique_lock lock(m_mutex);
        for (auto &[key, value] : m_map) {
            func(key, value);
        }
    }

    /**
     * @brief Provides thread-safe access to the underlying container (read-only)
     * @tparam Func Callable type accepting const reference to the underlying map
     * @return Result of the function invocation
     */
    template <typename Func> auto access(Func func) const {
        std::shared_lock lock(m_mutex);
        return func(m_map);
    }

    /**
     * @brief Provides thread-safe access to the underlying container (read-write)
     * @tparam Func Callable type accepting reference to the underlying map
     * @return Result of the function invocation
     */
    template <typename Func> auto modify_map(Func func) {
        std::unique_lock lock(m_mutex);
        return func(m_map);
    }

    /**
     * @brief Returns the allocator associated with the container
     */
    allocator_type get_allocator() const noexcept { return m_map.get_allocator(); }

  private:
    std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> m_map;
    mutable std::shared_mutex m_mutex;
};