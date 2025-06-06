/**
 * @file concurrent_vector.hpp
 * @brief A thread-safe vector implementation with concurrent access support
 */

#pragma once

#include <algorithm>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>
#include <vector>

/**
 * @brief Thread-safe vector container with concurrent access support
 *
 * @tparam T The element type stored in the vector
 * @tparam Allocator The allocator type (default: std::allocator<T>)
 *
 * This implementation provides thread safety guarantees for all operations
 * using a combination of reader-writer locks (shared_mutex) for read operations
 * and exclusive locks (unique_lock) for write operations. The design prioritizes
 * thread safety while maintaining reasonable performance characteristics.
 */
template <typename T, typename Allocator = std::allocator<T>> class concurrent_vector {
  public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = typename std::vector<T, Allocator>::size_type;
    using reference = T &;
    using const_reference = const T &;

    /**
     * @brief Constructs an empty container with default-constructed allocator
     */
    concurrent_vector() = default;

    /**
     * @brief Constructs an empty container with specified allocator
     * @param alloc The allocator to use for all memory allocations
     */
    explicit concurrent_vector(const Allocator &alloc) : m_vec(alloc) {}

    /**
     * @brief Appends the given element to the end of the container
     * @param value The value of the element to append
     */
    void push_back(const T &value) {
        std::unique_lock lock(m_mutex);
        m_vec.push_back(value);
    }

    /**
     * @brief Appends the given element to the end of the container (move version)
     * @param value The value of the element to append
     */
    void push_back(T &&value) {
        std::unique_lock lock(m_mutex);
        m_vec.push_back(std::move(value));
    }

    /**
     * @brief Accesses specified element with bounds checking
     * @param pos Position of the element to return
     * @return std::optional containing the element if pos is valid
     */
    std::optional<T> at(size_type pos) const {
        std::shared_lock lock(m_mutex);
        if (pos >= m_vec.size()) {
            return std::nullopt;
        }
        return m_vec[pos];
    }

    /**
     * @brief Accesses specified element without bounds checking
     * @param pos Position of the element to return
     * @return std::optional containing the element if pos is valid
     */
    std::optional<T> operator[](size_type pos) const {
        std::shared_lock lock(m_mutex);
        if (pos >= m_vec.size()) {
            return std::nullopt;
        }
        return m_vec[pos];
    }

    /**
     * @brief Returns the number of elements in the container
     */
    size_type size() const {
        std::shared_lock lock(m_mutex);
        return m_vec.size();
    }

    /**
     * @brief Checks if the container is empty
     */
    bool empty() const {
        std::shared_lock lock(m_mutex);
        return m_vec.empty();
    }

    /**
     * @brief Erases all elements from the container
     */
    void clear() {
        std::unique_lock lock(m_mutex);
        m_vec.clear();
    }

    /**
     * @brief Reserves storage for at least the specified number of elements
     * @param new_cap New capacity of the vector
     */
    void reserve(size_type new_cap) {
        std::unique_lock lock(m_mutex);
        m_vec.reserve(new_cap);
    }

    /**
     * @brief Returns the number of elements that can be held in allocated storage
     */
    size_type capacity() const {
        std::shared_lock lock(m_mutex);
        return m_vec.capacity();
    }

    /**
     * @brief Reduces memory usage by freeing unused memory
     */
    void shrink_to_fit() {
        std::unique_lock lock(m_mutex);
        m_vec.shrink_to_fit();
    }

    /**
     * @brief Applies given function to each element (read-only)
     * @tparam Func Callable type accepting (const_reference) parameter
     * @param func The function to apply
     */
    template <typename Func> void for_each(Func func) const {
        std::shared_lock lock(m_mutex);
        for (const auto &element : m_vec) {
            func(element);
        }
    }

    /**
     * @brief Applies given function to each element (allows modification)
     * @tparam Func Callable type accepting (reference) parameter
     * @param func The function to apply
     */
    template <typename Func> void modify(Func func) {
        std::unique_lock lock(m_mutex);
        for (auto &element : m_vec) {
            func(element);
        }
    }

    /**
     * @brief Provides thread-safe access to the underlying container (read-only)
     * @tparam Func Callable type accepting const reference to the underlying vector
     * @return Result of the function invocation
     */
    template <typename Func> auto access(Func func) const {
        std::shared_lock lock(m_mutex);
        return func(m_vec);
    }

    /**
     * @brief Provides thread-safe access to the underlying container (read-write)
     * @tparam Func Callable type accepting reference to the underlying vector
     * @return Result of the function invocation
     */
    template <typename Func> auto modify_vector(Func func) {
        std::unique_lock lock(m_mutex);
        return func(m_vec);
    }

    /**
     * @brief Returns the allocator associated with the container
     */
    allocator_type get_allocator() const noexcept { return m_vec.get_allocator(); }

    /**
     * @brief Finds the first element satisfying specific criteria
     * @tparam Predicate Type of the predicate function
     * @param pred Predicate function which returns true for the required element
     * @return std::optional containing the first matching element if found
     */
    template <typename Predicate> std::optional<T> find_if(Predicate pred) const {
        std::shared_lock lock(m_mutex);
        auto it = std::find_if(m_vec.begin(), m_vec.end(), pred);
        if (it != m_vec.end()) {
            return *it;
        }
        return std::nullopt;
    }

  private:
    std::vector<T, Allocator> m_vec;
    mutable std::shared_mutex m_mutex;
};