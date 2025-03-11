#ifndef MUTEX_DATA_H
#define MUTEX_DATA_H

#include <mutex>
#include <shared_mutex>

template <typename T> class MutexData {
  public:
    class ReadLock {
      public:
        ReadLock(const T &data, std::shared_lock<std::shared_mutex> lock) : m_data(data), m_lock(std::move(lock)) {}

        const T &operator*() const { return m_data; }
        const T *operator->() const { return &m_data; }

      private:
        const T &m_data;
        std::shared_lock<std::shared_mutex> m_lock;
    };

    class WriteLock {
      public:
        WriteLock(T &data, std::unique_lock<std::shared_mutex> lock) : m_data(data), m_lock(std::move(lock)) {}

        T &operator*() { return m_data; }
        T *operator->() { return &m_data; }

      private:
        T &m_data;
        std::unique_lock<std::shared_mutex> m_lock;
    };

    template <typename... Args> explicit MutexData(Args &&...args) : m_data(std::forward<Args>(args)...) {}

    ReadLock read() {
        std::shared_lock lock(m_mutex);
        return ReadLock(m_data, std::move(lock));
    }

    WriteLock write() {
        std::unique_lock lock(m_mutex);
        return WriteLock(m_data, std::move(lock));
    }

  private:
    T m_data;
    mutable std::shared_mutex m_mutex;
};

#endif