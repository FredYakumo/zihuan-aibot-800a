# 工具函数API参考

本文档提供了AIBot Zihuan项目中使用的工具函数和类的参考。这些工具包括时间格式化、字符串操作、线程管理和其他辅助函数。

## 字符串工具 (string_utils.h)

提供字符串处理的常用功能。

### `trim`
```cpp
/**
 * @brief 从字符串中移除前导和尾随空白字符
 * @param s 要修剪的输入字符串
 * @return 修剪后的字符串
 */
std::string trim(const std::string& s);
```

### `split`
```cpp
/**
 * @brief 使用分隔符将字符串分割为子字符串向量
 * @param s 要分割的输入字符串
 * @param delimiter 用作分隔符的字符
 * @return 子字符串向量
 */
std::vector<std::string> split(const std::string& s, char delimiter);
```

### `to_lower`
```cpp
/**
 * @brief 将字符串中所有字符转换为小写
 * @param s 要转换的输入字符串
 * @return 小写字符串
 */
std::string to_lower(const std::string& s);
```

## 时间工具 (time_utils.h)

提供时间格式化和转换功能。

### `get_current_time_str`
```cpp
/**
 * @brief 获取当前时间并格式化为字符串
 * @param format 格式字符串（默认："%Y-%m-%d %H:%M:%S"）
 * @return 格式化的时间字符串
 */
std::string get_current_time_str(const std::string& format = "%Y-%m-%d %H:%M:%S");
```

### `timestamp_to_string`
```cpp
/**
 * @brief 将时间戳转换为格式化字符串
 * @param timestamp Unix时间戳（秒）
 * @param format 格式字符串
 * @return 格式化的时间字符串
 */
std::string timestamp_to_string(time_t timestamp, const std::string& format);
```

## 线程工具 (thread_utils.h)

提供线程管理和同步功能。

### `ThreadPool` 类
```cpp
/**
 * @brief 简单的线程池实现
 */
class ThreadPool {
public:
    /**
 * @brief 构造新的线程池对象
 * @param num_threads 池中线程数量
 */
    explicit ThreadPool(size_t num_threads);

    /**
 * @brief 向线程池添加任务
 * @tparam F 函数类型
 * @tparam Args 参数类型
 * @param f 要执行的函数
 * @param args 传递给函数的参数
 * @return std::future<typename std::result_of<F(Args...)>::type>
 */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    /**
 * @brief 销毁线程池对象
 */
    ~ThreadPool();
};
```

## 文件工具 (file_utils.h)

提供文件操作的辅助功能。

### `read_file_content`
```cpp
/**
 * @brief 读取文件的全部内容到字符串中
 * @param file_path 文件路径
 * @return 文件内容字符串，失败时返回空字符串
 */
std::string read_file_content(const std::string& file_path);
```

### `write_file_content`
```cpp
/**
 * @brief 将内容写入文件
 * @param file_path 文件路径
 * @param content 要写入的内容
 * @param append 是否追加到文件（默认：false）
 * @return 成功时返回true，否则返回false
 */
bool write_file_content(const std::string& file_path, const std::string& content, bool append = false);
```

## 日志工具 (log_utils.h)

基于spdlog的日志工具封装。

### `init_logger`
```cpp
/**
 * @brief 使用默认设置初始化日志器
 * @param log_file 日志文件路径
 * @param level 日志级别（默认：spdlog::level::info）
 */
void init_logger(const std::string& log_file, spdlog::level::level_enum level = spdlog::level::info);
```

### `get_logger`
```cpp
/**
 * @brief 获取主日志器实例
 * @return 日志器的引用
 */
spdlog::logger& get_logger();
```

## 跨平台工具 (cross_platform_utils.h)

提供跨平台兼容的功能实现。

### `get_current_directory`
```cpp
/**
 * @brief 获取当前工作目录
 * @return 当前目录路径
 */
std::string get_current_directory();
```

### `create_directory`
```cpp
/**
 * @brief 创建目录
 * @param dir_path 要创建的目录路径
 * @return 成功或目录已存在时返回true，否则返回false
 */
bool create_directory(const std::string& dir_path);
```