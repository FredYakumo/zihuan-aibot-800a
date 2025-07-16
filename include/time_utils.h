#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <iomanip>
#include <string>
#include <chrono>

inline std::string system_clock_to_string(const std::chrono::system_clock::time_point &time) {
    // Convert time_point to time_t
    auto time_t = std::chrono::system_clock::to_time_t(time);

    // Convert to local time (or UTC if preferred)
    std::tm tm = *std::localtime(&time_t);

    // Format as YYYY年MM月dd日 HH:mm:SS
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y年%m月%d日 %H:%M:%S");
    std::string time_str = oss.str();
    return time_str;
}

inline std::chrono::system_clock::time_point timestamp_to_timepoint(uint64_t timestamp) {
    return std::chrono::system_clock::from_time_t(timestamp);
}

/**
 * @brief Convert database time string (format: "%Y-%m-%d %H:%M:%S") to std::chrono::system_clock::time_point.
 * @param db_time_str Database time string.
 * @return time_point corresponding to the input string.
 */
inline std::chrono::system_clock::time_point db_str_to_time_point(const std::string &db_time_str) {
    std::tm tm = {};
    std::istringstream ss(db_time_str);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse time string: " + db_time_str);
    }
    std::time_t time_c = std::mktime(&tm);
    return std::chrono::system_clock::from_time_t(time_c);
}

#endif