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

#endif