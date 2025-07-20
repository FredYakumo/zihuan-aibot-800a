#pragma once

#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/base_sink.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

namespace spdlog {
namespace sinks {

// Custom sink that combines daily rotation with file size rotation
// File rotation behavior:
// - Current logs: aibot_800a_2025-07-20.txt (newest)
// - When size limit reached: current -> highest available number
// - .1 = oldest archived, .2, .3... = progressively newer, highest number = newest archived
template<typename Mutex>
class daily_rotating_file_sink : public base_sink<Mutex> {
public:
    daily_rotating_file_sink(filename_t base_filename, std::size_t max_size, int rotation_hour = 0, int rotation_minute = 0, bool enable_daily_rotation = true)
        : base_filename_(std::move(base_filename))
        , max_size_(max_size)
        , rotation_hour_(rotation_hour)
        , rotation_minute_(rotation_minute)
        , enable_daily_rotation_(enable_daily_rotation)
        , current_size_(0) {
        
        auto now = std::chrono::system_clock::now();
        if (enable_daily_rotation_) {
            rotation_tp_ = next_rotation_tp_(now);
            current_date_ = get_date_string(now);
            current_filename_ = calc_filename_(current_date_, 0);
        } else {
            // For non-daily rotation, use base filename directly
            current_filename_ = base_filename_ + ".txt";
            // Set rotation_tp_ to far future to effectively disable daily rotation
            rotation_tp_ = now + std::chrono::hours(24 * 365 * 100); // 100 years in future
        }
        
        file_helper_.open(current_filename_, false);
        current_size_ = file_helper_.size();
    }

protected:
    void sink_it_(const details::log_msg& msg) override {
        auto time = msg.time;
        
        // Check if we need to rotate based on date
        if (time >= rotation_tp_) {
            rotate_by_date_(time);
        }
        
        // Format the message
        memory_buf_t formatted;
        base_sink<Mutex>::formatter_->format(msg, formatted);
        
        // Check if we need to rotate based on file size
        if (current_size_ + formatted.size() > max_size_) {
            rotate_by_size_();
        }
        
        // Write to file
        file_helper_.write(formatted);
        current_size_ += formatted.size();
    }

    void flush_() override {
        file_helper_.flush();
    }

private:
    void rotate_by_date_(const log_clock::time_point& time) {
        file_helper_.close();
        
        // Update to new date
        current_date_ = get_date_string(time);
        current_filename_ = calc_filename_(current_date_, 0);
        
        // Calculate next rotation time
        rotation_tp_ = next_rotation_tp_(time);
        
        // Open new file
        file_helper_.open(current_filename_, false);
        current_size_ = file_helper_.size();
    }
    
    void rotate_by_size_() {
        file_helper_.close();
        
        // Find the next available number (highest existing + 1)
        int next_number = 1;
        while (next_number <= max_files_) {
            filename_t test_name;
            if (enable_daily_rotation_) {
                test_name = calc_filename_(current_date_, next_number);
            } else {
                test_name = calc_filename_("", next_number); // date parameter not used for non-daily
            }
            if (!details::os::path_exists(test_name)) {
                break;
            }
            next_number++;
        }
        
        // If we've reached max files, remove the oldest (.1) and shift all files down
        if (next_number > max_files_) {
            filename_t oldest;
            if (enable_daily_rotation_) {
                oldest = calc_filename_(current_date_, 1);
            } else {
                oldest = calc_filename_("", 1);
            }
            details::os::remove(oldest);
            
            // Shift all files: .2 -> .1, .3 -> .2, etc.
            for (int i = 2; i <= max_files_; ++i) {
                filename_t src, target;
                if (enable_daily_rotation_) {
                    src = calc_filename_(current_date_, i);
                    target = calc_filename_(current_date_, i - 1);
                } else {
                    src = calc_filename_("", i);
                    target = calc_filename_("", i - 1);
                }
                if (details::os::path_exists(src)) {
                    details::os::rename(src, target);
                }
            }
            next_number = max_files_;
        }
        
        // Move current file to the next available number (newest archived)
        filename_t archive_name;
        if (enable_daily_rotation_) {
            archive_name = calc_filename_(current_date_, next_number);
        } else {
            archive_name = calc_filename_("", next_number);
        }
        if (details::os::path_exists(current_filename_)) {
            details::os::rename(current_filename_, archive_name);
        }
        
        // Create new current file
        file_helper_.open(current_filename_, false);
        current_size_ = 0;
    }
    
    filename_t calc_filename_(const std::string& date, std::size_t index) {
        if (!enable_daily_rotation_) {
            // For non-daily rotation (like latest logs)
            if (index == 0) {
                return base_filename_ + ".txt";
            } else {
                return fmt::format("{}.{}.txt", base_filename_, index);
            }
        } else {
            // For daily rotation
            if (index == 0) {
                return fmt::format("{}_{}.txt", base_filename_, date);
            } else {
                return fmt::format("{}_{}.{}.txt", base_filename_, date, index);
            }
        }
    }
    
    std::string get_date_string(const log_clock::time_point& time) {
        auto time_t = log_clock::to_time_t(time);
        auto tm = *std::localtime(&time_t);
        char date_str[32];
        std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", &tm);
        return std::string(date_str);
    }
    
    log_clock::time_point next_rotation_tp_(const log_clock::time_point& time) {
        auto time_t = log_clock::to_time_t(time);
        auto tm = *std::localtime(&time_t);
        
        tm.tm_hour = rotation_hour_;
        tm.tm_min = rotation_minute_;
        tm.tm_sec = 0;
        
        auto rotation_time = log_clock::from_time_t(std::mktime(&tm));
        
        if (rotation_time > time) {
            return rotation_time;
        }
        
        // Add one day
        tm.tm_mday += 1;
        return log_clock::from_time_t(std::mktime(&tm));
    }

    filename_t base_filename_;
    std::size_t max_size_;
    int rotation_hour_;
    int rotation_minute_;
    bool enable_daily_rotation_;
    std::size_t current_size_;
    std::string current_date_;
    filename_t current_filename_;
    log_clock::time_point rotation_tp_;
    details::file_helper file_helper_;
    
    static const int max_files_ = 1000; // Maximum number of rotating files per day
};

using daily_rotating_file_sink_mt = daily_rotating_file_sink<std::mutex>;
using daily_rotating_file_sink_st = daily_rotating_file_sink<details::null_mutex>;

} // namespace sinks
} // namespace spdlog
