#pragma once

#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <spdlog/details/file_helper.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace spdlog {
    namespace sinks {

        /**
         * @brief A sink that rotates logs based on both date and file size
         *
         * This class implements a custom sink for spdlog that performs log rotation
         * based on two criteria:
         * 1. Daily rotation at a configurable time
         * 2. Size-based rotation when file exceeds maximum size
         *
         * File rotation behavior:
         * - Current logs: aibot_800a_2025-07-20.txt (newest)
         * - When size limit reached: current -> highest available number
         * - .1 = oldest archived, .2, .3... = progressively newer, highest number = newest archived
         */
        template <typename Mutex> class daily_rotating_file_sink : public base_sink<Mutex> {
          public:
            /**
             * @brief Constructs a daily rotating file sink
             *
             * @param base_filename Base name of the log file (without date part or extension)
             * @param max_size Maximum size of a single log file in bytes
             * @param rotation_hour Hour of the day to perform daily rotation (24h format)
             * @param rotation_minute Minute of the hour to perform daily rotation
             * @param enable_daily_rotation Whether to enable rotation based on date
             */
            daily_rotating_file_sink(filename_t base_filename, std::size_t max_size, int rotation_hour = 0,
                                     int rotation_minute = 0, bool enable_daily_rotation = true)
                : base_filename(std::move(base_filename)), max_size(max_size), rotation_hour(rotation_hour),
                  rotation_minute(rotation_minute), enable_daily_rotation(enable_daily_rotation), current_size(0) {

                auto now = std::chrono::system_clock::now();
                if (enable_daily_rotation) {
                    rotation_tp = next_rotation_tp(now);
                    current_date = get_date_string(now);
                    current_filename = calc_filename(current_date, 0);
                    spdlog::debug("Daily rotating file sink initialized with date-based rotation at {}:{}",
                                  rotation_hour, rotation_minute);
                } else {
                    // For non-daily rotation, use base filename directly
                    current_filename = base_filename + ".txt";
                    // Set rotation_tp to far future to effectively disable daily rotation
                    rotation_tp = now + std::chrono::hours(24 * 365 * 100); // 100 years in future
                    spdlog::debug("Daily rotating file sink initialized with size-based rotation only");
                }

                file_helper.open(current_filename, false);
                current_size = file_helper.size();
                spdlog::debug("Log file opened: {}, initial size: {} bytes", current_filename, current_size);
            }

          protected:
            /**
             * @brief Processes a log message, performs rotation if needed
             *
             * @param msg The log message to process
             */
            void sink_it(const details::log_msg &msg) override {
                auto time = msg.time;

                // Check if we need to rotate based on date
                if (time >= rotation_tp) {
                    rotate_by_date(time);
                }

                // Format the message
                memory_buf_t formatted;
                base_sink<Mutex>::formatter_->format(msg, formatted);

                // Check if we need to rotate based on file size
                if (current_size + formatted.size() > max_size) {
                    rotate_by_size();
                }

                // Write to file
                file_helper.write(formatted);
                current_size += formatted.size();
            }

            /**
             * @brief Flushes the log file
             */
            void flush() override { file_helper.flush(); }

          private:
            /**
             * @brief Performs rotation based on date change
             *
             * @param time Current time point
             */
            void rotate_by_date(const log_clock::time_point &time) {
                file_helper.close();
                spdlog::debug("Rotating log file by date");

                // Update to new date
                current_date = get_date_string(time);
                current_filename = calc_filename(current_date, 0);

                // Calculate next rotation time
                rotation_tp = next_rotation_tp(time);

                // Open new file
                file_helper.open(current_filename, false);
                current_size = file_helper.size();
                spdlog::debug("New log file created: {}", current_filename);
            }

            /**
             * @brief Performs rotation based on file size
             */
            void rotate_by_size() {
                file_helper.close();
                spdlog::debug("Rotating log file by size");

                // Find the next available number (highest existing + 1)
                int next_number = 1;
                while (next_number <= max_files) {
                    filename_t test_name;
                    if (enable_daily_rotation) {
                        test_name = calc_filename(current_date, next_number);
                    } else {
                        test_name = calc_filename("", next_number); // date parameter not used for non-daily
                    }
                    if (!details::os::path_exists(test_name)) {
                        break;
                    }
                    next_number++;
                }

                // If we've reached max files, remove the oldest (.1) and shift all files down
                if (next_number > max_files) {
                    filename_t oldest;
                    if (enable_daily_rotation) {
                        oldest = calc_filename(current_date, 1);
                    } else {
                        oldest = calc_filename("", 1);
                    }
                    details::os::remove(oldest);

                    // Shift all files: .2 -> .1, .3 -> .2, etc.
                    for (int i = 2; i <= max_files; ++i) {
                        filename_t src, target;
                        if (enable_daily_rotation) {
                            src = calc_filename(current_date, i);
                            target = calc_filename(current_date, i - 1);
                        } else {
                            src = calc_filename("", i);
                            target = calc_filename("", i - 1);
                        }
                        if (details::os::path_exists(src)) {
                            details::os::rename(src, target);
                        }
                    }
                    next_number = max_files;
                }

                // Move current file to the next available number (newest archived)
                filename_t archive_name;
                if (enable_daily_rotation) {
                    archive_name = calc_filename(current_date, next_number);
                } else {
                    archive_name = calc_filename("", next_number);
                }
                if (details::os::path_exists(current_filename)) {
                    details::os::rename(current_filename, archive_name);
                }

                // Create new current file
                file_helper.open(current_filename, false);
                current_size = 0;
                spdlog::debug("New size-rotated log file created: {}", current_filename);
            }

            /**
             * @brief Calculates the filename based on date and index
             *
             * @param date Date string for daily rotation (empty for non-daily)
             * @param index Rotation index (0 for current, >0 for archived)
             * @return filename_t The calculated filename
             */
            filename_t calc_filename(const std::string &date, std::size_t index) {
                if (!enable_daily_rotation) {
                    // For non-daily rotation (like latest logs)
                    if (index == 0) {
                        return base_filename + ".txt";
                    } else {
                        return fmt::format("{}.{}.txt", base_filename, index);
                    }
                } else {
                    // For daily rotation
                    if (index == 0) {
                        return fmt::format("{}_{}.txt", base_filename, date);
                    } else {
                        return fmt::format("{}_{}.{}.txt", base_filename, date, index);
                    }
                }
            }

            /**
             * @brief Gets a formatted date string for the log filename
             *
             * @param time The time point to format
             * @return std::string Formatted date string (YYYY-MM-DD)
             */
            std::string get_date_string(const log_clock::time_point &time) {
                auto time_t = log_clock::to_time_t(time);
                auto tm = *std::localtime(&time_t);
                char date_str[32];
                std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", &tm);
                return std::string(date_str);
            }

            /**
             * @brief Calculates the next time point for daily rotation
             *
             * @param time Current time point
             * @return log_clock::time_point Next rotation time point
             */
            log_clock::time_point next_rotation_tp(const log_clock::time_point &time) {
                auto time_t = log_clock::to_time_t(time);
                auto tm = *std::localtime(&time_t);

                tm.tm_hour = rotation_hour;
                tm.tm_min = rotation_minute;
                tm.tm_sec = 0;

                auto rotation_time = log_clock::from_time_t(std::mktime(&tm));

                if (rotation_time > time) {
                    return rotation_time;
                }

                // Add one day
                tm.tm_mday += 1;
                return log_clock::from_time_t(std::mktime(&tm));
            }

            // Member variables
            filename_t base_filename;          ///< Base filename without date or extension
            std::size_t max_size;              ///< Maximum size of a single log file
            int rotation_hour;                 ///< Hour of the day to rotate log (24h format)
            int rotation_minute;               ///< Minute of the hour to rotate log
            bool enable_daily_rotation;        ///< Whether daily rotation is enabled
            std::size_t current_size;          ///< Current size of the active log file
            std::string current_date;          ///< Current date string (YYYY-MM-DD)
            filename_t current_filename;       ///< Current active log filename
            log_clock::time_point rotation_tp; ///< Next scheduled rotation time point
            details::file_helper file_helper;  ///< File helper for I/O operations

            /// Maximum number of rotating files per day
            static const int max_files = 1000;
        };

        using daily_rotating_file_sink_mt = daily_rotating_file_sink<std::mutex>;
        using daily_rotating_file_sink_st = daily_rotating_file_sink<details::null_mutex>;

    } // namespace sinks
} // namespace spdlog
