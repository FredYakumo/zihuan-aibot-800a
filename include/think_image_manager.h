#pragma once

#include "config.h"
#include <filesystem>
#include <random>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace bot_adapter {

    class ThinkImageManager {
      public:
        static ThinkImageManager &instance() {
            static ThinkImageManager manager;
            return manager;
        }

        // Initialize by loading all images from the configured directory
        void initialize() {
            m_images.clear();
            m_usage_counts.clear();

            const auto &dir_path = Config::instance().think_pictures_dir;
            if (dir_path.empty()) {
                spdlog::warn("Think pictures directory not configured");
                return;
            }

            try {
                for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
                    if (entry.is_regular_file()) {
                        const auto &path = entry.path();
                        // Only consider common image file extensions
                        const auto &ext = path.extension().string();
                        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".gif" || ext == ".JPG" ||
                            ext == ".JPEG" || ext == ".PNG" || ext == ".GIF") {
                            m_images.push_back(path.string());
                            m_usage_counts[path.string()] = 0;
                        }
                    }
                }
                spdlog::info("Loaded {} thinking images from {}", m_images.size(), dir_path);
            } catch (const std::exception &e) {
                spdlog::error("Failed to load thinking images: {}", e.what());
            }
        }

        // Get a random image path with penalty for frequently used images
        // Returns an empty string if no images are available and no fallback URL is configured
        std::string get_random_image_path() {
            if (m_images.empty()) {
                // Fallback to the configured URL if it's available
                const auto &url = Config::instance().think_image_url;
                if (url.empty()) {
                    return "";
                }
                return url;
            }

            // If we only have one image, just return it
            if (m_images.size() == 1) {
                m_usage_counts[m_images[0]]++;
                return m_images[0];
            }

            // Calculate weights based on usage counts (less used = higher weight)
            std::vector<double> weights;
            for (const auto &img : m_images) {
                // Add 1 to avoid division by zero and ensure newer images get fair chance
                double weight = 1.0 / (m_usage_counts[img] + 1.0);
                weights.push_back(weight);
            }

            // Create a distribution based on the weights
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(weights.begin(), weights.end());

            // Get a random index based on the weights
            int selected_idx = dist(gen);

            // Increment the usage count for the selected image
            m_usage_counts[m_images[selected_idx]]++;

            return m_images[selected_idx];
        }

      private:
        ThinkImageManager() = default;

        std::vector<std::string> m_images;
        std::unordered_map<std::string, int> m_usage_counts;
    };

} // namespace bot_adapter