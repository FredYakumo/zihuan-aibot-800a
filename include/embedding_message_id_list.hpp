#pragma once

#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <cstdint>
#include <linalg_boost/linalg_boost.hpp>
#include <shared_mutex>
#include <algorithm>
#include <spdlog/spdlog.h>

class embedding_message_id_list {
public:
    std::vector<uint64_t> find_similary_message(const std::string &query, float threshold = 0.5f, size_t top_k = 5) const {
        std::vector<uint64_t> ret;
        if (embedding_mat.empty()) {
            return ret;
        }
        auto target_emb = neural_network::get_model_set().text_embedding_model->embed(query);
        auto mean_pooled = wheel::linalg_boost::mean_pooling(target_emb);
        // std::shared_lock lock(mutex);
        auto sim = wheel::linalg_boost::batch_cosine_similarity(embedding_mat, mean_pooled);
        std::vector<std::pair<float, uint64_t>> sim_pairs;
        for (int i = 0; i < sim.size(); ++i) {
            if (sim[i] > threshold) {
                sim_pairs.emplace_back(sim[i], message_id_list[i]);
            }
        }

        // Sort by similarity descending
        std::sort(sim_pairs.begin(), sim_pairs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Take top-k and log
        size_t count = std::min(sim_pairs.size(), top_k);
        for (size_t i = 0; i < count; ++i) {
            ret.push_back(sim_pairs[i].second);
            spdlog::info("Similar message ID: {}, Similarity: {:.4f}", sim_pairs[i].second, sim_pairs[i].first);
        }

        return ret;
    }

    void add_message(uint64_t message_id, const std::string &content) {
        if (content.empty()) {
            return;
        }
        auto emb = neural_network::get_model_set().text_embedding_model->embed(content);
        auto pooled = wheel::linalg_boost::mean_pooling(emb);
        // std::unique_lock lock(mutex);
        embedding_mat.push_back(pooled);
        message_id_list.push_back(message_id);
    }
private:
    neural_network::emb_mat_t embedding_mat;
    std::vector<uint64_t> message_id_list;
    std::shared_mutex mutex;
};