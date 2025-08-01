#pragma once

#include "db_knowledge.hpp"
#include <optional>
namespace vec_db {
    std::vector<DBKnowledge> query_knowledge_from_vec_db(const std::string_view query, float certainty_threshold = 0.7f, std::optional<size_t> top_k = std::nullopt);


    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
        float certainty_threshold, std::optional<size_t> top_k = std::nullopt);
}