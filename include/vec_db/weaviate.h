#pragma once

#include "db_knowledge.hpp"
namespace vec_db {
    std::vector<DBKnowledge> query_knowledge(const std::string_view query, float certainty_threshold = 0.7f, size_t top_k = 5);


    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
        float certainty_threshold, size_t top_k);
}