#pragma once

#include "db_knowledge.hpp"
#include <optional>
#include <utility>

namespace vec_db {
    /**
     * @brief Segment Chinese text using LAC (Lexical Analysis of Chinese)
     * 
     * @param query The text to segment
     * @return std::pair<std::string, std::string> A pair of strings: 
     *         first = full segmented text, second = keyword text (nouns and verbs only)
     */
    std::pair<std::string, std::string> segment_query(const std::string_view query);

    /**
     * @brief Query knowledge from vector database using hybrid search
     * 
     * @param query The query text
     * @param certainty_threshold Minimum certainty threshold for results
     * @param top_k Optional limit on number of results
     * @return std::vector<DBKnowledge> List of matching knowledge items
     */
    std::vector<DBKnowledge> query_knowledge_from_vec_db(
        const std::string_view query, 
        float certainty_threshold = 0.7f, 
        std::optional<size_t> top_k = std::nullopt);

    /**
     * @brief Create a GraphQL query for hybrid search
     * 
     * @param schema The collection name
     * @param emb Vector embedding for similarity search
     * @param text_query Text for BM25 keyword search
     * @param certainty_threshold Minimum certainty threshold
     * @param top_k Optional limit on number of results
     * @return std::string The formatted GraphQL query
     */
    std::string graphql_query(
        const std::string_view schema, 
        const neural_network::emb_vec_t &emb,
        const std::string &text_query,
        float certainty_threshold, 
        std::optional<size_t> top_k = std::nullopt);

    /**
     * @brief Create a GraphQL query for vector search (backwards compatibility)
     */
    std::string graphql_query(
        const std::string_view schema, 
        const neural_network::emb_vec_t &emb,
        float certainty_threshold, 
        std::optional<size_t> top_k = std::nullopt);
}