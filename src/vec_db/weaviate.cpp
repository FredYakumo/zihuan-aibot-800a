#include "vec_db/weaviate.h"
#include "config.h"
#include "get_optional.hpp"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <chrono>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <general-wheel-cpp/linalg_boost/linalg_boost.hpp>
#include <general-wheel-cpp/string_utils.hpp>
#include <iterator>
#include <string>

namespace vec_db {
    /**
     * Helper function to segment a query using LAC
     * Returns a pair containing:
     * 1. The full segmented query with all words
     * 2. A keyword-only query with just nouns and verbs for BM25 search
     */
    std::pair<std::string, std::string> segment_query(const std::string_view query) {
        std::string processed_query = std::string(query);
        std::string keyword_query = processed_query; // For BM25 search

#ifdef __USE_PADDLE_INFERENCE__
        try {
            auto &lac_model = neural_network::get_model_set().lac_model;
            if (lac_model) {
                auto lac_result = lac_model->run(processed_query);

                // Reconstruct query with segmented words
                processed_query.clear();
                keyword_query.clear();

                std::vector<std::string> keywords;

                for (const auto &item : lac_result) {
                    processed_query += item.word + " ";

                    // 包含所有名词类型和命名实体
                    // 名词类别:
                    // n: 普通名词
                    // nr: 人名
                    // ns: 地名
                    // nt: 机构名
                    // nw: 作品名
                    // nz: 其他专有名词
                    // 实体类型:
                    // PER: 人物
                    // LOC: 地点
                    // ORG: 组织机构
                    // TIME: 时间
                    // 其他重要词性:
                    // v: 动词
                    // vd, vn: 动词衍生词
                    // a: 形容词
                    if (item.tag.starts_with("n") || item.tag.starts_with("v") || item.tag.starts_with("a") ||
                        item.tag.starts_with("PER") || item.tag.starts_with("LOC") || item.tag.starts_with("ORG") ||
                        item.tag.starts_with("TIME")) {
                        keywords.push_back(item.word);
                    }
                }

                // Join keywords with proper spacing for BM25 search
                keyword_query = wheel::join_str(keywords.begin(), keywords.end(), " ");

                spdlog::info("LAC segmentation result: {}", processed_query);
                spdlog::info("Keyword query for BM25: {}", keyword_query);
            }
        } catch (const std::exception &e) {
            spdlog::error("LAC segmentation failed: {}", e.what());
            // Fallback to original query if LAC fails
            processed_query = std::string(query);
            keyword_query = processed_query;
        }
#endif

        return {processed_query, keyword_query};
    }
    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
                              const std::string &text_query, float certainty_threshold, std::optional<size_t> top_k) {
        // If text query is empty, use vector search only
        if (text_query.empty()) {
            std::string query_base = fmt::format(
                R"({{
                        Get{{
                            {}(
                        nearVector: {{
                            vector: [{}],
                            certainty: {}
                        }})",
                "AIBot_knowledge",
                wheel::join_str(std::cbegin(emb), std::cend(emb), ",", [](auto v) { return std::to_string(v); }),
                certainty_threshold);

            if (top_k.has_value()) {
                query_base += fmt::format(
                    R"(
                        limit: {})",
                    top_k.value());
            }

            query_base += R"(
                        ) {
                            keyword
                            content
                            creator_name
                            create_time
                            knowledge_class_filter
                            _additional {
                                certainty
                            }
                        }
                }
            })";

            return query_base;
        }

        // Use hybrid search with both vector and BM25
        std::string query_base = fmt::format(
            R"({{
                    Get{{
                        {}(
                    hybrid: {{
                        vector: [{}],
                        query: "{}",
                        alpha: 0.6,
                        properties: ["keyword"]
                    }})",
            "AIBot_knowledge",
            wheel::join_str(std::cbegin(emb), std::cend(emb), ",", [](auto v) { return std::to_string(v); }),
            text_query);

        if (top_k.has_value()) {
            query_base += fmt::format(
                R"(
                    limit: {})",
                top_k.value());
        }

        query_base += R"(
                    ) {
                        keyword
                        content
                        creator_name
                        create_time
                        knowledge_class_filter
                        _additional {
                            certainty
                            score
                            id
                        }
                    }
            }
        })";

        return query_base;
    }

    // Keep backward compatibility with the old function
    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
                              float certainty_threshold, std::optional<size_t> top_k) {
        return graphql_query(schema, emb, "", certainty_threshold, top_k);
    }

    std::vector<DBKnowledge> query_knowledge_from_vec_db(const std::string_view query, float certainty_threshold,
                                                         std::optional<size_t> top_k) {
        std::vector<DBKnowledge> results;
        if (query.empty()) {
            return results; // Return empty if query is empty
        }

        // Process query with LAC for Chinese word segmentation to get:
        // 1. processed_query: Full segmented text with all words (for vector embedding)
        // 2. keyword_query: Only key terms (nouns, verbs, adjectives) for BM25 search
        auto [processed_query, keyword_query] = segment_query(query);

        // Generate vector embeddings using the full processed query
        auto emb = neural_network::get_model_set().text_embedding_model->embed(processed_query);

        auto emb_pooled = wheel::linalg_boost::mean_pooling(emb);

        // Execute hybrid search combining:
        // - Vector similarity for semantic matching (applied to all fields)
        // - BM25 keyword search only on the keyword field
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        auto &config = Config::instance();
        cpr::Response response =
            cpr::Post(cpr::Url{fmt::format("http://{}:{}/v1/graphql", config.vec_db_url, config.vec_db_port)},
                      cpr::Body{
                          nlohmann::json{{"query", graphql_query("AIBot_knowledge", emb_pooled, keyword_query,
                                                                 certainty_threshold, top_k)}}
                              .dump(),
                      },
                      cpr::Header{{"Content-Type", "application/json"}});
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (response.status_code != 200) {
            spdlog::error("Weaviate query failed: {} - {}", response.status_code, response.text);
            return results; // Return empty if query fails
        }

        /* Weaviate's Example response:
        {
            "data": {
                "Get": {
                "Publication": [
                    {
                    "_additional": {
                        "distance": -1.1920929e-07
                    },
                    "name": "The New York Times Company"
                    },
                    {
                    "_additional": {
                        "distance": 0.059879005
                    },
                    "name": "New York Times"
                    },
                    {
                    "_additional": {
                        "distance": 0.09176409
                    },
                    "name": "International New York Times"
                    },
                    {
                    "_additional": {
                        "distance": 0.13954824
                    },
                    "name": "New Yorker"
                    },
                    ...
                ]
                }
            }
        }
        */

        try {
            auto json_response = nlohmann::json::parse(response.text);

            // Log the full response for debugging
            spdlog::debug("Weaviate raw response: {}", response.text);

            if (auto data = get_optional(json_response, "data"); data.has_value()) {
                if (auto get = get_optional(data.value(), "Get"); get.has_value()) {
                    if (auto knowledge = get_optional(get.value(), "AIBot_knowledge"); knowledge.has_value()) {
                        size_t total_found = knowledge.value().size();

                        // 处理所有找到的知识
                        for (const auto &item : knowledge.value()) {
                            DBKnowledge db_knowledge;
                            if (auto keyword_opt = get_optional(item, "keyword");
                                keyword_opt.has_value() && keyword_opt->is_array()) {
                                db_knowledge.keyword.clear();
                                for (const auto &kw : *keyword_opt) {
                                    if (kw.is_string()) {
                                        db_knowledge.keyword.push_back(kw.get<std::string>());
                                    }
                                }
                            } else {
                                db_knowledge.keyword.clear();
                            }
                            db_knowledge.content = get_optional(item, "content").value_or("");
                            db_knowledge.creator_name = get_optional(item, "creator_name").value_or("");
                            db_knowledge.create_time = get_optional(item, "create_time").value_or("");
                            db_knowledge.knowledge_class_filter =
                                get_optional(item, "knowledge_class_filter").value_or("");

                            if (auto additional = get_optional(item, "_additional"); additional.has_value()) {
                                // Handle potential type mismatches for certainty and score
                                float certainty = 0.0f;
                                float hybrid_score = 0.0f;

                                // Try to extract certainty as float
                                try {
                                    auto certainty_opt = get_optional(additional.value(), "certainty");
                                    if (certainty_opt.has_value() && !certainty_opt.value().is_null()) {
                                        if (certainty_opt.value().is_number()) {
                                            certainty = certainty_opt.value().get<float>();
                                        } else if (certainty_opt.value().is_string()) {
                                            // If it's a string, try to convert to float
                                            certainty = std::stof(certainty_opt.value().get<std::string>());
                                        }
                                    }
                                } catch (const std::exception &e) {
                                    spdlog::warn("Failed to parse certainty: {}", e.what());
                                    certainty = 0.0f;
                                }

                                // Try to extract score as float
                                try {
                                    auto score_opt = get_optional(additional.value(), "score");
                                    if (score_opt.has_value() && !score_opt.value().is_null()) {
                                        if (score_opt.value().is_number()) {
                                            hybrid_score = score_opt.value().get<float>();
                                        } else if (score_opt.value().is_string()) {
                                            // If it's a string, try to convert to float
                                            hybrid_score = std::stof(score_opt.value().get<std::string>());
                                        }
                                    }
                                } catch (const std::exception &e) {
                                    spdlog::warn("Failed to parse score: {}", e.what());
                                    hybrid_score = 0.0f;
                                }

                                db_knowledge.certainty = certainty;
                                
                                // 如果certainty为0但有hybrid_score，则使用hybrid_score作为certainty的替代
                                if (certainty <= 0.0f && hybrid_score > 0.0f) {
                                    db_knowledge.certainty = hybrid_score;
                                    // spdlog::info("Using hybrid_score {} as certainty because certainty is {}", hybrid_score, certainty);
                                }

                                // Only add results that meet the score threshold
                                if (hybrid_score >= certainty_threshold) {
                                    // Log only results that pass the threshold
                                    // 只有当certainty不为0时才在日志中显示
                                    if (db_knowledge.certainty > 0.0f) {
                                        spdlog::info("Found knowledge (filtered): filter={}, content={}, keywords={}, "
                                                     "creator={}, time={}, certainty={}, hybrid_score={}",
                                                     db_knowledge.knowledge_class_filter,
                                                     db_knowledge.content.substr(0, 50) +
                                                         (db_knowledge.content.length() > 50 ? "..." : ""),
                                                     wheel::join_str(std::cbegin(db_knowledge.keyword),
                                                                     std::cend(db_knowledge.keyword), ","),
                                                     db_knowledge.creator_name, db_knowledge.create_time,
                                                     db_knowledge.certainty, hybrid_score);
                                    } else {
                                        spdlog::info("Found knowledge (filtered): filter={}, content={}, keywords={}, "
                                                     "creator={}, time={}, hybrid_score={}",
                                                     db_knowledge.knowledge_class_filter,
                                                     db_knowledge.content.substr(0, 50) +
                                                         (db_knowledge.content.length() > 50 ? "..." : ""),
                                                     wheel::join_str(std::cbegin(db_knowledge.keyword),
                                                                     std::cend(db_knowledge.keyword), ","),
                                                     db_knowledge.creator_name, db_knowledge.create_time, hybrid_score);
                                    }

                                    results.push_back(db_knowledge);
                                } else {
                                    // Log skipped results at debug level instead of info
                                    spdlog::debug("Skipped low-score result: {}, score={}, threshold={}",
                                                  db_knowledge.content.substr(0, 30) +
                                                      (db_knowledge.content.length() > 30 ? "..." : ""),
                                                  hybrid_score, certainty_threshold);
                                }
                            } else {
                                db_knowledge.certainty = 0.0f;

                                // Without score information, assume it passes filter and log it
                                // 只有当certainty不为0时才在日志中显示
                                if (db_knowledge.certainty > 0.0f) {
                                    spdlog::info("Found knowledge (unscored): filter={}, content={}, keywords={}, "
                                                 "creator={}, time={}, certainty={}",
                                                 db_knowledge.knowledge_class_filter,
                                                 db_knowledge.content.substr(0, 50) +
                                                     (db_knowledge.content.length() > 50 ? "..." : ""),
                                                 wheel::join_str(std::cbegin(db_knowledge.keyword),
                                                                 std::cend(db_knowledge.keyword), ","),
                                                 db_knowledge.creator_name, db_knowledge.create_time,
                                                 db_knowledge.certainty);
                                } else {
                                    spdlog::info("Found knowledge (unscored): filter={}, content={}, keywords={}, "
                                                 "creator={}, time={}",
                                                 db_knowledge.knowledge_class_filter,
                                                 db_knowledge.content.substr(0, 50) +
                                                     (db_knowledge.content.length() > 50 ? "..." : ""),
                                                 wheel::join_str(std::cbegin(db_knowledge.keyword),
                                                                 std::cend(db_knowledge.keyword), ","),
                                                 db_knowledge.creator_name, db_knowledge.create_time);
                                }

                                // Without score information, we can't filter, so include it anyway
                                results.push_back(db_knowledge);
                            }
                        }

                        // 查询完成后显示统计信息
                        spdlog::info("查询知识完成: 原始结果 {} 条, 过滤后 {} 条, 耗时: {} ms", total_found,
                                     results.size(), duration.count());
                    }
                }
            } else {
                spdlog::info("Query knowledge 失败, 耗时: {} ms, 返回 {}, ", duration.count(), response.text);
            }
        } catch (const std::exception &e) {
            spdlog::error("Error parsing Weaviate response: {}", e.what());
        }
        return results;
    }
} // namespace vec_db