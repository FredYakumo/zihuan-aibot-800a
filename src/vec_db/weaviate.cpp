#include "vec_db/weaviate.h"
#include "config.h"
#include "get_optional.hpp"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <chrono>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <general-wheel-cpp/string_utils.hpp>
#include <iterator>
#include <string>

namespace vec_db {
    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
                              float certainty_threshold, std::optional<size_t> top_k) {
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
                        _additional {
                            certainty
                        }
                    }
            }
        })";

        return query_base;
    }

    std::vector<DBKnowledge> query_knowledge_from_vec_db(const std::string_view query, float certainty_threshold,
                                                         std::optional<size_t> top_k) {
        std::vector<DBKnowledge> results;
        if (query.empty()) {
            return results; // Return empty if query is empty
        }
        auto emb = neural_network::get_model_set().text_embedding_model->embed(std::string(query));
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        auto &config = Config::instance();
        cpr::Response response = cpr::Post(
            cpr::Url{fmt::format("http://{}:{}/v1/graphql", config.vec_db_url, config.vec_db_port)},
            cpr::Body{
                nlohmann::json{{"query", graphql_query("AIBot_knowledge", emb, certainty_threshold, top_k)}}.dump(),
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

            if (auto data = get_optional(json_response, "data"); data.has_value()) {
                if (auto get = get_optional(data.value(), "Get"); get.has_value()) {
                    if (auto knowledge = get_optional(get.value(), "AIBot_knowledge"); knowledge.has_value()) {
                        spdlog::info("Query knowledge 完成, 返回 {} 条结果, 耗时: {} ms", knowledge.value().size(),
                                     duration.count());
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
                                db_knowledge.certainty = get_optional(additional.value(), "certainty").value_or(0.0f);
                            } else {
                                db_knowledge.certainty = 0.0f;
                            }
                            spdlog::info(
                                "Found knowledge: knowledge_class_filter={}, content={}, keywords={}, creator_name={}, create_time={}, certainty={}, ",
                                db_knowledge.knowledge_class_filter,
                                db_knowledge.content,
                                wheel::join_str(std::cbegin(db_knowledge.keyword), std::cend(db_knowledge.keyword), ","),
                                db_knowledge.creator_name,
                                db_knowledge.create_time,
                                db_knowledge.certainty);
                            results.push_back(db_knowledge);
                        }
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