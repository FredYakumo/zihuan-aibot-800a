#include "vec_db/weaviate.h"
#include "config.h"
#include "get_optional.hpp"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"
#include <chrono>
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <iterator>
#include <string>
#include <string_utils.hpp>

auto config = Config::instance();

namespace vec_db {
    std::string graphql_query(const std::string_view schema, const neural_network::emb_vec_t &emb,
                              float certainty_threshold, size_t top_k) {
        return fmt::format(
            R"({{
                    Get{{
                        {}(
                    nearVector: {{
                        vector: [{}],
                        certainty: {}
                    }}
                    limit: {}
                    ) {{
                        key
                        value
                        creator_name
                        create_dt
                        _additional {{
                            certainty
                        }}
                    }}
            }})",
            "AIBot_knowledge",
            wheel::join_str(std::cbegin(emb), std::cend(emb), ",", [](auto v) { return std::to_string(v); }), schema,
            certainty_threshold, top_k);
    }

    std::vector<DBKnowledge> query_knowledge(const std::string_view query, float certainty_threshold, size_t top_k) {
        std::vector<DBKnowledge> results;
        if (query.empty()) {
            return results; // Return empty if query is empty
        }
        auto emb = neural_network::get_model_set().text_embedding_model.embed(std::string(query));
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        cpr::Response response =
            cpr::Get(cpr::Url{fmt::format("http://{}:{}/v1/graphql", config.vec_db_url, config.vec_db_port)},
                     cpr::Body{
                         nlohmann::json{{"query", graphql_query("AIBot_knowledge", emb, certainty_threshold, top_k)}},
                     });
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        if (response.status_code != 200) {
            spdlog::error("Weaviate query failed: {} - {}", response.status_code, response.error.message);
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
                                     (end_time - start_time).count());
                        for (const auto &item : knowledge.value()) {
                            DBKnowledge db_knowledge;
                            db_knowledge.key = get_optional(item, "key").value_or("");
                            db_knowledge.value = get_optional(item, "value").value_or("");
                            db_knowledge.creator_name = get_optional(item, "creator_name").value_or("");
                            db_knowledge.create_dt = get_optional(item, "create_dt").value_or("");

                            if (auto additional = get_optional(item, "_additional"); additional.has_value()) {
                                db_knowledge.certainty = get_optional(additional.value(), "certainty").value_or(0.0f);
                            } else {
                                db_knowledge.certainty = 0.0f;
                            }
                            spdlog::info(
                                "Found knowledge: key={}, value={}, creator_name={}, create_dt={}, certainty={}",
                                db_knowledge.key, db_knowledge.value, db_knowledge.creator_name, db_knowledge.create_dt,
                                db_knowledge.certainty);
                            results.push_back(db_knowledge);
                        }
                    }
                }
            }
        } catch (const std::exception &e) {
            spdlog::error("Error parsing Weaviate response: {}", e.what());
        }
        return results;
    }
} // namespace vec_db