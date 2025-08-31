#include "agent/action_agent.h"
#include "config.h"
#include "get_optional.hpp"
#include "neural_network/model_set.h"
#include "neural_network/text_model/lac/tag_constants.hpp"
#include <filesystem>
#include <fstream>
#include <general-wheel-cpp/linalg_boost/linalg_boost.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using namespace neural_network::lac::tags;

namespace agent {

    /**
     * Structure to hold different types of reference terms
     * Each set contains specific reference terms that identify a particular target category
     */
    struct ReferenceTerms {
        std::set<std::string> self_references;         // References to the bot itself
        std::set<std::string> general_llm_references;  // References to general language models
        std::set<std::string> specific_llm_references; // References to specific named models
        std::set<std::string> system_prompt_terms;     // Terms related to system prompts
        std::set<std::string> request_indicators;      // Terms indicating a request/question
    };

    /**
     * Resolves the path to the agent dictionary file
     *
     * @return The resolved path to the agent dictionary file from configuration
     * @throws std::runtime_error if the agent dictionary file cannot be found in configuration
     */
    std::string resolve_agent_dict_path() {
        // Get paths directly from config
        const std::vector<std::string>& config_paths = Config::instance().agent_dict_alt_paths;
        
        if (config_paths.empty()) {
            throw std::runtime_error("No agent_dict.json paths configured in config file");
        }
        
        // Use the first path from the configuration
        std::string references_path = config_paths[0];
        
        // Verify the file exists
        if (!std::filesystem::exists(references_path)) {
            throw std::runtime_error("Could not find agent_dict.json file at configured path: " + references_path);
        }
        
        return references_path;
    }

    /**
     * Loads reference terms from a JSON file
     *
     * @param filename Path to the JSON file containing reference terms
     * @return Structured sets of reference terms by category
     * @throws std::runtime_error if the file cannot be loaded or parsed correctly
     */
    ReferenceTerms load_reference_terms(const std::string &filename) {
        ReferenceTerms references;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open references file: " + filename);
        }

        nlohmann::json json_data;
        file >> json_data;

        auto refs_opt = get_optional<nlohmann::json>(json_data, "references");
        if (!refs_opt) {
            throw std::runtime_error("Missing 'references' field in " + filename);
        }
        
        const auto &refs = *refs_opt;

        // Load each reference category
        auto self_refs_opt = get_optional<nlohmann::json>(refs, "self_references");
        if (!self_refs_opt) {
            throw std::runtime_error("Missing 'self_references' field in " + filename);
        }
        for (const auto &ref : *self_refs_opt) {
            references.self_references.insert(ref.get<std::string>());
        }

        auto general_llm_refs_opt = get_optional<nlohmann::json>(refs, "general_llm_references");
        if (!general_llm_refs_opt) {
            throw std::runtime_error("Missing 'general_llm_references' field in " + filename);
        }
        for (const auto &ref : *general_llm_refs_opt) {
            references.general_llm_references.insert(ref.get<std::string>());
        }

        auto specific_llm_refs_opt = get_optional<nlohmann::json>(refs, "specific_llm_references");
        if (!specific_llm_refs_opt) {
            throw std::runtime_error("Missing 'specific_llm_references' field in " + filename);
        }
        for (const auto &ref : *specific_llm_refs_opt) {
            references.specific_llm_references.insert(ref.get<std::string>());
        }

        auto system_prompt_terms_opt = get_optional<nlohmann::json>(refs, "system_prompt_terms");
        if (!system_prompt_terms_opt) {
            throw std::runtime_error("Missing 'system_prompt_terms' field in " + filename);
        }
        for (const auto &term : *system_prompt_terms_opt) {
            references.system_prompt_terms.insert(term.get<std::string>());
        }
        
        // Load request indicators
        auto request_indicators_opt = get_optional<nlohmann::json>(refs, "request_indicators");
        if (!request_indicators_opt) {
            throw std::runtime_error("Missing 'request_indicators' field in " + filename);
        }
        for (const auto &indicator : *request_indicators_opt) {
            references.request_indicators.insert(indicator.get<std::string>());
        }

        return references;
    }

    /**
     * Identifies the target of a system prompt request
     *
     * This function analyzes segmented Chinese text to determine if the user
     * is asking about system prompts and identifies who/what the request is targeting.
     *
     * @param segmented_words Vector of word segments from LAC model
     * @return The identified target of the system prompt request, or NONE if not a request
     */
    /**
     * Checks if a text contains system prompt related terms using both exact matching and semantic similarity
     *
     * @param text The text to check
     * @param cachedTerms Optional pointer to cached ReferenceTerms
     * @return true if the text contains system prompt related terms, false otherwise
     */
    bool contains_system_prompt_terms(const std::string &text, const ReferenceTerms* cachedTerms) {
        if (!cachedTerms) {
            throw std::invalid_argument("cached reference terms must be provided");
        }
        
        // Get reference phrases from cached terms
        std::vector<std::string> reference_phrases;
        reference_phrases.assign(cachedTerms->system_prompt_terms.begin(),
                                 cachedTerms->system_prompt_terms.end());

        // Get embedding for input text
        auto input_embedding = neural_network::get_model_set().text_embedding_model->embed(text);

        // Flatten to a 1D vector if necessary (depends on model output format)
        std::vector<float> input_vector;
        if (!input_embedding.empty() && !input_embedding[0].empty()) {
            // Use the first token embedding or the mean of all token embeddings
            input_vector = input_embedding[0]; // First token ([CLS]) often represents the whole sentence
        } else {
            spdlog::warn("Empty embedding for input text");
            return false;
        }

        // Compare with reference phrases
        float max_similarity = -1.0f;
        std::string best_match;

        for (const auto &phrase : reference_phrases) {
            auto phrase_embedding = neural_network::get_model_set().text_embedding_model->embed(phrase);

            if (!phrase_embedding.empty() && !phrase_embedding[0].empty()) {
                std::vector<float> phrase_vector = phrase_embedding[0];

                // Calculate cosine similarity
                float similarity = wheel::linalg_boost::cosine_similarity(input_vector.data(), phrase_vector.data(),
                                                                          input_vector.size());

                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    best_match = phrase;
                }
            }
        }

        // Similarity threshold (empirically set; can be tuned)
        // 较低阈值可捕捉更多语义改写, 较高阈值则更严格。根据嵌入模型特性这里先取 0.70。
        constexpr float SIMILARITY_THRESHOLD = 0.70f;

        if (max_similarity >= SIMILARITY_THRESHOLD) {
            spdlog::debug("System prompt term found via semantic similarity only: {} (similarity: {})", best_match,
                          max_similarity);
            return true;
        }

        spdlog::debug("No system prompt term found (max similarity: {})", max_similarity);
        return false;
    }

    std::optional<SystemPromptTarget>
    identify_system_prompt_target(const std::vector<neural_network::lac::OutputItem> &segmented_words, 
                                  const ReferenceTerms* cachedTerms) {
        if (!cachedTerms) {
            throw std::invalid_argument("cached reference terms must be provided");
        }
        
        // Reconstruct the full text for embedding-based comparison
        std::string full_text;
        for (const auto &item : segmented_words) {
            full_text += item.word;
        }

        // Check if this contains system prompt related terms using both methods
        bool has_system_prompt_term = contains_system_prompt_terms(full_text, cachedTerms);

        // If no system prompt term is found, not a system prompt request
        if (!has_system_prompt_term) {
            spdlog::debug("No system prompt term found, not a system prompt request");
            return std::nullopt;
        }

        std::string debug_output = "Segmented words: ";
        for (const auto &item : segmented_words) {
            debug_output += item.word + "(" + item.tag + ") ";
        }
        spdlog::debug(debug_output);

        // Filter words by their POS tags to reduce unnecessary comparisons
        std::unordered_map<std::string, std::vector<std::string>> tag_word_map;
        for (const auto &item : segmented_words) {
            tag_word_map[item.tag].push_back(item.word);
        }

        // Check if this is a question/request using cached request indicators
        bool is_request = false;
        const std::set<std::string>& request_indicators = cachedTerms->request_indicators;

        for (const auto &tag : {V_B, V_I, D_B, D_I, R_B, R_I, P_B, P_I}) {
            auto it = tag_word_map.find(tag);
            if (it == tag_word_map.end())
                continue;

            for (const auto &word : it->second) {
                for (const auto &indicator : request_indicators) {
                    if (word.find(indicator) != std::string::npos) {
                        is_request = true;
                        spdlog::debug("Request indicator found in {}: {}", tag, word);
                        break;
                    }
                }
                if (is_request)
                    break;
            }
            if (is_request)
                break;
        }

        if (!is_request) {
            spdlog::debug("Not a request for system prompt");
            return std::nullopt;
        }

        // Use the provided cached reference terms
        const std::set<std::string>& self_references = cachedTerms->self_references;
        const std::set<std::string>& general_llm_references = cachedTerms->general_llm_references;
        const std::set<std::string>& specific_llm_references = cachedTerms->specific_llm_references;

        // Check each category using both direct matching and embedding similarity
        bool has_self_reference = false;
        bool has_general_llm_reference = false;
        bool has_specific_llm_reference = false;

        // Helper function for semantic similarity check
        auto check_semantic_similarity = [&full_text](const std::set<std::string> &references) -> bool {
            // First try exact matching (fast path)
            for (const auto &ref : references) {
                if (full_text.find(ref) != std::string::npos) {
                    spdlog::debug("Reference found via exact match: {}", ref);
                    return true;
                }
            }

            // If no exact match, try semantic similarity
            constexpr float SIMILARITY_THRESHOLD = 0.8f;
            auto input_embedding = neural_network::get_model_set().text_embedding_model->embed(full_text);

            if (input_embedding.empty() || input_embedding[0].empty()) {
                return false;
            }

            std::vector<float> input_vector = input_embedding[0];

            for (const auto &ref : references) {
                auto ref_embedding = neural_network::get_model_set().text_embedding_model->embed(ref);
                if (!ref_embedding.empty() && !ref_embedding[0].empty()) {
                    std::vector<float> ref_vector = ref_embedding[0];

                    float similarity = wheel::linalg_boost::cosine_similarity(input_vector.data(), ref_vector.data(),
                                                                              input_vector.size());

                    if (similarity >= SIMILARITY_THRESHOLD) {
                        spdlog::debug("Reference found via semantic similarity: {} (similarity: {})", ref, similarity);
                        return true;
                    }
                }
            }

            return false;
        };

        // Check each category
        has_self_reference = check_semantic_similarity(self_references);
        has_general_llm_reference = check_semantic_similarity(general_llm_references);
        has_specific_llm_reference = check_semantic_similarity(specific_llm_references);

        // Determine the target (prioritize in this order)
        if (has_self_reference) {
            spdlog::debug("Target identified: SELF");
            return SystemPromptTarget::SELF;
        } else if (has_specific_llm_reference) {
            spdlog::debug("Target identified: SPECIFIC_LLM");
            return SystemPromptTarget::SPECIFIC_LLM;
        } else if (has_general_llm_reference) {
            spdlog::debug("Target identified: GENERAL_LLM");
            return SystemPromptTarget::GENERAL_LLM;
        } else {
            spdlog::debug("Target identified: GENERIC");
            return SystemPromptTarget::GENERIC;
        }
    }

    ActionAgent::ActionAgent(std::shared_ptr<bot_adapter::BotAdapter> adapter) 
        : adapter(std::move(adapter)), reference_terms(std::make_unique<ReferenceTerms>()) {
        load_reference_terms();
    }
    
    ActionAgent::~ActionAgent() = default;
    
    void ActionAgent::load_reference_terms() {
        std::string references_path = resolve_agent_dict_path();
        spdlog::debug("Loading reference terms from: {}", references_path);

        std::ifstream file(references_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open references file: " + references_path);
        }

        nlohmann::json json_data;
        file >> json_data;

        auto refs_opt = get_optional<nlohmann::json>(json_data, "references");
        if (!refs_opt) {
            throw std::runtime_error("Missing 'references' field in " + references_path);
        }
        
        const auto &refs = *refs_opt;

        // Load each reference category
        auto self_refs_opt = get_optional<nlohmann::json>(refs, "self_references");
        if (!self_refs_opt) {
            throw std::runtime_error("Missing 'self_references' field in " + references_path);
        }
        for (const auto &ref : *self_refs_opt) {
            reference_terms->self_references.insert(ref.get<std::string>());
        }

        auto general_llm_refs_opt = get_optional<nlohmann::json>(refs, "general_llm_references");
        if (!general_llm_refs_opt) {
            throw std::runtime_error("Missing 'general_llm_references' field in " + references_path);
        }
        for (const auto &ref : *general_llm_refs_opt) {
            reference_terms->general_llm_references.insert(ref.get<std::string>());
        }

        auto specific_llm_refs_opt = get_optional<nlohmann::json>(refs, "specific_llm_references");
        if (!specific_llm_refs_opt) {
            throw std::runtime_error("Missing 'specific_llm_references' field in " + references_path);
        }
        for (const auto &ref : *specific_llm_refs_opt) {
            reference_terms->specific_llm_references.insert(ref.get<std::string>());
        }

        auto system_prompt_terms_opt = get_optional<nlohmann::json>(refs, "system_prompt_terms");
        if (!system_prompt_terms_opt) {
            throw std::runtime_error("Missing 'system_prompt_terms' field in " + references_path);
        }
        for (const auto &term : *system_prompt_terms_opt) {
            reference_terms->system_prompt_terms.insert(term.get<std::string>());
        }
        
        // Load request indicators
        auto request_indicators_opt = get_optional<nlohmann::json>(refs, "request_indicators");
        if (!request_indicators_opt) {
            throw std::runtime_error("Missing 'request_indicators' field in " + references_path);
        }
        for (const auto &indicator : *request_indicators_opt) {
            reference_terms->request_indicators.insert(indicator.get<std::string>());
        }
    }

    ActionAgentResult ActionAgent::process_user_input(const bot_adapter::Sender &sender,
                                                      const std::string &user_input) {
        ActionAgentResult result;

        // Process input with LAC seg
        auto seg_res = neural_network::get_model_set().lac_model->run(user_input);

        // Check if user is requesting system prompt and identify the target using cached reference terms
        auto target = identify_system_prompt_target(seg_res, reference_terms.get());

        if (target.has_value()) {
            // User is asking for system prompt
            result.action_type = AgentActionType::SYSTEM_INFO_REPLY;

            // Different responses based on the target
            switch (target.value()) {
            case SystemPromptTarget::SELF:
                result.content_text = "您正在请求我(紫幻)的系统提示词信息。";
                break;
            case SystemPromptTarget::GENERAL_LLM:
                result.content_text = "您正在请求通用大语言模型的系统提示词信息。";
                break;
            case SystemPromptTarget::SPECIFIC_LLM:
                result.content_text = "您正在请求特定大语言模型的系统提示词信息。";
                break;
            case SystemPromptTarget::GENERIC:
                result.content_text = "您正在请求系统提示词的一般信息。";
                break;
            default:
                result.content_text = "您正在请求系统提示词信息。";
                break;
            }

            spdlog::info("User requested system prompt information for target: {}", static_cast<int>(target.value()));
        } else {
            // Continue with normal processing
            result.action_type = AgentActionType::TEXT_REPLY;
            result.content_text = "收到您的消息：" + user_input; // Placeholder response
        }

        return result;
    }
} // namespace agent