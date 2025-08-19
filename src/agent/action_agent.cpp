#include "agent/action_agent.h"
#include "neural_network/model_set.h"
#include "neural_network/text_model/lac/tag_constants.hpp"
#include <optional>
#include <set>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <general-wheel-cpp/linalg_boost/linalg_boost.hpp>

using namespace neural_network::lac::tags;

namespace agent {

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
     * @return true if the text contains system prompt related terms, false otherwise
     */
    bool contains_system_prompt_terms(const std::string &text) {
        // System prompt 纯向量语义
        const std::vector<std::string> reference_phrases = {
            // Core bilingual phrases
            "系统提示词", "system prompt", "系统指令", "prompt template",
            // Common Chinese paraphrases
            "预设提示", "指令模板", "预设指令", "系统预设", "角色设定", "人设指令",
            // Model related directives
            "模型指令", "模型提示词",
            // Prompt engineering related
            "prompt engineering", "system instruction"
        };
        
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
                float similarity = wheel::linalg_boost::cosine_similarity(input_vector.data(), phrase_vector.data(), input_vector.size());
                
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
            spdlog::debug("System prompt term found via semantic similarity only: {} (similarity: {})", 
                          best_match, max_similarity);
            return true;
        }
        
        spdlog::debug("No system prompt term found (max similarity: {})", max_similarity);
        return false;
    }

    std::optional<SystemPromptTarget>
    identify_system_prompt_target(const std::vector<neural_network::lac::OutputItem> &segmented_words) {
        // Reconstruct the full text for embedding-based comparison
        std::string full_text;
        for (const auto &item : segmented_words) {
            full_text += item.word;
        }
        
        // Check if this contains system prompt related terms using both methods
        bool has_system_prompt_term = contains_system_prompt_terms(full_text);

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

        // Check if this is a question/request
        bool is_request = false;
        const std::set<std::string> request_indicators = {"告诉", "是什么", "查看", "知道", "获取",
                                                          "看看", "给我",   "说说", "展示", "输出"};

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

        // identify the target of the request
        
        // Self references
        const std::set<std::string> self_references = {"你的",   "你们的",   "紫幻的", "紫幻",
                                                       "自己的", "本机器人", "您的"};

        // General LLM references
        const std::set<std::string> general_llm_references = {"大模型的", "大模型",   "大语言模型", "语言模型",
                                                              "AI模型",   "人工智能", "AI的",       "LLM"};

        // Specific LLM references
        const std::set<std::string> specific_llm_references = {
            "ChatGPT", "GPT",     "Claude", "文心一言",  "星火", "通义千问", "讯飞星火", "Gemini",
            "Llama",   "Mistral", "OpenAI", "Anthropic", "百度", "阿里",     "腾讯",     "智谱"};

        // Check each category using both direct matching and embedding similarity
        bool has_self_reference = false;
        bool has_general_llm_reference = false;
        bool has_specific_llm_reference = false;
        
        // Helper function for semantic similarity check
        auto check_semantic_similarity = [&full_text](const std::set<std::string>& references) -> bool {
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
                    
                    float similarity = wheel::linalg_boost::cosine_similarity(input_vector.data(), ref_vector.data(), input_vector.size());
                    
                    if (similarity >= SIMILARITY_THRESHOLD) {
                        spdlog::debug("Reference found via semantic similarity: {} (similarity: {})", 
                                     ref, similarity);
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

    ActionAgentResult ActionAgent::process_user_input(const bot_adapter::Sender &sender,
                                                      const std::string &user_input) {
        ActionAgentResult result;

        // Process input with LAC seg
        auto seg_res = neural_network::get_model_set().lac_model->run(user_input);

        // Check if user is requesting system prompt and identify the target
        auto target = identify_system_prompt_target(seg_res);

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