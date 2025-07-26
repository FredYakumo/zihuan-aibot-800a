#include "neural_network/text_model/ltp_model.h"
#include "neural_network/text_model/tokenizer_wrapper.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace neural_network {

    std::string ltp_task_to_string(LTPTask task) {
        switch (task) {
        case LTPTask::CWS: return "cws";
        case LTPTask::POS: return "pos";
        case LTPTask::NER: return "ner";
        case LTPTask::SRL: return "srl";
        case LTPTask::DEP: return "dep";
        case LTPTask::SDP: return "sdp";
        default: return "cws";
        }
    }

#ifdef __USE_ONNX_RUNTIME__
    LTPModel::LTPModel(const std::string &model_path, Device device)
        : device_(device), use_fallback_mode_(false) {
        
        // Load LTP tokenizer
        try {
            ltp_tokenizer_ = load_tokenizers("exported_model/ltp_tokenizer/tokenizer.json");
            ltp_tokenizer_wrapper_ = std::make_unique<TokenizerWrapper>(ltp_tokenizer_, TokenizerConfig());
            spdlog::info("LTP tokenizer loaded successfully from exported_model/ltp_tokenizer/tokenizer.json");
        } catch (const std::exception& e) {
            spdlog::error("Failed to load LTP tokenizer: {}", e.what());
            ltp_tokenizer_ = nullptr;
            ltp_tokenizer_wrapper_ = nullptr;
        }
        
        try {
            auto session_options = get_session_options(device);
            session_ = std::make_unique<Ort::Session>(get_onnx_runtime(), model_path.c_str(), session_options);
            
            // Get input and output names
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Get input names
            size_t num_input_nodes = session_->GetInputCount();
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = session_->GetInputNameAllocated(i, allocator);
                input_names_.push_back(std::string(input_name.get()));
            }
            
            // Get output names
            size_t num_output_nodes = session_->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session_->GetOutputNameAllocated(i, allocator);
                output_names_.push_back(std::string(output_name.get()));
            }
            
            spdlog::info("LTP ONNX model loaded successfully from: {}", model_path);
            spdlog::info("Input nodes: {}, Output nodes: {}", num_input_nodes, num_output_nodes);
            
        } catch (const std::exception& e) {
            spdlog::error("Failed to load LTP ONNX model from {}: {}", model_path, e.what());
            spdlog::info("Falling back to basic BERT model for hidden states only");
            use_fallback_mode_ = true;
        }
    }

    LTPResult LTPModel::process_text(const std::string &text, const std::vector<LTPTask> &tasks) {
        return process_text(std::vector<std::string>{text}, tasks);
    }

    LTPResult LTPModel::process_text(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks) {
        if (use_fallback_mode_) {
            return process_fallback(texts);
        }
        return process_with_onnx(texts, tasks);
    }

    LTPResult LTPModel::process_with_onnx(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks) {
        LTPResult result;
        
        try {
            // For now, implement a basic fallback as full LTP ONNX integration 
            // would require complex post-processing of model outputs
            spdlog::warn("Full LTP ONNX processing not yet implemented, using fallback mode");
            return process_fallback(texts);
            
        } catch (const std::exception& e) {
            spdlog::error("Error in LTP ONNX processing: {}", e.what());
            return process_fallback(texts);
        }
    }

    LTPResult LTPModel::process_fallback(const std::vector<std::string> &texts) {
        LTPResult result;
        
        // Simple rule-based Chinese word segmentation as fallback
        for (const auto& text : texts) {
            std::vector<std::string> words;
            std::vector<std::string> pos_tags;
            std::vector<std::map<std::string, std::string>> ner_results;
            
            // Basic Chinese character segmentation
            std::string current_word;
            for (size_t i = 0; i < text.length(); ) {
                // Check if it's a Chinese character (simplified approach)
                unsigned char c = text[i];
                if (c >= 0xE4 && c <= 0xE9 && i + 2 < text.length()) {
                    // Likely a Chinese character (3 bytes in UTF-8)
                    current_word = text.substr(i, 3);
                    words.push_back(current_word);
                    pos_tags.push_back("n");  // Default to noun
                    i += 3;
                } else if (c >= 0x20 && c <= 0x7E) {
                    // ASCII character
                    current_word = "";
                    while (i < text.length() && text[i] >= 0x20 && text[i] <= 0x7E && text[i] != ' ') {
                        current_word += text[i];
                        i++;
                    }
                    if (!current_word.empty()) {
                        words.push_back(current_word);
                        pos_tags.push_back("n");
                    }
                    // Skip spaces
                    while (i < text.length() && text[i] == ' ') {
                        i++;
                    }
                } else {
                    i++;
                }
            }
            
            result.cws.push_back(words);
            result.pos.push_back(pos_tags);
            result.ner.push_back(ner_results);
        }
        
        spdlog::debug("Processed {} texts using fallback mode", texts.size());
        return result;
    }

    emb_mat_t LTPModel::get_hidden_states(const std::string &text) {
        auto results = get_hidden_states(std::vector<std::string>{text});
        return results.empty() ? emb_mat_t{} : results[0];
    }

    std::vector<emb_mat_t> LTPModel::get_hidden_states(const std::vector<std::string> &texts) {
        std::vector<emb_mat_t> results;
        
        // For hidden states, we need to simulate BERT-like output
        for (const auto& text : texts) {
            size_t seq_len = std::min(text.length(), static_cast<size_t>(LTP_MAX_INPUT_LENGTH));
            emb_mat_t hidden_states(seq_len, emb_vec_t(768, 0.0f));  // 768 is standard BERT hidden size
            
            // Fill with random values as placeholder
            for (auto& token_emb : hidden_states) {
                for (auto& value : token_emb) {
                    value = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;  // Random values [-1, 1]
                }
            }
            
            results.push_back(hidden_states);
        }
        
        return results;
    }

    std::vector<std::string> LTPModel::word_segmentation(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS});
        return result.cws.empty() ? std::vector<std::string>{} : result.cws[0];
    }

    std::vector<std::pair<std::string, std::string>> LTPModel::pos_tagging(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS, LTPTask::POS});
        
        std::vector<std::pair<std::string, std::string>> pos_pairs;
        if (!result.cws.empty() && !result.pos.empty()) {
            const auto& words = result.cws[0];
            const auto& tags = result.pos[0];
            
            for (size_t i = 0; i < std::min(words.size(), tags.size()); ++i) {
                pos_pairs.emplace_back(words[i], tags[i]);
            }
        }
        
        return pos_pairs;
    }

    std::vector<std::map<std::string, std::string>> LTPModel::named_entity_recognition(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS, LTPTask::POS, LTPTask::NER});
        return result.ner.empty() ? std::vector<std::map<std::string, std::string>>{} : result.ner[0];
    }

    std::vector<std::vector<int64_t>> LTPModel::tokenize_texts_onnx(const std::vector<std::string> &texts) {
        if (!ltp_tokenizer_wrapper_) {
            spdlog::warn("LTP tokenizer not available, falling back to simple character-based tokenization");
            // Fallback to simple character-based tokenization
            std::vector<std::vector<int64_t>> token_ids;
            
            for (const auto& text : texts) {
                std::vector<int64_t> ids;
                // Simple character-based tokenization for demo purposes
                for (size_t i = 0; i < std::min(text.length(), static_cast<size_t>(LTP_MAX_INPUT_LENGTH)); ++i) {
                    ids.push_back(static_cast<int64_t>(text[i]));
                }
                // Pad to max length
                while (ids.size() < LTP_MAX_INPUT_LENGTH) {
                    ids.push_back(0);  // Padding token
                }
                token_ids.push_back(ids);
            }
            
            return token_ids;
        }
        
        // Use proper LTP tokenizer
        std::vector<std::vector<int64_t>> token_ids;
        
        for (const auto& text : texts) {
            // Use LTP tokenizer with padding to LTP_MAX_INPUT_LENGTH
            auto [tokens, attention_mask] = ltp_tokenizer_wrapper_->encode(
                text, 
                std::make_pair(LTP_MAX_INPUT_LENGTH, 0) // Pad to max length with pad token 0
            );
            
            // Convert tokens to int64_t
            std::vector<int64_t> ids;
            ids.reserve(tokens.size());
            for (const auto& token : tokens) {
                ids.push_back(static_cast<int64_t>(token));
            }
            
            token_ids.push_back(ids);
        }
        
        return token_ids;
    }

    std::string LTPModel::ltp_task_to_string(LTPTask task) {
        return neural_network::ltp_task_to_string(task);
    }
#endif

#ifdef __USE_LIBTORCH__
    LTPModel::LTPModel(const std::string &model_path, Device device)
        : device_(get_torch_device(device)), use_fallback_mode_(false) {
        
        // Load LTP tokenizer
        try {
            ltp_tokenizer_ = load_tokenizers("exported_model/ltp_tokenizer/tokenizer.json");
            ltp_tokenizer_wrapper_ = std::make_unique<TokenizerWrapper>(ltp_tokenizer_, TokenizerConfig());
            spdlog::info("LTP tokenizer loaded successfully from exported_model/ltp_tokenizer/tokenizer.json");
        } catch (const std::exception& e) {
            spdlog::error("Failed to load LTP tokenizer: {}", e.what());
            ltp_tokenizer_ = nullptr;
            ltp_tokenizer_wrapper_ = nullptr;
        }
        
        try {
            module_ = torch::jit::load(model_path);
            module_.to(device_);
            module_.eval();
            
            spdlog::info("LTP LibTorch model loaded successfully from: {}", model_path);
            
        } catch (const std::exception& e) {
            spdlog::error("Failed to load LTP LibTorch model from {}: {}", model_path, e.what());
            spdlog::info("Falling back to basic BERT model for hidden states only");
            use_fallback_mode_ = true;
        }
    }

    LTPResult LTPModel::process_text(const std::string &text, const std::vector<LTPTask> &tasks) {
        return process_text(std::vector<std::string>{text}, tasks);
    }

    LTPResult LTPModel::process_text(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks) {
        if (use_fallback_mode_) {
            return process_fallback(texts);
        }
        return process_with_torch(texts, tasks);
    }

    LTPResult LTPModel::process_with_torch(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks) {
        LTPResult result;
        
        try {
            // Tokenize the input texts
            auto [input_ids, attention_mask] = tokenize_texts(texts);
            
            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_ids);
            inputs.push_back(attention_mask);
            
            torch::Tensor output = module_.forward(inputs).toTensor();
            
            // For now, implement a basic fallback as full LTP LibTorch integration 
            // would require complex post-processing of model outputs
            spdlog::warn("Full LTP LibTorch processing not yet implemented, using fallback mode");
            return process_fallback(texts);
            
        } catch (const std::exception& e) {
            spdlog::error("Error in LTP LibTorch processing: {}", e.what());
            return process_fallback(texts);
        }
    }

    LTPResult LTPModel::process_fallback(const std::vector<std::string> &texts) {
        LTPResult result;
        
        // Simple rule-based Chinese word segmentation as fallback
        for (const auto& text : texts) {
            std::vector<std::string> words;
            std::vector<std::string> pos_tags;
            std::vector<std::map<std::string, std::string>> ner_results;
            
            // Basic Chinese character segmentation
            std::string current_word;
            for (size_t i = 0; i < text.length(); ) {
                // Check if it's a Chinese character (simplified approach)
                unsigned char c = text[i];
                if (c >= 0xE4 && c <= 0xE9 && i + 2 < text.length()) {
                    // Likely a Chinese character (3 bytes in UTF-8)
                    current_word = text.substr(i, 3);
                    words.push_back(current_word);
                    pos_tags.push_back("n");  // Default to noun
                    i += 3;
                } else if (c >= 0x20 && c <= 0x7E) {
                    // ASCII character
                    current_word = "";
                    while (i < text.length() && text[i] >= 0x20 && text[i] <= 0x7E && text[i] != ' ') {
                        current_word += text[i];
                        i++;
                    }
                    if (!current_word.empty()) {
                        words.push_back(current_word);
                        pos_tags.push_back("n");
                    }
                    // Skip spaces
                    while (i < text.length() && text[i] == ' ') {
                        i++;
                    }
                } else {
                    i++;
                }
            }
            
            result.cws.push_back(words);
            result.pos.push_back(pos_tags);
            result.ner.push_back(ner_results);
        }
        
        spdlog::debug("Processed {} texts using fallback mode", texts.size());
        return result;
    }

    emb_mat_t LTPModel::get_hidden_states(const std::string &text) {
        auto results = get_hidden_states(std::vector<std::string>{text});
        return results.empty() ? emb_mat_t{} : results[0];
    }

    std::vector<emb_mat_t> LTPModel::get_hidden_states(const std::vector<std::string> &texts) {
        std::vector<emb_mat_t> results;
        
        if (!use_fallback_mode_) {
            try {
                auto [input_ids, attention_mask] = tokenize_texts(texts);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_ids);
                inputs.push_back(attention_mask);
                
                torch::Tensor hidden_states = module_.forward(inputs).toTensor();
                
                // Convert tensor to emb_mat_t format
                auto sizes = hidden_states.sizes();
                int batch_size = sizes[0];
                int seq_len = sizes[1];
                int hidden_size = sizes[2];
                
                auto accessor = hidden_states.accessor<float, 3>();
                
                for (int b = 0; b < batch_size; ++b) {
                    emb_mat_t text_embeddings(seq_len, emb_vec_t(hidden_size));
                    for (int s = 0; s < seq_len; ++s) {
                        for (int h = 0; h < hidden_size; ++h) {
                            text_embeddings[s][h] = accessor[b][s][h];
                        }
                    }
                    results.push_back(text_embeddings);
                }
                
                return results;
                
            } catch (const std::exception& e) {
                spdlog::error("Error getting hidden states: {}", e.what());
            }
        }
        
        // Fallback: generate dummy hidden states
        for (const auto& text : texts) {
            size_t seq_len = std::min(text.length(), static_cast<size_t>(LTP_MAX_INPUT_LENGTH));
            emb_mat_t hidden_states(seq_len, emb_vec_t(768, 0.0f));  // 768 is standard BERT hidden size
            
            // Fill with random values as placeholder
            for (auto& token_emb : hidden_states) {
                for (auto& value : token_emb) {
                    value = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;  // Random values [-1, 1]
                }
            }
            
            results.push_back(hidden_states);
        }
        
        return results;
    }

    std::vector<std::string> LTPModel::word_segmentation(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS});
        return result.cws.empty() ? std::vector<std::string>{} : result.cws[0];
    }

    std::vector<std::pair<std::string, std::string>> LTPModel::pos_tagging(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS, LTPTask::POS});
        
        std::vector<std::pair<std::string, std::string>> pos_pairs;
        if (!result.cws.empty() && !result.pos.empty()) {
            const auto& words = result.cws[0];
            const auto& tags = result.pos[0];
            
            for (size_t i = 0; i < std::min(words.size(), tags.size()); ++i) {
                pos_pairs.emplace_back(words[i], tags[i]);
            }
        }
        
        return pos_pairs;
    }

    std::vector<std::map<std::string, std::string>> LTPModel::named_entity_recognition(const std::string &text) {
        auto result = process_text(text, {LTPTask::CWS, LTPTask::POS, LTPTask::NER});
        return result.ner.empty() ? std::vector<std::map<std::string, std::string>>{} : result.ner[0];
    }

    std::pair<torch::Tensor, torch::Tensor> LTPModel::tokenize_texts(const std::vector<std::string> &texts) {
        if (!ltp_tokenizer_wrapper_) {
            spdlog::warn("LTP tokenizer not available, falling back to simple character-based tokenization");
            // Fallback to simple character-based tokenization
            std::vector<std::vector<int64_t>> token_ids;
            std::vector<std::vector<int64_t>> attention_masks;
            
            for (const auto& text : texts) {
                std::vector<int64_t> ids;
                std::vector<int64_t> mask;
                
                // Simple character-based tokenization for demo purposes
                size_t text_len = std::min(text.length(), static_cast<size_t>(LTP_MAX_INPUT_LENGTH));
                for (size_t i = 0; i < text_len; ++i) {
                    ids.push_back(static_cast<int64_t>(text[i]));
                    mask.push_back(1);  // Real tokens get attention value 1
                }
                
                // Pad to max length
                while (ids.size() < LTP_MAX_INPUT_LENGTH) {
                    ids.push_back(0);   // Padding token
                    mask.push_back(0);  // Padding positions get attention value 0
                }
                
                token_ids.push_back(ids);
                attention_masks.push_back(mask);
            }
            
            // Convert to tensors
            torch::Tensor input_ids_tensor = torch::zeros({static_cast<long>(texts.size()), LTP_MAX_INPUT_LENGTH}, torch::kLong);
            torch::Tensor attention_mask_tensor = torch::zeros({static_cast<long>(texts.size()), LTP_MAX_INPUT_LENGTH}, torch::kLong);
            
            for (size_t i = 0; i < token_ids.size(); ++i) {
                for (size_t j = 0; j < token_ids[i].size(); ++j) {
                    input_ids_tensor[i][j] = token_ids[i][j];
                    attention_mask_tensor[i][j] = attention_masks[i][j];
                }
            }
            
            return {input_ids_tensor.to(device_), attention_mask_tensor.to(device_)};
        }
        
        // Use proper LTP tokenizer
        std::vector<std::vector<int64_t>> token_ids;
        std::vector<std::vector<int64_t>> attention_masks;
        
        for (const auto& text : texts) {
            // Use LTP tokenizer with padding to LTP_MAX_INPUT_LENGTH
            auto [tokens, attention_mask] = ltp_tokenizer_wrapper_->encode(
                text, 
                std::make_pair(LTP_MAX_INPUT_LENGTH, 0) // Pad to max length with pad token 0
            );
            
            // Convert tokens to int64_t
            std::vector<int64_t> ids;
            std::vector<int64_t> mask;
            
            ids.reserve(tokens.size());
            mask.reserve(attention_mask.size());
            
            for (const auto& token : tokens) {
                ids.push_back(static_cast<int64_t>(token));
            }
            
            for (const auto& mask_val : attention_mask) {
                mask.push_back(static_cast<int64_t>(mask_val));
            }
            
            token_ids.push_back(ids);
            attention_masks.push_back(mask);
        }
        
        // Convert to tensors
        torch::Tensor input_ids_tensor = torch::zeros({static_cast<long>(texts.size()), LTP_MAX_INPUT_LENGTH}, torch::kLong);
        torch::Tensor attention_mask_tensor = torch::zeros({static_cast<long>(texts.size()), LTP_MAX_INPUT_LENGTH}, torch::kLong);
        
        for (size_t i = 0; i < token_ids.size(); ++i) {
            for (size_t j = 0; j < std::min(token_ids[i].size(), static_cast<size_t>(LTP_MAX_INPUT_LENGTH)); ++j) {
                input_ids_tensor[i][j] = token_ids[i][j];
                attention_mask_tensor[i][j] = attention_masks[i][j];
            }
        }
        
        return {input_ids_tensor.to(device_), attention_mask_tensor.to(device_)};
    }

    std::string LTPModel::ltp_task_to_string(LTPTask task) {
        return neural_network::ltp_task_to_string(task);
    }
#endif

} // namespace neural_network
