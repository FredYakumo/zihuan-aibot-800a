#pragma once
#include "../nn.h"
#include "tokenizer_wrapper.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

#ifdef __USE_LIBTORCH__
#include <torch/script.h>
#include <c10/core/Device.h>
#endif

namespace neural_network {
    constexpr size_t LTP_MAX_INPUT_LENGTH = 512;

    /**
     * @brief LTP tasks supported by the model
     */
    enum class LTPTask {
        CWS,  // Chinese Word Segmentation (分词)
        POS,  // Part-of-Speech Tagging (词性标注)
        NER,  // Named Entity Recognition (命名实体识别)
        SRL,  // Semantic Role Labeling (语义角色标注)
        DEP,  // Dependency Parsing (依存句法分析)
        SDP   // Semantic Dependency Parsing (语义依存分析)
    };

    /**
     * @brief LTP processing results
     */
    struct LTPResult {
        std::vector<std::vector<std::string>> cws;  // Word segmentation results
        std::vector<std::vector<std::string>> pos;  // POS tagging results
        std::vector<std::vector<std::map<std::string, std::string>>> ner;  // NER results
        std::vector<std::vector<std::map<std::string, std::string>>> srl;  // SRL results
        std::vector<std::vector<std::map<std::string, std::string>>> dep;  // DEP results
        std::vector<std::vector<std::map<std::string, std::string>>> sdp;  // SDP results
        emb_mat_t hidden_states;  // Hidden states for fallback mode
    };

#ifdef __USE_ONNX_RUNTIME__
    /**
     * @brief LTP model for Chinese NLP tasks using ONNX Runtime
     */
    class LTPModel {
      public:
        /**
         * @brief Construct a new LTP Model object
         *
         * @param model_path Path to onnx model
         * @param device Device to run model
         */
        LTPModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Process Chinese text with specified tasks
         * 
         * @param text Input text
         * @param tasks List of tasks to perform
         * @return LTPResult containing results for each task
         */
        LTPResult process_text(const std::string &text, const std::vector<LTPTask> &tasks = {LTPTask::CWS, LTPTask::POS, LTPTask::NER});

        /**
         * @brief Process multiple Chinese texts with specified tasks
         * 
         * @param texts List of input texts
         * @param tasks List of tasks to perform
         * @return LTPResult containing results for each task
         */
        LTPResult process_text(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks = {LTPTask::CWS, LTPTask::POS, LTPTask::NER});

        /**
         * @brief Get hidden states for given text (fallback mode)
         * 
         * @param text Input text
         * @return Hidden states matrix
         */
        emb_mat_t get_hidden_states(const std::string &text);

        /**
         * @brief Get hidden states for multiple texts (fallback mode)
         * 
         * @param texts List of input texts
         * @return Hidden states matrices
         */
        std::vector<emb_mat_t> get_hidden_states(const std::vector<std::string> &texts);

        /**
         * @brief Perform word segmentation
         * 
         * @param text Input text
         * @return Word segmentation result
         */
        std::vector<std::string> word_segmentation(const std::string &text);

        /**
         * @brief Perform part-of-speech tagging
         * 
         * @param text Input text
         * @return POS tagging result (words and their POS tags)
         */
        std::vector<std::pair<std::string, std::string>> pos_tagging(const std::string &text);

        /**
         * @brief Perform named entity recognition
         * 
         * @param text Input text
         * @return NER result (entities and their types)
         */
        std::vector<std::map<std::string, std::string>> named_entity_recognition(const std::string &text);

      private:
        std::unique_ptr<Ort::Session> session_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
        Device device_;
        bool use_fallback_mode_;  // True if using BERT-like fallback instead of full LTP
        
        // LTP-specific tokenizer
        std::shared_ptr<tokenizers::Tokenizer> ltp_tokenizer_;
        std::unique_ptr<TokenizerWrapper> ltp_tokenizer_wrapper_;
        
        // Helper methods
        LTPResult process_with_onnx(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks);
        LTPResult process_fallback(const std::vector<std::string> &texts);
        std::string ltp_task_to_string(LTPTask task);
        std::vector<std::vector<int64_t>> tokenize_texts_onnx(const std::vector<std::string> &texts);
    };
#endif

#ifdef __USE_LIBTORCH__
    /**
     * @brief LTP model for Chinese NLP tasks using LibTorch
     */
    class LTPModel {
      public:
        /**
         * @brief Construct a new LTP Model object
         *
         * @param model_path Path to torch script model
         * @param device Device to run model
         */
        LTPModel(const std::string &model_path, Device device = Device::CPU);

        /**
         * @brief Process Chinese text with specified tasks
         * 
         * @param text Input text
         * @param tasks List of tasks to perform
         * @return LTPResult containing results for each task
         */
        LTPResult process_text(const std::string &text, const std::vector<LTPTask> &tasks = {LTPTask::CWS, LTPTask::POS, LTPTask::NER});

        /**
         * @brief Process multiple Chinese texts with specified tasks
         * 
         * @param texts List of input texts
         * @param tasks List of tasks to perform
         * @return LTPResult containing results for each task
         */
        LTPResult process_text(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks = {LTPTask::CWS, LTPTask::POS, LTPTask::NER});

        /**
         * @brief Get hidden states for given text (fallback mode)
         * 
         * @param text Input text
         * @return Hidden states matrix
         */
        emb_mat_t get_hidden_states(const std::string &text);

        /**
         * @brief Get hidden states for multiple texts (fallback mode)
         * 
         * @param texts List of input texts
         * @return Hidden states matrices
         */
        std::vector<emb_mat_t> get_hidden_states(const std::vector<std::string> &texts);

        /**
         * @brief Perform word segmentation
         * 
         * @param text Input text
         * @return Word segmentation result
         */
        std::vector<std::string> word_segmentation(const std::string &text);

        /**
         * @brief Perform part-of-speech tagging
         * 
         * @param text Input text
         * @return POS tagging result (words and their POS tags)
         */
        std::vector<std::pair<std::string, std::string>> pos_tagging(const std::string &text);

        /**
         * @brief Perform named entity recognition
         * 
         * @param text Input text
         * @return NER result (entities and their types)
         */
        std::vector<std::map<std::string, std::string>> named_entity_recognition(const std::string &text);

      private:
        torch::jit::script::Module module_;
        torch::Device device_;
        bool use_fallback_mode_;  // True if using BERT-like fallback instead of full LTP
        
        // LTP-specific tokenizer
        std::shared_ptr<tokenizers::Tokenizer> ltp_tokenizer_;
        std::unique_ptr<TokenizerWrapper> ltp_tokenizer_wrapper_;
        
        // Helper methods
        LTPResult process_with_torch(const std::vector<std::string> &texts, const std::vector<LTPTask> &tasks);
        LTPResult process_fallback(const std::vector<std::string> &texts);
        std::string ltp_task_to_string(LTPTask task);
        std::pair<torch::Tensor, torch::Tensor> tokenize_texts(const std::vector<std::string> &texts);
    };
#endif

} // namespace neural_network
