#pragma once
/**
 * @brief Lightweight Chinese lexical analysis and part-of-speech tagging model interface
 *
 * Wrapper for Baidu LAC model providing Chinese text segmentation and POS tagging
 * Supports multiple encoding formats, adapted for the Neural Network project
 * @see https://github.com/baidu/lac/blob/master/c%2B%2B/include/lac.h
 */
#include "neural_network/nn.h"

#ifdef __USE_PADDLE_INFERENCE__
#include <paddle_inference_api.h>
#include <unordered_map>

namespace neural_network::lac {
    /**
     * Encoding type settings for the LAC model
     */
    enum CODE_TYPE {
        CODE_GB18030 = 0, // GB18030 Chinese standard encoding
        CODE_UTF8 = 1,    // UTF-8 encoding
    };

    /**
     * Model output structure for word segmentation and POS tagging
     */
    struct OutputItem {
        std::string word; // Segmented word
        std::string tag;  // Part-of-speech tag
    };

    class Customization;

    /**
     * @class LAC
     * @brief Lightweight Chinese lexical analysis and part-of-speech tagging implementation
     *
     * This class provides Chinese word segmentation and part-of-speech tagging capabilities.
     * It supports both single query and batch processing modes, as well as custom dictionaries
     * for domain-specific word recognition.
     */
    class LAC {
      public:
        /**
         * @brief Copy constructor
         * @param lac LAC instance to copy from
         */
        LAC(LAC &lac);

        /**
         * @brief Initialize LAC with model path and encoding type
         * @param model_path Path to the model directory containing model and vocabulary files
         * @param type Encoding type (default: UTF8)
         */
        LAC(const std::string &model_path, CODE_TYPE type = CODE_UTF8);

        /**
         * @brief Process a single text query
         * @param query Input text to analyze
         * @return Vector of word segments with their POS tags
         */
        std::vector<OutputItem> run(const std::string &query);

        /**
         * @brief Process multiple text queries in batch mode
         * @param query Vector of input texts to analyze
         * @return Vector of results for each query, containing word segments with their POS tags
         */
        std::vector<std::vector<OutputItem>> run(const std::vector<std::string> &query);

        /**
         * @brief Load custom dictionary for word segmentation customization
         * @param filename Path to the custom dictionary file
         * @return Status code (0 for success)
         */
        int load_customization(const std::string &filename);

      private:
        /**
         * @brief Convert string input to tensor format for model inference
         * @param querys Vector of query strings to be processed
         * @return Status code (0 for success)
         */
        int feed_data(const std::vector<std::string> &querys);

        /**
         * @brief Parse model output tags into structured results
         * @param tag_ids Vector of tag IDs from the model
         * @param words Vector of segmented words
         * @param result Output parameter to store the structured results
         * @return Status code (0 for success)
         */
        int parse_targets(const std::vector<std::string> &tag_ids, const std::vector<std::string> &words,
                          std::vector<OutputItem> &result);

    // No device place conversion needed with paddle_infer zero-copy handles

        // Encoding type (must match the encoding of dictionary files)
        CODE_TYPE m_code_type;

        // Intermediate variables
        std::vector<std::string> m_seq_words;
        std::vector<std::vector<std::string>> m_seq_words_batch;
        std::vector<std::vector<size_t>> m_lod;
        std::vector<std::string> m_labels;
        std::vector<OutputItem> m_results;
        std::vector<std::vector<OutputItem>> m_results_batch;

        // Conversion dictionaries
        std::shared_ptr<std::unordered_map<int64_t, std::string>> m_id2label_dict; // Maps tag IDs to tag names
        std::shared_ptr<std::unordered_map<std::string, std::string>> m_q2b_dict;  // Full-width to half-width mapping
        std::shared_ptr<std::unordered_map<std::string, int64_t>> m_word2id_dict;  // Maps words to vocabulary IDs
        int64_t m_oov_id;                                                          // Out-of-vocabulary ID

    // Paddle inference structures
    neural_network::Device m_device;                            // Neural network device type
    std::shared_ptr<paddle_infer::Predictor> m_predictor;       // Model predictor (shared_ptr to match API)
    std::unique_ptr<paddle_infer::Tensor> m_input_tensor;       // Input tensor handle
    std::unique_ptr<paddle_infer::Tensor> m_output_tensor;      // Output tensor handle

        // Custom dictionary for manual intervention
        std::shared_ptr<Customization> m_custom;
    };

} // namespace lac
#endif // __USE_PADDLE_INFERENCE__
