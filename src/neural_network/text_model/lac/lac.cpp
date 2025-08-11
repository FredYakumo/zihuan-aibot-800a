#ifdef __USE_PADDLE_INFERENCE__

#include "neural_network/text_model/lac/lac.h"

#include "neural_network/text_model/lac/lac_custom.h"
#include "neural_network/text_model/lac/lac_util.h"

#include <paddle_inference_api.h>
#include <spdlog/spdlog.h>

namespace neural_network::lac {

    /**
     * @brief LAC constructor: initializes model and loads dictionaries
     * @param model_path Path to the directory containing model and vocabulary files
     * @param type Encoding type (UTF8 or GB18030)
     */
    LAC::LAC(const std::string &model_path, CODE_TYPE type)
        : m_code_type(type), m_lod(std::vector<std::vector<size_t>>(1)),
          m_id2label_dict(new std::unordered_map<int64_t, std::string>),
          m_q2b_dict(new std::unordered_map<std::string, std::string>),
          m_word2id_dict(new std::unordered_map<std::string, int64_t>), m_custom(nullptr) {

        // Load dictionaries
        std::string word_dict_path = model_path + "/conf/word.dic";
        load_word2id_dict(word_dict_path, *m_word2id_dict);
        std::string q2b_dict_path = model_path + "/conf/q2b.dic";
        load_q2b_dict(q2b_dict_path, *m_q2b_dict);
        std::string label_dict_path = model_path + "/conf/tag.dic";
        load_id2label_dict(label_dict_path, *m_id2label_dict);

        // Use paddle_infer::Config to load and optimize model
        m_device = neural_network::Device::CPU;
        paddle_infer::Config config;
        config.SetModel(model_path + "/model");
        config.DisableGpu();
        config.DisableGlogInfo();
        config.SetCpuMathLibraryNumThreads(1);
        // ZeroCopy API is default via GetInputHandle/GetOutputHandle
        {
            m_predictor = paddle_infer::CreatePredictor(config);
        }

        // Initialize input/output variables
        auto input_names = m_predictor->GetInputNames();
        m_input_tensor = m_predictor->GetInputHandle(input_names[0]);
        auto output_names = m_predictor->GetOutputNames();
        m_output_tensor = m_predictor->GetOutputHandle(output_names[0]);
        m_oov_id = m_word2id_dict->size() - 1;
        auto word_iter = m_word2id_dict->find("OOV");
        if (word_iter != m_word2id_dict->end()) {
            m_oov_id = word_iter->second;
        }
    }

    /**
     * @brief Copy constructor for thread-safe replication
     * @param lac LAC instance to copy from
     */
    LAC::LAC(LAC &lac)
        : m_code_type(lac.m_code_type), m_lod(std::vector<std::vector<size_t>>(1)),
          m_id2label_dict(lac.m_id2label_dict), m_q2b_dict(lac.m_q2b_dict), m_word2id_dict(lac.m_word2id_dict),
          m_oov_id(lac.m_oov_id), m_device(lac.m_device), m_predictor(lac.m_predictor), m_custom(lac.m_custom) {
        auto input_names = m_predictor->GetInputNames();
        m_input_tensor = m_predictor->GetInputHandle(input_names[0]);
        auto output_names = m_predictor->GetOutputNames();
        m_output_tensor = m_predictor->GetOutputHandle(output_names[0]);
    }

    /**
     * @brief Load user custom dictionary
     * @param filename Path to the custom dictionary file
     * @return Status code (0 for success)
     */
    int LAC::load_customization(const std::string &filename) {
        // Note: Multi-threaded hot loading can cause issues
        // Multiple threads share custom dictionary
        m_custom = std::make_shared<Customization>(filename);
        return 0;
    }

    /**
     * @brief Convert string input to tensor format
     * @param querys Vector of query strings to process
     * @return Status code (0 for success)
     */
    int LAC::feed_data(const std::vector<std::string> &querys) {
        m_seq_words_batch.clear();
        m_lod[0].clear();

        m_lod[0].push_back(0);
        int shape = 0;
        for (size_t i = 0; i < querys.size(); ++i) {
            split_words(querys[i], m_code_type, m_seq_words);
            m_seq_words_batch.push_back(m_seq_words);
            shape += m_seq_words.size();
            m_lod[0].push_back(shape);
        }

        m_input_tensor->SetLoD(m_lod);
        m_input_tensor->Reshape({shape, 1});

        // Prepare CPU buffer and copy
        std::vector<int64_t> input_buf;
        input_buf.reserve(shape);

        for (size_t i = 0; i < m_seq_words_batch.size(); ++i) {
            for (size_t j = 0; j < m_seq_words_batch[i].size(); ++j) {
                // Normalize characters (full-width to half-width conversion)
                std::string word = m_seq_words_batch[i][j];
                auto q2b_iter = m_q2b_dict->find(word);
                if (q2b_iter != m_q2b_dict->end()) {
                    word = q2b_iter->second;
                }

                // Map word to vocabulary ID
                int64_t word_id = m_oov_id;
                auto word_iter = m_word2id_dict->find(word);
                if (word_iter != m_word2id_dict->end()) {
                    word_id = word_iter->second;
                }
                input_buf.push_back(word_id);
            }
        }
        m_input_tensor->CopyFromCpu(input_buf.data());
        return 0;
    }

    /**
     * @brief Process output tags into structured results
     * @param tags Vector of tags from model output
     * @param words Vector of input words
     * @param result Output parameter to store structured results
     * @return Status code (0 for success)
     */
    int LAC::parse_targets(const std::vector<std::string> &tags, const std::vector<std::string> &words,
                           std::vector<OutputItem> &result) {
        result.clear();
        for (size_t i = 0; i < tags.size(); ++i) {
            // Push a new word for a B/S tag, otherwise append to the previous word
            if (result.empty() || tags[i].rfind("B") == tags[i].length() - 1 ||
                tags[i].rfind("S") == tags[i].length() - 1) {
                OutputItem output_item;
                output_item.word = words[i];
                output_item.tag = tags[i].substr(0, tags[i].length() - 2);
                result.push_back(output_item);
            } else {
                result[result.size() - 1].word += words[i];
            }
        }
        return 0;
    }

    /**
     * @brief Process a single text query
     * @param query Input text to analyze
     * @return Vector of word segments with their POS tags
     */
    std::vector<OutputItem> LAC::run(const std::string &query) {
        std::vector<std::string> query_vector = std::vector<std::string>({query});
        auto result = run(query_vector);
        return result[0];
    }

    /**
     * @brief Process multiple text queries in batch mode
     * @param querys Vector of input texts to analyze
     * @return Vector of results for each query, containing word segments with their POS tags
     */
    std::vector<std::vector<OutputItem>> LAC::run(const std::vector<std::string> &querys) {
        // Feed data to the model
        feed_data(querys);
        m_predictor->Run();

        // Decode model output
        std::vector<int64_t> output_buf;
        auto out_shape = m_output_tensor->shape();
        size_t output_size = 1;
        for (auto dim : out_shape)
            output_size *= static_cast<size_t>(dim);
        output_buf.resize(output_size);
        m_output_tensor->CopyToCpu(output_buf.data());

        m_labels.clear();
        m_results_batch.clear();

        for (size_t i = 0; i < m_lod[0].size() - 1; ++i) {
            // Extract labels for current sequence
            for (size_t j = 0; j < m_lod[0][i + 1] - m_lod[0][i]; ++j) {
                int64_t cur_label_id = output_buf[m_lod[0][i] + j];
                auto it = m_id2label_dict->find(cur_label_id);
                m_labels.push_back(it->second);
            }

            // Apply custom dictionary overrides if available
            if (m_custom) {
                m_custom->parse_customization(m_seq_words_batch[i], m_labels);
            }

            // Parse tags into structured results
            parse_targets(m_labels, m_seq_words_batch[i], m_results);
            m_labels.clear();
            m_results_batch.push_back(m_results);
        }

        return m_results_batch;
    }

} // namespace neural_network::lac
#endif // __USE_PADDLE_INFERENCE__