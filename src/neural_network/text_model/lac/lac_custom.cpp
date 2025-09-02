#ifdef __USE_PADDLE_INFERENCE__

#include "neural_network/text_model/lac/lac_custom.h"
#include "spdlog/spdlog.h"
#include <fstream>

namespace neural_network::lac {

    /**
     * @brief Load custom dictionary from file
     *
     * Parses the dictionary file and builds the internal data structures
     * for custom word segmentation and tagging
     *
     * @param customization_dic_path Path to custom dictionary file
     * @return RVAL SUCCESS if successful, FAILED otherwise
     */
    RVAL Customization::load_dict(const std::string &customization_dic_path) {
        std::ifstream fin;
        fin.open(customization_dic_path.c_str());
        if (!fin) {
            spdlog::error("Load customization dictionary failed: {} does not exist", customization_dic_path);
            return FAILED;
        }

        std::string line;

        // Temporary variable for processing Chinese characters
        std::vector<std::string> line_vector;

        while (getline(fin, line)) {
            if (line.length() < 1) {
                continue;
            }
            if (split_tokens(line, " ", line_vector) < SUCCESS) {
                spdlog::error("Load customization dictionary failed: format error in line: {}", line);
                return FAILED;
            }

            // Read dictionary file and store in appropriate data structures
            std::vector<std::string> chars;
            std::vector<std::string> phrase;
            std::vector<std::string> tags;
            std::vector<int> split;
            int length = 0;
            for (auto kv : line_vector) {
                if (kv.length() < 1) {
                    continue;
                }
                // Split Chinese string into characters
                std::string word = kv.substr(0, kv.rfind("/"));
                if (kv.length() > 1) {
                    split_words(word, CODE_UTF8, chars);
                } else {
                    split_words(kv, CODE_UTF8, chars);
                }

                phrase.insert(phrase.end(), chars.begin(), chars.end());
                length += chars.size();
                std::string tag = (word.length() < kv.size()) ? kv.substr(kv.rfind("/") + 1) : "";
                tags.push_back(tag);
                split.push_back(length);
            }
            int value = m_customization_dic.size();
            m_customization_dic.push_back(customization_term(tags, split));
            m_ac_dict.insert(phrase, value);
        }
        m_ac_dict.make_fail();

        fin.close();

        spdlog::info("Loaded customization dictionary: {} entries", m_customization_dic.size());
        return SUCCESS;
    }

    /**
     * @brief Apply dictionary-based interventions to LAC prediction results
     *
     * Modifies tag IDs based on dictionary matches using the Aho-Corasick automaton
     *
     * @param seq_chars Input character sequence
     * @param tag_ids Tag IDs to be modified based on dictionary matches
     * @return RVAL SUCCESS if successful, FAILED otherwise
     */
    RVAL Customization::parse_customization(const std::vector<std::string> &seq_chars,
                                            std::vector<std::string> &tag_ids) {
        // Query results from AC automaton
        std::vector<std::pair<int, int>> ac_res;
        m_ac_dict.search(seq_chars, ac_res);

        int pre_begin = -1, pre_end = -1;
        for (auto ac_pair : ac_res) {
            int value = ac_pair.second;
            int length = m_customization_dic[value].split.back();
            int begin = ac_pair.first - length + 1;

            // Preprocess query results
            if (pre_begin < begin && pre_end >= begin) {
                continue;
            }
            pre_begin = begin;
            pre_end = ac_pair.first;

            // Correct tags in the annotation
            for (size_t i = 0; i < m_customization_dic[value].split.size(); i++) {
                std::string tag = m_customization_dic[value].tags[i];
                for (int j = 0; j < m_customization_dic[value].split[i]; j++) {
                    if (tag.length() < 1) {
                        tag_ids[begin][tag_ids[begin].length() - 1] = 'I';
                    } else {
                        tag_ids[begin] = tag + "-I";
                    }
                    begin++;
                }
            }

            // Correct word segmentation in the annotation
            begin = ac_pair.first - length + 1;
            tag_ids[begin][tag_ids[begin].length() - 1] = 'B';
            for (size_t i = 0; i < m_customization_dic[value].split.size(); i++) {
                size_t ind = begin + m_customization_dic[value].split[i];
                if (ind < tag_ids.size()) {
                    tag_ids[ind][tag_ids[ind].length() - 1] = 'B';
                }
            }
        }
        return SUCCESS;
    }

} // namespace lac

#endif // __USE_PADDLE_INFERENCE__