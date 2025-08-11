#pragma once

/**
 * @brief Customization module for LAC Chinese lexical analysis and part-of-speech tagging
 *
 * Provides custom dictionary support for the LAC model to override default segmentation
 * and tagging behavior for domain-specific terms and phrases
 * @see https://github.com/baidu/lac/blob/master/c%2B%2B/include/lac_custom.h
 */
#ifdef __USE_PADDLE_INFERENCE__

#include <memory>
#include <string>
#include <vector>

#include "neural_network/text_model/lac/ahocorasick.h"
#include "neural_network/text_model/lac/lac_util.h"

namespace neural_network::lac {

    /**
     * @brief Customization term structure for word segmentation and tagging override
     *
     * Represents a single dictionary entry with its associated tags and segmentation information
     */
    struct customization_term {
        std::vector<std::string> tags; ///< POS tags for the term
        std::vector<int> split;        ///< Segmentation positions

        /**
         * @brief Construct a new customization term object
         *
         * @param tags POS tags for each segment of the term
         * @param split Segmentation positions within the term
         */
        customization_term(const std::vector<std::string> &tags, const std::vector<int> &split)
            : tags(tags), split(split) {}
    };

    /**
     * @brief Customization class for dictionary-based word segmentation and tagging override
     *
     * This class manages custom dictionary entries and provides methods to intervene
     * in LAC model's prediction results based on dictionary matches
     */
    class Customization {
      private:
        // Dictionary of customization terms with tags and segmentation information
        std::vector<customization_term> m_customization_dic;

        // Aho-Corasick automaton for efficient dictionary lookup
        AhoCorasick m_ac_dict;

      public:
        /**
         * @brief Construct a new Customization object with dictionary file
         *
         * @param customization_dic_path Path to the custom dictionary file
         */
        Customization(const std::string &customization_dic_path) { load_dict(customization_dic_path); }

        /**
         * @brief Load custom dictionary from file
         *
         * @param customization_dic_path Path to the custom dictionary file
         * @return RVAL Success or error code
         */
        RVAL load_dict(const std::string &customization_dic_path);

        /**
         * @brief Apply dictionary-based interventions to LAC prediction results
         *
         * @param seq_chars Input character sequence
         * @param tag_ids Tag IDs to be modified based on dictionary matches
         * @return RVAL Success or error code
         */
        RVAL parse_customization(const std::vector<std::string> &seq_chars, std::vector<std::string> &tag_ids);
    };

} // namespace neural_network::lac
#endif // __USE_PADDLE_INFERENCE__
