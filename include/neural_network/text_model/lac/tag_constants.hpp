#pragma once

#include <string>

/**
 * @brief Constants for LAC (Lexical Analysis of Chinese) tag IDs and their meanings
 *
 * Defines constants for tag IDs used in the LAC model for part-of-speech tagging
 * and named entity recognition. These values match the tag.dic file from the model.
 */

namespace neural_network::lac {

    /**
     * @brief Part-of-speech and named entity recognition tag IDs
     *
     * These constants represent the tag IDs from the LAC model's tag.dic file.
     * Each tag has a Beginning (B) and Inside (I) variant, following the BIO tagging scheme.
     *
     * Note: Some tags share the same ID values but have different semantic meanings.
     * For example:
     * - NR_B (16) = PER_B (16) - Both represent the beginning of a person name
     * - NS_B (18) = LOC_B (18) - Both represent the beginning of a location name
     * - NT_B (20) = ORG_B (20) - Both represent the beginning of an organization name
     * - T_B (34) = TIME_B (34) - Both represent the beginning of a time expression
     */
    enum TagID {
        // Part-of-speech tags
        A_B = 0,  // Adjective - Beginning
        A_I = 1,  // Adjective - Inside
        AD_B = 2, // Adverbial adjective - Beginning
        AD_I = 3, // Adverbial adjective - Inside
        AN_B = 4, // Nominal adjective - Beginning
        AN_I = 5, // Nominal adjective - Inside
        C_B = 6,  // Conjunction - Beginning
        C_I = 7,  // Conjunction - Inside
        D_B = 8,  // Adverb - Beginning
        D_I = 9,  // Adverb - Inside
        F_B = 10, // Direction - Beginning
        F_I = 11, // Direction - Inside
        M_B = 12, // Numeral - Beginning
        M_I = 13, // Numeral - Inside
        N_B = 14, // Noun - Beginning
        N_I = 15, // Noun - Inside

        // Person name tags (POS and NER)
        NR_B = 16,  // Person name - Beginning (POS tag)
        NR_I = 17,  // Person name - Inside (POS tag)
        PER_B = 16, // Person entity - Beginning (NER tag, same as NR_B)
        PER_I = 17, // Person entity - Inside (NER tag, same as NR_I)

        // Location name tags (POS and NER)
        NS_B = 18,  // Location name - Beginning (POS tag)
        NS_I = 19,  // Location name - Inside (POS tag)
        LOC_B = 18, // Location entity - Beginning (NER tag, same as NS_B)
        LOC_I = 19, // Location entity - Inside (NER tag, same as NS_I)

        // Organization name tags (POS and NER)
        NT_B = 20,  // Organization name - Beginning (POS tag)
        NT_I = 21,  // Organization name - Inside (POS tag)
        ORG_B = 20, // Organization entity - Beginning (NER tag, same as NT_B)
        ORG_I = 21, // Organization entity - Inside (NER tag, same as NT_I)

        NW_B = 22, // Work name - Beginning
        NW_I = 23, // Work name - Inside
        NZ_B = 24, // Other proper noun - Beginning
        NZ_I = 25, // Other proper noun - Inside
        P_B = 26,  // Preposition - Beginning
        P_I = 27,  // Preposition - Inside
        Q_B = 28,  // Quantifier - Beginning
        Q_I = 29,  // Quantifier - Inside
        R_B = 30,  // Pronoun - Beginning
        R_I = 31,  // Pronoun - Inside
        S_B = 32,  // Location noun - Beginning
        S_I = 33,  // Location noun - Inside

        // Time tags (POS and NER)
        T_B = 34,    // Time noun - Beginning (POS tag)
        T_I = 35,    // Time noun - Inside (POS tag)
        TIME_B = 34, // Time entity - Beginning (NER tag, same as T_B)
        TIME_I = 35, // Time entity - Inside (NER tag, same as T_I)

        U_B = 36,  // Auxiliary - Beginning
        U_I = 37,  // Auxiliary - Inside
        V_B = 38,  // Verb - Beginning
        V_I = 39,  // Verb - Inside
        VD_B = 40, // Adverbial verb - Beginning
        VD_I = 41, // Adverbial verb - Inside
        VN_B = 42, // Nominal verb - Beginning
        VN_I = 43, // Nominal verb - Inside
        W_B = 44,  // Punctuation - Beginning
        W_I = 45,  // Punctuation - Inside
        XC_B = 46, // Process - Beginning
        XC_I = 47, // Process - Inside

        // Other
        O = 48 // Outside (not part of any entity)
    };

    /**
     * @brief Returns string representation of tag IDs
     *
     * @param tagId The tag ID to convert
     * @param useNERTags If true, use named entity recognition tags (PER, LOC, ORG, TIME) instead of POS tags for
     * entities
     * @return std::string The string representation of the tag
     */
    inline std::string tag_id_to_string(int tag_id, bool use_ner_tags = false) {
        switch (tag_id) {
        case A_B:
            return "a-B";
        case A_I:
            return "a-I";
        case AD_B:
            return "ad-B";
        case AD_I:
            return "ad-I";
        case AN_B:
            return "an-B";
        case AN_I:
            return "an-I";
        case C_B:
            return "c-B";
        case C_I:
            return "c-I";
        case D_B:
            return "d-B";
        case D_I:
            return "d-I";
        case F_B:
            return "f-B";
        case F_I:
            return "f-I";
        case M_B:
            return "m-B";
        case M_I:
            return "m-I";
        case N_B:
            return "n-B";
        case N_I:
            return "n-I";

        // Person name tags (NR_B/PER_B share the same ID)
        case NR_B:
            return use_ner_tags ? "PER-B" : "nr-B";
        case NR_I:
            return use_ner_tags ? "PER-I" : "nr-I";

        // Location name tags (NS_B/LOC_B share the same ID)
        case NS_B:
            return use_ner_tags ? "LOC-B" : "ns-B";
        case NS_I:
            return use_ner_tags ? "LOC-I" : "ns-I";

        // Organization name tags (NT_B/ORG_B share the same ID)
        case NT_B:
            return use_ner_tags ? "ORG-B" : "nt-B";
        case NT_I:
            return use_ner_tags ? "ORG-I" : "nt-I";

        case NW_B:
            return "nw-B";
        case NW_I:
            return "nw-I";
        case NZ_B:
            return "nz-B";
        case NZ_I:
            return "nz-I";
        case P_B:
            return "p-B";
        case P_I:
            return "p-I";
        case Q_B:
            return "q-B";
        case Q_I:
            return "q-I";
        case R_B:
            return "r-B";
        case R_I:
            return "r-I";
        case S_B:
            return "s-B";
        case S_I:
            return "s-I";

        // Time tags (T_B/TIME_B share the same ID)
        case T_B:
            return use_ner_tags ? "TIME-B" : "t-B";
        case T_I:
            return use_ner_tags ? "TIME-I" : "t-I";

        case U_B:
            return "u-B";
        case U_I:
            return "u-I";
        case V_B:
            return "v-B";
        case V_I:
            return "v-I";
        case VD_B:
            return "vd-B";
        case VD_I:
            return "vd-I";
        case VN_B:
            return "vn-B";
        case VN_I:
            return "vn-I";
        case W_B:
            return "w-B";
        case W_I:
            return "w-I";
        case XC_B:
            return "xc-B";
        case XC_I:
            return "xc-I";
        case O:
            return "O";
        default:
            return "Unknown";
        }
    }

    /**
     * @brief Check if a tag represents a noun (includes regular nouns and named entities)
     *
     * @param tag_id The tag ID to check
     * @return bool True if the tag represents a noun
     */
    inline bool is_noun_tag(int tag_id) {
        return (tag_id == N_B || tag_id == N_I || tag_id == NR_B || tag_id == NR_I || tag_id == NS_B ||
                tag_id == NS_I || tag_id == NT_B || tag_id == NT_I || tag_id == NW_B || tag_id == NW_I ||
                tag_id == NZ_B || tag_id == NZ_I);
    }

    /**
     * @brief Check if a tag represents the beginning of an entity
     *
     * @param tag_id The tag ID to check
     * @return bool True if the tag represents the beginning of an entity
     */
    inline bool is_beginning_tag(int tag_id) { return (tag_id % 2 == 0) && (tag_id != O); }

    /**
     * @brief Check if a tag represents a named entity (person, location, organization, time)
     *
     * Note: This uses the same IDs as some POS tags, but with different semantic interpretations:
     * - NR_B/PER_B (16) - Person entity
     * - NS_B/LOC_B (18) - Location entity
     * - NT_B/ORG_B (20) - Organization entity
     * - T_B/TIME_B (34) - Time entity
     *
     * @param tag_id The tag ID to check
     * @return bool True if the tag represents a named entity
     */
    inline bool is_named_entity_tag(int tag_id) {
        return (tag_id == NR_B || tag_id == NR_I || // Person (NR_B = PER_B, NR_I = PER_I)
                tag_id == NS_B || tag_id == NS_I || // Location (NS_B = LOC_B, NS_I = LOC_I)
                tag_id == NT_B || tag_id == NT_I || // Organization (NT_B = ORG_B, NT_I = ORG_I)
                tag_id == T_B || tag_id == T_I);    // Time (T_B = TIME_B, T_I = TIME_I)
    }

    /**
     * @brief Get the base tag type (without B/I distinction)
     *
     * @param tag_id The tag ID to process
     * @return std::string The base tag type (e.g., "n" for N_B or N_I)
     */
    inline std::string get_base_tag_type(int tag_id) {
        std::string full_tag = tag_id_to_string(tag_id);
        // Extract everything before the hyphen
        size_t hyphen_pos = full_tag.find('-');
        if (hyphen_pos != std::string::npos) {
            return full_tag.substr(0, hyphen_pos);
        }
        return full_tag;
    }

    /**
     * @brief String constants for LAC tags
     *
     * Provides direct string constants for each tag without coupling to the enum values.
     * This allows for direct string comparison and usage in contexts where the enum values
     * are not needed.
     */
    namespace tags {
        // Part-of-speech tag strings - Beginning variants
        constexpr const char *A_B = "a-B";   // Adjective - Beginning
        constexpr const char *AD_B = "ad-B"; // Adverbial adjective - Beginning
        constexpr const char *AN_B = "an-B"; // Nominal adjective - Beginning
        constexpr const char *C_B = "c-B";   // Conjunction - Beginning
        constexpr const char *D_B = "d-B";   // Adverb - Beginning
        constexpr const char *F_B = "f-B";   // Direction - Beginning
        constexpr const char *M_B = "m-B";   // Numeral - Beginning
        constexpr const char *N_B = "n-B";   // Noun - Beginning
        constexpr const char *NR_B = "nr-B"; // Person name - Beginning
        constexpr const char *NS_B = "ns-B"; // Location name - Beginning
        constexpr const char *NT_B = "nt-B"; // Organization name - Beginning
        constexpr const char *NW_B = "nw-B"; // Work name - Beginning
        constexpr const char *NZ_B = "nz-B"; // Other proper noun - Beginning
        constexpr const char *P_B = "p-B";   // Preposition - Beginning
        constexpr const char *Q_B = "q-B";   // Quantifier - Beginning
        constexpr const char *R_B = "r-B";   // Pronoun - Beginning
        constexpr const char *S_B = "s-B";   // Location noun - Beginning
        constexpr const char *T_B = "t-B";   // Time noun - Beginning
        constexpr const char *U_B = "u-B";   // Auxiliary - Beginning
        constexpr const char *V_B = "v-B";   // Verb - Beginning
        constexpr const char *VD_B = "vd-B"; // Adverbial verb - Beginning
        constexpr const char *VN_B = "vn-B"; // Nominal verb - Beginning
        constexpr const char *W_B = "w-B";   // Punctuation - Beginning
        constexpr const char *XC_B = "xc-B"; // Process - Beginning

        // Part-of-speech tag strings - Inside variants
        constexpr const char *A_I = "a-I";   // Adjective - Inside
        constexpr const char *AD_I = "ad-I"; // Adverbial adjective - Inside
        constexpr const char *AN_I = "an-I"; // Nominal adjective - Inside
        constexpr const char *C_I = "c-I";   // Conjunction - Inside
        constexpr const char *D_I = "d-I";   // Adverb - Inside
        constexpr const char *F_I = "f-I";   // Direction - Inside
        constexpr const char *M_I = "m-I";   // Numeral - Inside
        constexpr const char *N_I = "n-I";   // Noun - Inside
        constexpr const char *NR_I = "nr-I"; // Person name - Inside
        constexpr const char *NS_I = "ns-I"; // Location name - Inside
        constexpr const char *NT_I = "nt-I"; // Organization name - Inside
        constexpr const char *NW_I = "nw-I"; // Work name - Inside
        constexpr const char *NZ_I = "nz-I"; // Other proper noun - Inside
        constexpr const char *P_I = "p-I";   // Preposition - Inside
        constexpr const char *Q_I = "q-I";   // Quantifier - Inside
        constexpr const char *R_I = "r-I";   // Pronoun - Inside
        constexpr const char *S_I = "s-I";   // Location noun - Inside
        constexpr const char *T_I = "t-I";   // Time noun - Inside
        constexpr const char *U_I = "u-I";   // Auxiliary - Inside
        constexpr const char *V_I = "v-I";   // Verb - Inside
        constexpr const char *VD_I = "vd-I"; // Adverbial verb - Inside
        constexpr const char *VN_I = "vn-I"; // Nominal verb - Inside
        constexpr const char *W_I = "w-I";   // Punctuation - Inside
        constexpr const char *XC_I = "xc-I"; // Process - Inside

        // Named entity recognition tag strings
        constexpr const char *PER_B = "PER-B";   // Person entity - Beginning
        constexpr const char *PER_I = "PER-I";   // Person entity - Inside
        constexpr const char *LOC_B = "LOC-B";   // Location entity - Beginning
        constexpr const char *LOC_I = "LOC-I";   // Location entity - Inside
        constexpr const char *ORG_B = "ORG-B";   // Organization entity - Beginning
        constexpr const char *ORG_I = "ORG-I";   // Organization entity - Inside
        constexpr const char *TIME_B = "TIME-B"; // Time entity - Beginning
        constexpr const char *TIME_I = "TIME-I"; // Time entity - Inside

        // Other
        constexpr const char *O = "O"; // Outside (not part of any entity)

        /**
         * @brief Array of all part-of-speech beginning tags
         */
        constexpr const char *ALL_POS_B_TAGS[] = {A_B,  AD_B, AN_B, C_B, D_B, F_B, M_B, N_B, NR_B, NS_B, NT_B, NW_B,
                                                  NZ_B, P_B,  Q_B,  R_B, S_B, T_B, U_B, V_B, VD_B, VN_B, W_B,  XC_B};

        /**
         * @brief Array of all part-of-speech inside tags
         */
        constexpr const char *ALL_POS_I_TAGS[] = {A_I,  AD_I, AN_I, C_I, D_I, F_I, M_I, N_I, NR_I, NS_I, NT_I, NW_I,
                                                  NZ_I, P_I,  Q_I,  R_I, S_I, T_I, U_I, V_I, VD_I, VN_I, W_I,  XC_I};

        /**
         * @brief Array of all named entity beginning tags
         */
        constexpr const char *ALL_NER_B_TAGS[] = {PER_B, LOC_B, ORG_B, TIME_B};

        /**
         * @brief Array of all named entity inside tags
         */
        constexpr const char *ALL_NER_I_TAGS[] = {PER_I, LOC_I, ORG_I, TIME_I};

        /**
         * @brief Array of all noun-related tags (beginning variants)
         */
        constexpr const char *NOUN_B_TAGS[] = {N_B, NR_B, NS_B, NT_B, NW_B, NZ_B, PER_B, LOC_B, ORG_B};

        /**
         * @brief Array of all noun-related tags (inside variants)
         */
        constexpr const char *NOUN_I_TAGS[] = {N_I, NR_I, NS_I, NT_I, NW_I, NZ_I, PER_I, LOC_I, ORG_I};

        /**
         * @brief Check if a tag string is a beginning tag
         *
         * @param tag The tag string to check
         * @return bool True if the tag is a beginning tag
         */
        inline bool is_beginning_tag(const std::string &tag) { return tag.size() >= 2 && tag[tag.size() - 1] == 'B'; }

        /**
         * @brief Check if a tag string is an inside tag
         *
         * @param tag The tag string to check
         * @return bool True if the tag is an inside tag
         */
        inline bool is_inside_tag(const std::string &tag) { return tag.size() >= 2 && tag[tag.size() - 1] == 'I'; }

        /**
         * @brief Check if a tag string is a named entity tag
         *
         * @param tag The tag string to check
         * @return bool True if the tag is a named entity tag
         */
        inline bool is_named_entity_tag(const std::string &tag) {
            return tag.find("PER") == 0 || tag.find("LOC") == 0 || tag.find("ORG") == 0 || tag.find("TIME") == 0;
        }

        /**
         * @brief Check if a tag string is a noun-related tag
         *
         * @param tag The tag string to check
         * @return bool True if the tag is a noun-related tag
         */
        inline bool is_noun_tag(const std::string &tag) { return tag[0] == 'n' || is_named_entity_tag(tag); }

        /**
         * @brief Get the corresponding inside tag for a beginning tag
         *
         * @param beginning_tag The beginning tag string
         * @return std::string The corresponding inside tag
         */
        inline std::string get_inside_tag_from_beginning_tag(const std::string &beginning_tag) {
            if (!is_beginning_tag(beginning_tag)) {
                return beginning_tag; // Not a beginning tag, return as is
            }

            return beginning_tag.substr(0, beginning_tag.size() - 1) + "I";
        }

        /**
         * @brief Get the base tag type without B/I suffix
         *
         * @param tag The tag string
         * @return std::string The base tag type
         */
        inline std::string get_base_tag_type(const std::string &tag) {
            // Find the last hyphen in the tag
            size_t hyphen_pos = tag.find_last_of('-');
            if (hyphen_pos != std::string::npos) {
                return tag.substr(0, hyphen_pos);
            }
            return tag;
        }
    } // namespace tags

} // namespace neural_network::lac
