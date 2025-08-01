#pragma once
/**
 * @brief Utility functions for LAC Chinese lexical analysis
 * 
 * Provides helper functions for string manipulation, dictionary loading,
 * and character encoding operations used by the LAC model
 * @see https://github.com/baidu/lac/blob/master/c%2B%2B/include/lac_util.h
 */
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

#include "lac.h"

namespace lac {

/**
 * @brief Function return value codes
 */
enum RVAL
{
    SUCCESS = 0,   ///< Operation completed successfully
    FAILED = -1,   ///< Operation failed
};

/**
 * @brief Split a string into tokens using specified pattern
 * 
 * @param line Input string to be split
 * @param pattern Delimiter pattern
 * @param tokens Output vector to store the split tokens
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL split_tokens(const std::string &line, const std::string &pattern,
                  std::vector<std::string> &tokens);

/**
 * @brief Load word-to-ID mapping dictionary from file
 * 
 * @param filepath Path to the dictionary file
 * @param kv_dict Output unordered_map to store the word-to-ID mappings
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL load_word2id_dict(const std::string &filepath,
                       std::unordered_map<std::string, int64_t> &kv_dict);

/**
 * @brief Load character normalization dictionary (full-width to half-width mapping)
 * 
 * @param filepath Path to the dictionary file
 * @param kv_dict Output unordered_map to store the character mappings
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL load_q2b_dict(const std::string &filepath,
                   std::unordered_map<std::string, std::string> &kv_dict);

/**
 * @brief Load ID-to-label mapping dictionary for decoding
 * 
 * @param filepath Path to the dictionary file
 * @param kv_dict Output unordered_map to store the ID-to-label mappings
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL load_id2label_dict(const std::string &filepath,
                        std::unordered_map<int64_t, std::string> &kv_dict);

/**
 * @brief Get the length of the next character in a GB18030 encoded string
 * 
 * @param str Input string
 * @return Length in bytes of the next character
 */
int get_next_gb18030(const char *str);

/**
 * @brief Get the length of the next character in a UTF-8 encoded string
 * 
 * @param str Input string
 * @return Length in bytes of the next character
 */
int get_next_utf8(const char *str);

/**
 * @brief Get the length of the next character based on encoding type
 * 
 * @param str Input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @return Length in bytes of the next character
 */
int get_next_word(const char *str, CODE_TYPE codetype);

/**
 * @brief Split a string into individual characters based on encoding
 * 
 * @param input Input string buffer
 * @param len Length of the input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @param words Output vector to store the split characters
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL split_words(const char *input, int len, CODE_TYPE codetype, std::vector<std::string> &words);

/**
 * @brief Split a string into individual characters based on encoding
 * 
 * @param input Input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @param words Output vector to store the split characters
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL split_words(const std::string &input, CODE_TYPE codetype, std::vector<std::string> &words);

} // namespace lac