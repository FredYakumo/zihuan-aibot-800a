#include "neural_network/text_model/lac_util.h"
#include "spdlog/spdlog.h"

namespace lac {

/**
 * @brief Split a string into tokens using specified pattern
 * 
 * Splits the input string using the given pattern as delimiter
 * and stores the resulting tokens in the provided vector
 */
RVAL split_tokens(const std::string &line, const std::string &pattern,
                  std::vector<std::string> &tokens)
{
    if ("" == line || "" == pattern)
    {
        spdlog::debug("Split tokens failed: empty line or pattern");
        return FAILED;
    }

    tokens.clear();
    int pos = 0;
    int size = line.size();

    for (int i = 0; i < size; i++)
    {
        pos = line.find(pattern, i);

        if (-1 != pos)
        {
            tokens.push_back(line.substr(i, pos - i));
            i = pos + pattern.size() - 1;
        }
        else
        {
            tokens.push_back(line.substr(i));
            break;
        }
    } // end of for
    return SUCCESS;
}

/**
 * @brief Load word-to-ID mapping dictionary from file
 * 
 * Loads a dictionary from file where each line contains an ID and word
 * separated by a tab character, storing them in an unordered_map
 */
RVAL load_word2id_dict(const std::string &filepath,
                       std::unordered_map<std::string, int64_t> &kv_dict)
{
    kv_dict.clear();
    std::ifstream infile(filepath);
    if (infile.fail())
    {
        spdlog::error("Failed to open word2id dictionary file: {}", filepath);
        return FAILED;
    }

    std::string line = "";
    std::vector<std::string> tokens;
    while (std::getline(infile, line) && infile.good())
    {
        split_tokens(line, "\t", tokens);
        if ("" == line || 2 != tokens.size())
        {
            continue;
        }
        int64_t val = std::stoll(tokens[0]);
        std::string key = tokens[1];
        kv_dict[key] = val;
    }
    infile.close();
    spdlog::debug("Loaded word2id dictionary with {} entries", kv_dict.size());
    return SUCCESS;
}

/**
 * @brief Load character normalization dictionary (full-width to half-width mapping)
 * 
 * Loads a dictionary from file where each line contains a mapping between
 * full-width and half-width characters separated by a tab character
 */
RVAL load_q2b_dict(const std::string &filepath,
                   std::unordered_map<std::string, std::string> &kv_dict)
{
    kv_dict.clear();
    std::ifstream infile(filepath);
    if (infile.fail())
    {
        spdlog::error("Failed to open q2b dictionary file: {}", filepath);
        return FAILED;
    }

    std::string line = "";
    std::vector<std::string> tokens;
    while (std::getline(infile, line) && infile.good())
    {
        split_tokens(line, "\t", tokens);
        if ("" == line || 2 != tokens.size())
        {
            continue;
        }
        kv_dict[tokens[0]] = tokens[1];
    }
    infile.close();
    spdlog::debug("Loaded q2b dictionary with {} entries", kv_dict.size());
    return SUCCESS;
}

/**
 * @brief Load ID-to-label mapping dictionary for decoding
 * 
 * Loads a dictionary from file where each line contains an ID and label
 * separated by a tab character, storing them in an unordered_map
 */
RVAL load_id2label_dict(const std::string &filepath,
                        std::unordered_map<int64_t, std::string> &kv_dict)
{
    kv_dict.clear();
    std::ifstream infile(filepath);
    if (infile.fail())
    {
        spdlog::error("Failed to open id2label dictionary file: {}", filepath);
        return FAILED;
    }

    std::string line = "";
    std::vector<std::string> tokens;
    while (std::getline(infile, line) && infile.good())
    {
        split_tokens(line, "\t", tokens);
        if ("" == line || 2 != tokens.size())
        {
            continue;
        }
        int64_t key = std::stoll(tokens[0]);
        std::string val = tokens[1];
        kv_dict[key] = val;
    }
    infile.close();
    spdlog::debug("Loaded id2label dictionary with {} entries", kv_dict.size());
    return SUCCESS;
}

/**
 * @brief Get the length of the next character in a GB18030 encoded string
 * 
 * Analyzes the byte pattern to determine the character boundary in GB18030 encoding
 * 
 * @param str Input string
 * @return Length in bytes of the next character
 */
int get_next_gb18030(const char *str)
{
    unsigned char *str_in = (unsigned char *)str;
    if (str_in[0] < 0x80)
    {
        return 1;
    }
    if (str_in[0] >= 0x81 && str_in[0] <= 0xfe &&
        str_in[1] >= 0x40 && str_in[1] <= 0xFE && str_in[1] != 0x7F)
    {
        return 2;
    }
    if (str_in[0] >= 0x81 && str_in[0] <= 0xfe &&
        str_in[1] >= 0x30 && str_in[1] <= 0x39 &&
        str_in[2] >= 0x81 && str_in[2] <= 0xfe &&
        str_in[3] >= 0x30 && str_in[3] <= 0x39)
    {
        return 4;
    }
    return 0;
}

/**
 * @brief Get the length of the next character in a UTF-8 encoded string
 * 
 * Analyzes the byte pattern to determine the character boundary in UTF-8 encoding
 * 
 * @param str Input string
 * @return Length in bytes of the next character
 */
int get_next_utf8(const char *str)
{
    unsigned char *str_in = (unsigned char *)str;
    if (str_in[0] < 0x80)
    {
        return 1;
    }
    if (str_in[0] >= 0xC2 && str_in[0] < 0xE0 &&
        str_in[1] >> 6 == 2)
    {
        return 2;
    }
    if (str_in[0] >> 4 == 14 && str_in[1] >> 6 == 2 &&
        str_in[2] >> 6 == 2 && (str_in[0] > 0xE0 || str_in[1] >= 0xA0))
    {
        return 3;
    }
    if (str_in[0] >> 3 == 30 && str_in[1] >> 6 == 2 && str_in[2] >> 6 == 2 &&
        str_in[3] >> 6 == 2 && str_in[0] <= 0xF4 && (str_in[0] > 0xF0 || str_in[1] >= 0x90))
    {
        return 4;
    }
    return 0;
}

/**
 * @brief Get the length of the next character based on encoding type
 * 
 * Determines the character boundary length based on the specified encoding
 * 
 * @param str Input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @return Length in bytes of the next character
 */
int get_next_word(const char *str, CODE_TYPE codetype)
{
    int len = 0;
    switch (codetype)
    {
    case CODE_GB18030:
        len = get_next_gb18030(str);
        break;
    case CODE_UTF8:
        len = get_next_utf8(str);
        break;
    default:
        len = 0;
        break;
    }
    len = len == 0 ? 1 : len;
    return len;
}

/**
 * @brief Split a string into individual characters based on encoding
 * 
 * Divides the input string into individual characters according to the specified
 * encoding type and stores them in the provided vector
 * 
 * @param input Input string buffer
 * @param len Length of the input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @param words Output vector to store the split characters
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL split_words(const char *input, int len, CODE_TYPE codetype, std::vector<std::string> &words)
{
    words.clear();
    char *p = (char *)input;
    int temp_len = 0;
    std::string key;
    for (int i = 0; i < len; i += temp_len)
    {
        temp_len = get_next_word(p, codetype);
        key.assign(p, temp_len);
        words.push_back(key);
        p += temp_len;
    }
    return SUCCESS;
}

/**
 * @brief Split a string into individual characters based on encoding
 * 
 * Overloaded version that takes a std::string instead of a character buffer
 * 
 * @param input Input string
 * @param codetype Encoding type (GB18030 or UTF-8)
 * @param words Output vector to store the split characters
 * @return RVAL SUCCESS if successful, FAILED otherwise
 */
RVAL split_words(const std::string &input, CODE_TYPE codetype, std::vector<std::string> &words)
{
    const char *p = input.c_str();
    int len = input.length();
    return split_words(p, len, codetype, words);
}

} // namespace lac