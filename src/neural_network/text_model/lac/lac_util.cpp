#ifdef __USE_PADDLE_INFERENCE__
#include "neural_network/text_model/lac/lac_util.h"
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
    std::ifstream fin(filepath);
    if (!fin) {
        spdlog::error("Open word2id dict file failed: {}", filepath);
        return FAILED;
    }
    
    std::string line;
    std::vector<std::string> tokens;
    while (getline(fin, line)) {
        split_tokens(line, "\t", tokens);
        if (tokens.size() != 2) {
            spdlog::warn("Invalid line in word2id dictionary: {}", line);
            continue;
        }
        
        int64_t id = std::stoll(tokens[0]);
        std::string word = tokens[1];
        kv_dict[word] = id;
    }
    fin.close();
    return SUCCESS;
}

/**
 * @brief Load character normalization dictionary
 */
RVAL load_q2b_dict(const std::string &filepath,
                   std::unordered_map<std::string, std::string> &kv_dict)
{
    std::ifstream fin(filepath);
    if (!fin) {
        spdlog::error("Open q2b dict file failed: {}", filepath);
        return FAILED;
    }
    
    std::string line;
    std::vector<std::string> tokens;
    while (getline(fin, line)) {
        split_tokens(line, "\t", tokens);
        if (tokens.size() != 2) {
            spdlog::warn("Invalid line in q2b dictionary: {}", line);
            continue;
        }
        
        std::string full_width = tokens[0];
        std::string half_width = tokens[1];
        kv_dict[full_width] = half_width;
    }
    fin.close();
    return SUCCESS;
}

/**
 * @brief Load ID-to-label mapping dictionary
 */
RVAL load_id2label_dict(const std::string &filepath,
                        std::unordered_map<int64_t, std::string> &kv_dict)
{
    std::ifstream fin(filepath);
    if (!fin) {
        spdlog::error("Open id2label dict file failed: {}", filepath);
        return FAILED;
    }
    
    std::string line;
    std::vector<std::string> tokens;
    while (getline(fin, line)) {
        split_tokens(line, "\t", tokens);
        if (tokens.size() != 2) {
            spdlog::warn("Invalid line in id2label dictionary: {}", line);
            continue;
        }
        
        int64_t id = std::stoll(tokens[0]);
        std::string label = tokens[1];
        kv_dict[id] = label;
    }
    fin.close();
    return SUCCESS;
}

int get_next_gb18030(const char *str)
{
    uint8_t one = (uint8_t)(*str);
    if (one < 0x80) {
        // 0xxx xxxx, 0～127, ASCII字符
        return 1;
    }
    
    uint8_t two = (uint8_t)(*(str + 1));
    if (one >= 0x81 && one <= 0xfe && two >= 0x40 && two <= 0xfe && two != 0x7f) {
        // 2字节-常用汉字
        return 2;
    }
    
    uint8_t three = (uint8_t)(*(str + 2));
    uint8_t four = (uint8_t)(*(str + 3));
    if (one >= 0x81 && one <= 0xfe && two >= 0x30 && two <= 0x39 
        && three >= 0x81 && three <= 0xfe && four >= 0x30 && four <= 0x39) {
        // 4字节-扩展汉字、生僻字
        return 4;
    }
    
    return 1;  // 默认情况
}

int get_next_utf8(const char *str)
{
    uint8_t first = (uint8_t)(*str);
    if (first < 0x80) {
        // 0xxx xxxx, 0～127, ASCII字符
        return 1;
    }
    
    // 获取首字节中开头连续1的个数，即为字节数
    int bytes = 0;
    if ((first & 0xE0) == 0xC0) {  // 110x xxxx, 2字节字符
        bytes = 2;
    } else if ((first & 0xF0) == 0xE0) {  // 1110 xxxx, 3字节字符
        bytes = 3;
    } else if ((first & 0xF8) == 0xF0) {  // 1111 0xxx, 4字节字符
        bytes = 4;
    } else {
        // 无效的UTF-8首字节
        return 1;
    }
    
    // 验证后续字节是否符合10xx xxxx格式
    for (int i = 1; i < bytes; i++) {
        if (((uint8_t)(*(str + i)) & 0xC0) != 0x80) {
            // 不符合格式，按1字节处理
            return 1;
        }
    }
    
    return bytes;
}

int get_next_word(const char *str, CODE_TYPE codetype)
{
    if (codetype == CODE_TYPE::CODE_GB18030) {
        return get_next_gb18030(str);
    } else {
        return get_next_utf8(str);
    }
}

RVAL split_words(const char *input, int len, CODE_TYPE codetype, std::vector<std::string> &words)
{
    if (input == nullptr || len <= 0) {
        return FAILED;
    }
    
    words.clear();
    const char *p = input;
    int offset = 0;
    
    while (offset < len) {
        int word_len = get_next_word(p + offset, codetype);
        if (word_len <= 0) {
            return FAILED;
        }
        
        words.push_back(std::string(p + offset, word_len));
        offset += word_len;
    }
    
    return SUCCESS;
}

RVAL split_words(const std::string &input, CODE_TYPE codetype, std::vector<std::string> &words)
{
    return split_words(input.c_str(), input.length(), codetype, words);
}

} // namespace lac
#endif // __USE_PADDLE_INFERENCE__
