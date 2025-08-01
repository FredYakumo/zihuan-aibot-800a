#pragma once

/**
 * @brief Aho-Corasick automaton implementation for efficient string matching
 * @see https://github.com/baidu/lac/blob/master/c%2B%2B/include/ahocorasick.h
 * Provides an implementation of the Aho-Corasick algorithm for efficient
 * multi-pattern string matching used by the LAC customization dictionary
 */

#include<vector>
#include<utility>
#include<string>

namespace lac {

/**
 * @brief Node structure for Aho-Corasick trie
 * 
 * Represents a node in the Aho-Corasick trie with character information,
 * transitions to child nodes, and failure links
 */
struct Node {
    std::vector<Node*> next;     ///< Transitions to child nodes
    std::string key;             ///< Character at this node
    int value;                   ///< Value associated with this node, -1 if none
    Node* fail;                  ///< Failure link to fallback node

    /**
     * @brief Construct a new Node object
     */
    Node() : value(-1), fail(nullptr) {}

    /**
     * @brief Get child node with specified character
     * 
     * @param str Character to look for
     * @return Node* Pointer to child node if found, nullptr otherwise
     */
    Node* get_child(const std::string &str);

    /**
     * @brief Add child node with specified character
     * 
     * @param str Character to add
     * @return Node* Pointer to new or existing child node
     */
    Node* add_child(const std::string &str);
};

/**
 * @brief Aho-Corasick automaton for multi-pattern string matching
 * 
 * Implements the Aho-Corasick algorithm for efficient matching of multiple
 * string patterns in a text simultaneously
 */
class AhoCorasick {
private:
    Node* m_root; ///< Root node of the trie

public:
    /**
     * @brief Construct a new AhoCorasick object
     */
    AhoCorasick() {
        m_root = new Node();
    }

    /**
     * @brief Destroy the AhoCorasick object
     */
    ~AhoCorasick();
    
    /**
     * @brief Insert a pattern into the automaton
     * 
     * @param chars Vector of characters representing the pattern
     * @param value Value to associate with this pattern
     */
    void insert(const std::vector<std::string> &chars, int value);

    /**
     * @brief Build failure links for the automaton
     * 
     * Creates failure transitions that are used when a pattern match fails
     */
    void make_fail();

    /**
     * @brief Search for patterns in a text
     * 
     * @param sentence Input text as a vector of characters
     * @param res Output vector of (start, end) indices of matches
     * @param backtrack Whether to allow backtracking
     * @return int Number of matches found
     */
    int search(const std::vector<std::string> &sentence, std::vector<std::pair<int, int>> &res, bool backtrack = false);
};

} // namespace lac