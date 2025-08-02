#include "neural_network/text_model/ahocorasick.h"
#include <queue>

namespace lac {

    Node *Node::get_child(const std::string &str) {
        for (auto i : next) {
            if (i->key == str) {
                return i;
            }
        }
        return nullptr;
    }

    Node *Node::add_child(const std::string &str) {
        for (auto i : next) {
            if (i->key == str) {
                return i;
            }
        }
        Node *child = new Node();
        child->key = str;
        next.push_back(child);
        return child;
    }

    AhoCorasick::~AhoCorasick() {
        std::queue<Node *> que;
        que.push(m_root);

        // Breadth-first traversal to delete all nodes
        while (!que.empty()) {
            Node *tmp = que.front();
            que.pop();
            for (auto child : tmp->next) {
                que.push(child);
            }
            delete tmp;
        }
    }

    /* Add AC automaton item */
    void AhoCorasick::insert(const std::vector<std::string> &chars, int value) {
        if (chars.empty() || value < 0) {
            return;
        }

        Node *root = m_root;
        for (auto i : chars) {
            root = root->add_child(i);
        }
        root->value = value;
    }

    /* Generate failure links for the AC automaton */
    void AhoCorasick::make_fail() {

        m_root->fail = nullptr;
        std::queue<Node *> que;
        for (auto child : m_root->next) {
            child->fail = m_root;
            que.push(child);
        }

        /* Breadth-first traversal to set failure pointers */
        while (!que.empty()) {
            Node *current = que.front();
            que.pop();
            for (auto child : current->next) {
                Node *current_fail = current->fail;

                // If current node has a fail pointer, try to set its child's fail pointer
                while (current_fail) {
                    if (current_fail->get_child(child->key)) {
                        child->fail = current_fail->get_child(child->key);
                        break;
                    }
                    current_fail = current_fail->fail;
                }

                // If current node's fail pointer doesn't have a matching child, set child's fail to root
                if (current_fail == nullptr) {
                    child->fail = m_root;
                }

                que.push(child);
            }
        }
    }

    /* Search for patterns and return multi-pattern matching results */
    int AhoCorasick::search(const std::vector<std::string> &sentence, std::vector<std::pair<int, int>> &res,
                            bool backtrack) {
        Node *child = nullptr, *p = m_root;
        for (size_t i = 0; i < sentence.size(); i++) {
            child = p->get_child(sentence[i]);
            while (child == nullptr) {
                if (p == m_root) {
                    break;
                }
                p = p->fail;
                child = p->get_child(sentence[i]);
            }

            if (child) {
                p = child;

                while (child != m_root) {
                    // Pattern match found
                    if (child->value >= 0) {
                        res.push_back(std::make_pair(i, child->value));
                    }

                    // No backtracking, used for maximum length matching
                    if (!backtrack) {
                        break;
                    }

                    child = child->fail;
                }
            }
        }
        return 0;
    }

} // namespace lac