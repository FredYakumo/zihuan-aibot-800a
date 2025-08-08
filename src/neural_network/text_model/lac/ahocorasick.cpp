#ifdef __USE_PADDLE_INFERENCE__
#include "neural_network/text_model/lac/ahocorasick.h"
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
} // namespace lac
#endif // __USE_PADDLE_INFERENCE__
