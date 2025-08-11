#ifdef __USE_PADDLE_INFERENCE__
#include "neural_network/text_model/lac/ahocorasick.h"
#include <queue>

namespace neural_network::lac {

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

    void AhoCorasick::make_fail() {
        std::queue<Node *> q;
        // Root's immediate children: fail -> root
        for (auto child : m_root->next) {
            child->fail = m_root;
            q.push(child);
        }

        while (!q.empty()) {
            Node *u = q.front();
            q.pop();

            for (auto v : u->next) {
                // Find failure link for v
                Node *f = u->fail;
                while (f) {
                    Node *cand = f->get_child(v->key);
                    if (cand) {
                        v->fail = cand;
                        break;
                    }
                    if (f == m_root) {
                        v->fail = m_root;
                        break;
                    }
                    f = f->fail;
                }
                if (!v->fail) v->fail = m_root;
                q.push(v);
            }
        }
    }

    int AhoCorasick::search(const std::vector<std::string> &sentence, std::vector<std::pair<int, int>> &res,
                             bool backtrack) {
        res.clear();
        if (sentence.empty()) return 0;

        Node *state = m_root;
        for (int i = 0; i < static_cast<int>(sentence.size()); ++i) {
            const std::string &ch = sentence[i];
            // Follow transitions; if none, follow fail links
            Node *next_state = state->get_child(ch);
            while (!next_state && state != m_root) {
                state = state->fail ? state->fail : m_root;
                next_state = state->get_child(ch);
            }
            if (!next_state) {
                next_state = m_root->get_child(ch);
            }
            if (next_state) state = next_state; else state = m_root;

            // Report matches by following output via fail links
            Node *t = state;
            while (t && t != m_root) {
                if (t->value != -1) {
                    int end = i;
                    // We don't store length in node; rely on external customization dict's split.
                    // Here we just record the end index; begin will be computed by caller with split.
                    res.emplace_back(end, t->value);
                    if (!backtrack) break; // Only report the longest if not backtracking
                }
                t = t->fail;
            }
        }
        return static_cast<int>(res.size());
    }
} // namespace lac
#endif // __USE_PADDLE_INFERENCE__
