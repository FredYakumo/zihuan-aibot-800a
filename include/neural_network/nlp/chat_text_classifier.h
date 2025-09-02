#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace neural_network {
    namespace nlp {

        // Label IDs as constexpr
        namespace labels {
            constexpr int IDLE_CHAT = 0;         // 闲聊
            constexpr int ADVERTISEMENT = 1;     // 广告
            constexpr int POSITIVE = 2;          // 正面
            constexpr int NEGATIVE = 3;          // 负面
            constexpr int YOU = 4;               // 你
            constexpr int ME = 5;                // 我
            constexpr int MENTION_PERSON = 6;    // 提及明确人物
            constexpr int MODEL_INFO = 7;        // 模型信息
            constexpr int SYSTEM_PROMPT = 8;     // 系统提示词
            constexpr int CHAT_HISTORY = 9;      // 聊天记录
            constexpr int WEB_SEARCH = 10;       // 联网搜索
            constexpr int TIME = 11;             // 时间
            constexpr int LOCATION = 12;         // 地点
            constexpr int FOOD = 13;             // 吃饭
            constexpr int PREFERENCE = 14;       // 喜好
            constexpr int ART = 15;              // 艺术
            constexpr int TECHNOLOGY = 16;       // 科技
            constexpr int IT = 17;               // IT
            constexpr int HARDWARE = 18;         // 硬件
            constexpr int MATH = 19;             // 数学
            constexpr int SOCIETY = 20;          // 社会
            constexpr int POLITICS = 21;         // 政治
            constexpr int ECONOMY = 22;          // 经济
            constexpr int PROG_CPP = 23;         // 编程-C++
            constexpr int PROG_CSHARP = 24;      // 编程-C#
            constexpr int PROG_JAVA = 25;        // 编程-Java
            constexpr int PROG_PYTHON = 26;      // 编程-Python
            constexpr int PROG_JAVASCRIPT = 27;  // 编程-JavaScript
            constexpr int PROG_GO = 28;          // 编程-Go
            constexpr int PROG_RUST = 29;        // 编程-Rust
            constexpr int PROG_DATABASE = 30;    // 编程-数据库
            constexpr int PROG_ALGORITHM = 31;   // 编程-算法
            constexpr int PROG_IMPROVEMENT = 32; // 编程-提升
            constexpr int PROG_WORK = 33;        // 编程-工作
            constexpr int OPERATING_SYSTEM = 34; // 操作系统
            constexpr int AI = 35;               // 人工智能
            constexpr int GAME_MOBA = 36;        // 游戏-MOBA
            constexpr int GAME_MMORPG = 37;      // 游戏-MMORPG
            constexpr int GAME_FPS = 38;         // 游戏-FPS
            constexpr int GAME_GENSHIN = 39;     // 游戏-原神
            constexpr int GAME_ESPORTS = 40;     // 游戏-电竞
            constexpr int GAME_OTHER = 41;       // 游戏-其他
        }

        // Chinese label names as constexpr
        namespace label_names_zh {
            constexpr char IDLE_CHAT[] = "闲聊";
            constexpr char ADVERTISEMENT[] = "广告";
            constexpr char POSITIVE[] = "正面";
            constexpr char NEGATIVE[] = "负面";
            constexpr char YOU[] = "你";
            constexpr char ME[] = "我";
            constexpr char MENTION_PERSON[] = "提及明确人物";
            constexpr char MODEL_INFO[] = "模型信息";
            constexpr char SYSTEM_PROMPT[] = "系统提示词";
            constexpr char CHAT_HISTORY[] = "聊天记录";
            constexpr char WEB_SEARCH[] = "联网搜索";
            constexpr char TIME[] = "时间";
            constexpr char LOCATION[] = "地点";
            constexpr char FOOD[] = "吃饭";
            constexpr char PREFERENCE[] = "喜好";
            constexpr char ART[] = "艺术";
            constexpr char TECHNOLOGY[] = "科技";
            constexpr char IT[] = "IT";
            constexpr char HARDWARE[] = "硬件";
            constexpr char MATH[] = "数学";
            constexpr char SOCIETY[] = "社会";
            constexpr char POLITICS[] = "政治";
            constexpr char ECONOMY[] = "经济";
            constexpr char PROG_CPP[] = "编程-C++";
            constexpr char PROG_CSHARP[] = "编程-C#";
            constexpr char PROG_JAVA[] = "编程-Java";
            constexpr char PROG_PYTHON[] = "编程-Python";
            constexpr char PROG_JAVASCRIPT[] = "编程-JavaScript";
            constexpr char PROG_GO[] = "编程-Go";
            constexpr char PROG_RUST[] = "编程-Rust";
            constexpr char PROG_DATABASE[] = "编程-数据库";
            constexpr char PROG_ALGORITHM[] = "编程-算法";
            constexpr char PROG_IMPROVEMENT[] = "编程-提升";
            constexpr char PROG_WORK[] = "编程-工作";
            constexpr char OPERATING_SYSTEM[] = "操作系统";
            constexpr char AI[] = "人工智能";
            constexpr char GAME_MOBA[] = "游戏-MOBA";
            constexpr char GAME_MMORPG[] = "游戏-MMORPG";
            constexpr char GAME_FPS[] = "游戏-FPS";
            constexpr char GAME_GENSHIN[] = "游戏-原神";
            constexpr char GAME_ESPORTS[] = "游戏-电竞";
            constexpr char GAME_OTHER[] = "游戏-其他";
        }

        class ChatTextClassifier {
          public:
            ChatTextClassifier();
            ~ChatTextClassifier();

            void train(const std::vector<std::string> &training_data);
            std::string classify(const std::string &text);

            // Get all available label IDs
            static const std::unordered_set<int>& getAllLabelIds() {
                static const std::unordered_set<int> all_label_ids = {
                    labels::IDLE_CHAT,
                    labels::ADVERTISEMENT,
                    labels::POSITIVE,
                    labels::NEGATIVE,
                    labels::YOU,
                    labels::ME,
                    labels::MENTION_PERSON,
                    labels::MODEL_INFO,
                    labels::SYSTEM_PROMPT,
                    labels::CHAT_HISTORY,
                    labels::WEB_SEARCH,
                    labels::TIME,
                    labels::LOCATION,
                    labels::FOOD,
                    labels::PREFERENCE,
                    labels::ART,
                    labels::TECHNOLOGY,
                    labels::IT,
                    labels::HARDWARE,
                    labels::MATH,
                    labels::SOCIETY,
                    labels::POLITICS,
                    labels::ECONOMY,
                    labels::PROG_CPP,
                    labels::PROG_CSHARP,
                    labels::PROG_JAVA,
                    labels::PROG_PYTHON,
                    labels::PROG_JAVASCRIPT,
                    labels::PROG_GO,
                    labels::PROG_RUST,
                    labels::PROG_DATABASE,
                    labels::PROG_ALGORITHM,
                    labels::PROG_IMPROVEMENT,
                    labels::PROG_WORK,
                    labels::OPERATING_SYSTEM,
                    labels::AI,
                    labels::GAME_MOBA,
                    labels::GAME_MMORPG,
                    labels::GAME_FPS,
                    labels::GAME_GENSHIN,
                    labels::GAME_ESPORTS,
                    labels::GAME_OTHER
                };
                return all_label_ids;
            }

            // Get all Chinese label names
            static const std::unordered_map<int, std::string>& getLabelNameMap() {
                static const std::unordered_map<int, std::string> label_name_map = {
                    {labels::IDLE_CHAT, label_names_zh::IDLE_CHAT},
                    {labels::ADVERTISEMENT, label_names_zh::ADVERTISEMENT},
                    {labels::POSITIVE, label_names_zh::POSITIVE},
                    {labels::NEGATIVE, label_names_zh::NEGATIVE},
                    {labels::YOU, label_names_zh::YOU},
                    {labels::ME, label_names_zh::ME},
                    {labels::MENTION_PERSON, label_names_zh::MENTION_PERSON},
                    {labels::MODEL_INFO, label_names_zh::MODEL_INFO},
                    {labels::SYSTEM_PROMPT, label_names_zh::SYSTEM_PROMPT},
                    {labels::CHAT_HISTORY, label_names_zh::CHAT_HISTORY},
                    {labels::WEB_SEARCH, label_names_zh::WEB_SEARCH},
                    {labels::TIME, label_names_zh::TIME},
                    {labels::LOCATION, label_names_zh::LOCATION},
                    {labels::FOOD, label_names_zh::FOOD},
                    {labels::PREFERENCE, label_names_zh::PREFERENCE},
                    {labels::ART, label_names_zh::ART},
                    {labels::TECHNOLOGY, label_names_zh::TECHNOLOGY},
                    {labels::IT, label_names_zh::IT},
                    {labels::HARDWARE, label_names_zh::HARDWARE},
                    {labels::MATH, label_names_zh::MATH},
                    {labels::SOCIETY, label_names_zh::SOCIETY},
                    {labels::POLITICS, label_names_zh::POLITICS},
                    {labels::ECONOMY, label_names_zh::ECONOMY},
                    {labels::PROG_CPP, label_names_zh::PROG_CPP},
                    {labels::PROG_CSHARP, label_names_zh::PROG_CSHARP},
                    {labels::PROG_JAVA, label_names_zh::PROG_JAVA},
                    {labels::PROG_PYTHON, label_names_zh::PROG_PYTHON},
                    {labels::PROG_JAVASCRIPT, label_names_zh::PROG_JAVASCRIPT},
                    {labels::PROG_GO, label_names_zh::PROG_GO},
                    {labels::PROG_RUST, label_names_zh::PROG_RUST},
                    {labels::PROG_DATABASE, label_names_zh::PROG_DATABASE},
                    {labels::PROG_ALGORITHM, label_names_zh::PROG_ALGORITHM},
                    {labels::PROG_IMPROVEMENT, label_names_zh::PROG_IMPROVEMENT},
                    {labels::PROG_WORK, label_names_zh::PROG_WORK},
                    {labels::OPERATING_SYSTEM, label_names_zh::OPERATING_SYSTEM},
                    {labels::AI, label_names_zh::AI},
                    {labels::GAME_MOBA, label_names_zh::GAME_MOBA},
                    {labels::GAME_MMORPG, label_names_zh::GAME_MMORPG},
                    {labels::GAME_FPS, label_names_zh::GAME_FPS},
                    {labels::GAME_GENSHIN, label_names_zh::GAME_GENSHIN},
                    {labels::GAME_ESPORTS, label_names_zh::GAME_ESPORTS},
                    {labels::GAME_OTHER, label_names_zh::GAME_OTHER}
                };
                return label_name_map;
            }

          private:
            // Private member variables and methods
        };

    } // namespace nlp
} // namespace neural_network