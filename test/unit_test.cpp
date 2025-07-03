#include "adapter_message.h"
#include "net.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <iterator>
#include <neural_network/nn.h>
#include <string>
#include <general-wheel-cpp/string_utils.hpp>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.

using namespace wheel;

TEST(UnitTest, MsgPropTest) {
    bot_adapter::PlainTextMessage msg{"abc"};
    GTEST_LOG_(INFO) << msg.to_json();
}

TEST(UnitTest, MainTest) { GTEST_LOG_(INFO) << "Main unit test"; }

TEST(UnitTest, ONNX_RUNTIME) {
    spdlog::info(Ort::GetVersionString());
    for (const auto &provider : Ort::GetAvailableProviders()) {
        spdlog::info("Available provider: {}", provider);
    }
}

TEST(UnitTest, BotAdapterTest) {
    spdlog::set_level(spdlog::level::debug);
    // bot_adapter::BotAdapter adapter{"ws://localhost:13378/all"};
    // const auto test_sender_id = 3507578481;
    // adapter.register_event<bot_adapter::GroupMessageEvent>([&adapter](const bot_adapter::GroupMessageEvent &e) {
    //     spdlog::info("接受到消息, 从sender: {}, group: {}", e.sender.id, e.group.name);
    //     if (e.sender.id == test_sender_id) {
    //         adapter.send_message(e.group,
    //                             bot_adapter::make_message_chain_list(
    //                                 bot_adapter::AtTargetMessage(e.sender.id),
    //                                 bot_adapter::PlainTextMessage(" test successed")
    //                                 ));
    //     }
    // });

    // adapter.start();
}

TEST(UnitTest, ReplaceString) {
    std::string str0 = "#联网 abc";
    std::string new_str0 = replace_str(str0, "#联网", "");
    EXPECT_EQ(new_str0, " abc");

    std::string str = "abc #联网(123)";
    std::string keyword = "#联网";
    std::string new_str = replace_keyword_and_parentheses_content(str, keyword, "");
    EXPECT_EQ(new_str, "abc ");
}

constexpr std::string_view test_url = "ws://localhost:13378/all";

TEST(UnitTest, GetMessageIdTest) {
    // spdlog::set_level(spdlog::level::debug);
    // bot_adapter::BotAdapter adapter{test_url};
    // adapter.get_message_id(2259, 790544814,
    //                        [](const nlohmann::json &data) { spdlog::debug("message data: {}", data.dump()); });

    // adapter.start();
}

// TEST(UnitTest, SendLongText) {
//     spdlog::set_level(spdlog::level::debug);
//     bot_adapter::BotAdapter adapter{test_url};
//     adapter.send_long_plain_text_replay(
//         bot_adapter::GroupSender(3507578481, "FredYakumo", std::nullopt, "", std::nullopt,
//                                  std::chrono::system_clock::now(),
//                                  bot_adapter::Group(790544814, "AIBot-800b灰度测试", "")),
//         "dsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibic"
//         "vxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffd"
//         "saudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicv"
//         "xngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffds"
//         "audhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvx"
//         "ngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsa"
//         "udhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxn"
//         "gbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsau"
//         "dhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxng"
//         "bnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaud"
//         "hasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngb"
//         "nudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudh"
//         "asudhasuidhuiashduihuivnxcuibicvxngbnudguidfgudfugidfngfdgdffffffffdsaudhasudhasuidhuiashduihuivnxcuibicvxngbn"
//         "udguidfgudfugidfngfdgdffffffff");

//     adapter.start();
// }

TEST(UnitTest, TestTokenizer) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("embedding.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "杀猪盘"};
    std::vector<neural_network::token_id_vec_with_mask_t> batch_token;
    std::vector<std::vector<float>> batch_embedding;

    for (const auto &text : batch_text) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        spdlog::info("\"{}\" tokens: [{}], mask: [{}]", text,
                     join_str(std::cbegin(token_mask.first), std::cend(token_mask.first), ",",
                              [](auto i) { return std::to_string(i); }),
                     join_str(std::cbegin(token_mask.second), std::cend(token_mask.second), ",",
                              [](auto i) { return std::to_string(i); }));
        auto embedding = embedder.embed(token_mask.first, token_mask.second);
        batch_token.emplace_back(std::move(token_mask));
        spdlog::info("Embedding dim: {}", embedding.size());
        spdlog::info("embedding:\n[{}]...", join_str(std::cbegin(embedding), std::cbegin(embedding) + 100, ",",
                                                     [](auto f) { return std::to_string(f); }));
        batch_embedding.emplace_back(std::move(embedding));
    }

    spdlog::info("cosine similarity: {}",
                 neural_network::cosine_similarity_with_padding(batch_embedding[0], batch_embedding[1]));
}

TEST(UnitTest, TestCosineSimilarity) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("embedding.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    std::vector<neural_network::token_id_vec_with_mask_t> batch_token;
    std::vector<std::vector<float>> batch_embedding;
    for (const auto &text : batch_text) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        spdlog::info("\"{}\" tokens: [{}], mask: [{}]", text,
                     join_str(std::cbegin(token_mask.first), std::cend(token_mask.first), ",",
                              [](auto i) { return std::to_string(i); }),
                     join_str(std::cbegin(token_mask.second), std::cend(token_mask.second), ",",
                              [](auto i) { return std::to_string(i); }));
        auto embedding = embedder.embed(token_mask.first, token_mask.second);
        batch_token.emplace_back(std::move(token_mask));
        spdlog::info("Embedding dim: {}", embedding.size());
        spdlog::info("embedding:\n[{}]...", join_str(std::cbegin(embedding), std::cbegin(embedding) + 100, ",",
                                                     [](auto f) { return std::to_string(f); }));
        batch_embedding.emplace_back(std::move(embedding));
    }

    neural_network::token_id_vec_with_mask_t target_token = tokenizer_wrapper.encode_with_mask("杀猪盘");
    auto target_embedding = embedder.embed(target_token.first, target_token.second);

    ncnn::Net cos_similarity_net;
    cos_similarity_net.load_param("cosine_sim.ncnn.param");
    cos_similarity_net.load_model("cosine_sim.ncnn.bin");

    std::vector<float> &target_emb = target_embedding;
    ncnn::Mat in0(1, target_emb.size(), target_emb.data()); // 形状: 1×嵌入维度

    size_t batch_size = batch_embedding.size();
    if (batch_size == 0) {
        spdlog::error("Batch embedding is empty");
        return;
    }
    size_t embedding_dim = batch_embedding[0].size();
    std::vector<float> in1_data;
    in1_data.reserve(batch_size * embedding_dim);
    for (const auto &emb : batch_embedding) {
        if (emb.size() != embedding_dim) {
            spdlog::error("Inconsistent embedding dimensions in batch");
            return;
        }
        in1_data.insert(in1_data.end(), emb.begin(), emb.end());
    }
    ncnn::Mat in1(batch_size, embedding_dim, in1_data.data());

    ncnn::Extractor ex = cos_similarity_net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in1);

    ncnn::Mat out;
    ex.extract("out0", out);
    spdlog::info("{}x{}", out.w, out.h);

    // for (int i = 0; i < out.w; ++i) {
    //     spdlog::info("Similarity[{}]: {}", i, out[i]);
    // }

    for (int i = 0; i < batch_size; ++i) {
        spdlog::info("Similarity[{}]: {}", i, out[i]);
    }
}

TEST(UnitTest, TestCosineSimilarityOnnx) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("embedding.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的价值"};
    std::vector<neural_network::token_id_vec_with_mask_t> batch_token;
    std::vector<std::vector<float>> batch_embedding;
    for (const auto &text : batch_text) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        // spdlog::info("\"{}\" tokens: [{}], mask: [{}]", text,
        //              join_str(std::cbegin(token_mask.first), std::cend(token_mask.first), ",",
        //                       [](auto i) { return std::to_string(i); }),
        //              join_str(std::cbegin(token_mask.second), std::cend(token_mask.second), ",",
        //                       [](auto i) { return std::to_string(i); }));
        auto embedding = embedder.embed(token_mask.first, token_mask.second);
        batch_token.emplace_back(std::move(token_mask));
        // spdlog::info("Embedding dim: {}", embedding.size());
        // spdlog::info("embedding:\n[{}]...", join_str(std::cbegin(embedding), std::cbegin(embedding) + 100, ",",
        //                                              [](auto f) { return std::to_string(f); }));
        batch_embedding.emplace_back(std::move(embedding));
    }

    spdlog::info("正在拷贝额外的向量");
    const auto emb = batch_embedding[0];
    for (size_t i = 0; i < 400000; ++i) {
        batch_embedding.push_back(emb);
    }

    spdlog::info("开始进行推理");

    neural_network::token_id_vec_with_mask_t target_token = tokenizer_wrapper.encode_with_mask("杀猪盘");
    auto target_embedding = embedder.embed(target_token.first, target_token.second);

    neural_network::CosineSimilarityONNXModel coreml_model{"cosine_sim.onnx", neural_network::get_onnx_session_opts_core_ml()};

    auto start = std::chrono::high_resolution_clock::now();
    auto res = coreml_model.inference(target_embedding, batch_embedding);
    auto end = std::chrono::high_resolution_clock::now();
    auto coreml_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // spdlog::info("Similarity (ONNX):");
    // for (const auto r : res) {
    //     spdlog::info(r);
    // }

    neural_network::CosineSimilarityONNXModel cpu_model{"cosine_sim.onnx"};

    start = std::chrono::high_resolution_clock::now();
    auto res_core_ml = cpu_model.inference(target_embedding, batch_embedding);
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // spdlog::info("Similarity (Core ML ONNX):");
    // for (const auto r : res_core_ml) {
    //     spdlog::info(r);
    // }

    spdlog::info("Inference time (Core ML): {} ms", coreml_duration);
    spdlog::info("Inference time (CPU): {} ms", cpu_duration);
}