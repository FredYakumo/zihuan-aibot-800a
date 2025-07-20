#include "net.h"
#include "neural_network/text_model.h"
#include "neural_network/model_set.h"
#include "utils.h"
#include <general-wheel-cpp/string_utils.hpp>
#include <gtest/gtest.h>
#include <iterator>
#include <neural_network/nn.h>
#include <string>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.

using namespace wheel;

TEST(UnitTest, MainTest) { GTEST_LOG_(INFO) << "Main unit test"; }

#ifdef __USE_ONNX_RUNTIME__
TEST(UnitTest, ONNX_RUNTIME) {
    spdlog::info(Ort::GetVersionString());
    for (const auto &provider : Ort::GetAvailableProviders()) {
        spdlog::info("Available provider: {}", provider);
    }
}
#endif

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


#ifdef __USE_ONNX_RUNTIME__
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
#endif

#ifdef __USE_ONNX_RUNTIME__
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

    neural_network::CosineSimilarityModel coreml_model{"cosine_sim.onnx", neural_network::Device::CPU};

    auto start = std::chrono::high_resolution_clock::now();
    auto res = coreml_model.inference(target_embedding, batch_embedding);
    auto end = std::chrono::high_resolution_clock::now();
    auto coreml_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // spdlog::info("Similarity (ONNX):");
    // for (const auto r : res) {
    //     spdlog::info(r);
    // }

    neural_network::CosineSimilarityModel cpu_model{"cosine_sim.onnx"};

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

#endif

#ifdef __USE_LIBTORCH__

TEST(UnitTest, TestTextEmbeddingLibTorchMPS) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();
    
    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 50; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }

    spdlog::info("Single");
    auto start_single = std::chrono::high_resolution_clock::now();
    for (const auto &i : test_texts) {
        model_set.text_embedding_model->embed(i);
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    spdlog::info("Single embedding total time: {} ms", duration_single);

    spdlog::info("Batch");
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);
}



TEST(UnitTest, TestTextEmbeddingLibTorchMPSLargeBatch) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();
    
    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 2000; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);
}

TEST(UnitTest, TestTextEmbeddingLibTorchCPU) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();
    
    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 50; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }

    spdlog::info("Single");
    auto start_single = std::chrono::high_resolution_clock::now();
    for (const auto &i : test_texts) {
        model_set.text_embedding_model->embed(i);
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    spdlog::info("Single embedding total time: {} ms", duration_single);

    spdlog::info("Batch");
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);
}

TEST(UnitTest, TestCosineSimilarityLibTorch) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();
    
    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings for similarity test...");
    
    // 计算目标文本的嵌入向量
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 使用批量嵌入函数计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    auto batch_embeddings = model_set.text_embedding_model->embed(batch_text);
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed).count();
    
    spdlog::info("Batch embedding computation took {} ms", embed_duration);
    spdlog::info("Generated {} embeddings with dimension {}", batch_embeddings.size(), 
                 batch_embeddings.empty() ? 0 : batch_embeddings[0].size());
    
    for (size_t i = 0; i < batch_text.size() && i < batch_embeddings.size(); ++i) {
        spdlog::info("Text: \"{}\" - Embedding dim: {}", batch_text[i], batch_embeddings[i].size());
    }
    
    spdlog::info("Computing cosine similarities using LibTorch model...");
    
    // 使用LibTorch余弦相似度模型计算相似度
    auto start_time = std::chrono::high_resolution_clock::now();
    auto similarity_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    spdlog::info("LibTorch cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results:");
    
    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size(); ++i) {
        spdlog::info("  \"{}\" <-> \"{}\": {:.6f}", target_text, batch_text[i], similarity_scores[i]);
    }
    
    // 验证结果
    EXPECT_EQ(similarity_scores.size(), batch_text.size());
    
    // 验证相似度范围在[-1, 1]之间
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f);
        EXPECT_LE(score, 1.0f);
    }
    
    // 验证"杀猪"与自身的相似度最高
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");
    
    spdlog::info("LibTorch cosine similarity test completed successfully!");
}

#endif