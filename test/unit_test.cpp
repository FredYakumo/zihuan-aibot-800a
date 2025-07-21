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
TEST(UnitTest, TestCosineSimilarityONNX) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings for similarity test...");
    
    // 计算目标文本的嵌入向量
    auto target_embedding = embedder.embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> batch_embeddings;
    for (const auto &text : batch_text) {
        auto embedding = embedder.embed(text);
        batch_embeddings.emplace_back(std::move(embedding));
    }
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed).count();
    
    spdlog::info("Batch embedding computation took {} ms", embed_duration);
    spdlog::info("Generated {} embeddings with dimension {}", batch_embeddings.size(), 
                 batch_embeddings.empty() ? 0 : batch_embeddings[0].size());
    
    for (size_t i = 0; i < batch_text.size() && i < batch_embeddings.size(); ++i) {
        spdlog::info("Text: \"{}\" - Embedding dim: {}", batch_text[i], batch_embeddings[i].size());
    }
    
    spdlog::info("Computing cosine similarities using ONNX model...");
    
    // 使用ONNX余弦相似度模型计算相似度
    neural_network::CosineSimilarityModel cosine_model{"cosine_sim.onnx", neural_network::Device::CPU};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto similarity_scores = cosine_model.inference(target_embedding, batch_embeddings);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    spdlog::info("ONNX cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results:");
    
    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size(); ++i) {
        spdlog::info("  \"{}\" <-> \"{}\": {:.6f}", target_text, batch_text[i], similarity_scores[i]);
    }
    
    // 验证结果
    EXPECT_EQ(similarity_scores.size(), batch_text.size());
    
    // 验证相似度范围在[-1, 1]之间, 容许误差
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f - 1e-6);
        EXPECT_LE(score, 1.0f + 1e-6);
    }
    
    // 验证"杀猪"与自身的相似度最高
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");
    
    spdlog::info("ONNX cosine similarity test completed successfully!");
}
#endif

#ifdef __USE_ONNX_RUNTIME__
TEST(UnitTest, TestBatchTextEmbeddingONNX) {
    neural_network::init_onnx_runtime();
    neural_network::init_model_set(neural_network::Device::CPU);
    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.onnx");
    const auto tokenizer = neural_network::load_tokenizers("exported_model/tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings for similarity test...");
    
    // 计算目标文本的嵌入向量
    auto target_embedding = embedder.embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> batch_embeddings = embedder.embed(batch_text);
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed).count();
    
    spdlog::info("Batch embedding computation took {} ms", embed_duration);
    spdlog::info("Generated {} embeddings with dimension {}", batch_embeddings.size(), 
                 batch_embeddings.empty() ? 0 : batch_embeddings[0].size());
    
    for (size_t i = 0; i < batch_text.size() && i < batch_embeddings.size(); ++i) {
        spdlog::info("Text: \"{}\" - Embedding dim: {}", batch_text[i], batch_embeddings[i].size());
    }
    
    spdlog::info("Computing cosine similarities using ONNX model...");
    
    // 使用ONNX余弦相似度模型计算相似度
    neural_network::CosineSimilarityModel cosine_model{"cosine_sim.onnx", neural_network::Device::CPU};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto similarity_scores = cosine_model.inference(target_embedding, batch_embeddings);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    spdlog::info("ONNX cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results:");
    
    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size(); ++i) {
        spdlog::info("  \"{}\" <-> \"{}\": {:.6f}", target_text, batch_text[i], similarity_scores[i]);
    }
    
    // 验证结果
    EXPECT_EQ(similarity_scores.size(), batch_text.size());
    
    // 验证相似度范围在[-1, 1]之间, 容许误差
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f - 1e-6);
        EXPECT_LE(score, 1.0f + 1e-6);
    }
    
    // 验证"杀猪"与自身的相似度最高
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");
    
    spdlog::info("ONNX cosine similarity test completed successfully!");
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

TEST(UnitTest, TestCosineSimilarityMPSEmbeddingWithCPUSimilarity) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();
    
    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings using MPS for CPU similarity verification...");
    
    // 使用MPS计算目标文本的嵌入向量
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 使用MPS批量嵌入函数计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    auto batch_embeddings = model_set.text_embedding_model->embed(batch_text);
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed).count();
    
    spdlog::info("MPS batch embedding computation took {} ms", embed_duration);
    spdlog::info("Generated {} embeddings with dimension {}", batch_embeddings.size(), 
                 batch_embeddings.empty() ? 0 : batch_embeddings[0].size());
    
    for (size_t i = 0; i < batch_text.size() && i < batch_embeddings.size(); ++i) {
        spdlog::info("Text: \"{}\" - Embedding dim: {}", batch_text[i], batch_embeddings[i].size());
    }
    
    spdlog::info("Computing cosine similarities using CPU implementation...");
    
    // 使用CPU版本的余弦相似度函数计算相似度
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> similarity_scores;
    similarity_scores.reserve(batch_embeddings.size());
    
    for (const auto &embedding : batch_embeddings) {
        if (target_embedding.size() != embedding.size()) {
            spdlog::error("Embedding dimension mismatch: target {} vs candidate {}", 
                         target_embedding.size(), embedding.size());
            continue;
        }
        float similarity = neural_network::cpu::cosine_similarity(
            target_embedding.data(), embedding.data(), target_embedding.size());
        similarity_scores.push_back(similarity);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    spdlog::info("CPU cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results (MPS embedding + CPU similarity):");
    
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
    
    spdlog::info("MPS embedding + CPU similarity test completed successfully!");
    
    // 额外验证：与LibTorch模型结果对比（如果存在）
    if (model_set.cosine_similarity_model) {
        spdlog::info("Comparing with LibTorch cosine similarity model results...");
        auto libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
        
        for (size_t i = 0; i < std::min(similarity_scores.size(), libtorch_scores.size()); ++i) {
            float diff = std::abs(similarity_scores[i] - libtorch_scores[i]);
            spdlog::info("  Text[{}]: CPU={:.6f}, LibTorch={:.6f}, Diff={:.6f}", 
                        i, similarity_scores[i], libtorch_scores[i], diff);
            
            // 允许一定的数值误差（约0.001）
            EXPECT_LT(diff, 0.001f) << "Significant difference between CPU and LibTorch results for text " << i;
        }
    }
}

TEST(UnitTest, TestCosineSimilarityMPSIndividualEmbeddingWithCPUSimilarity) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();
    
    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings individually using MPS for CPU similarity verification...");
    
    // 使用MPS逐个计算目标文本的嵌入向量
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 逐个计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> batch_embeddings;
    batch_embeddings.reserve(batch_text.size());
    
    for (size_t i = 0; i < batch_text.size(); ++i) {
        auto embedding = model_set.text_embedding_model->embed(batch_text[i]);
        batch_embeddings.emplace_back(std::move(embedding));
        spdlog::info("Individual embedding[{}] for \"{}\": dim={}", i, batch_text[i], batch_embeddings[i].size());
    }
    
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed).count();
    
    spdlog::info("Individual MPS embedding computation took {} ms", embed_duration);
    spdlog::info("Generated {} embeddings with dimension {}", batch_embeddings.size(), 
                 batch_embeddings.empty() ? 0 : batch_embeddings[0].size());
    
    spdlog::info("Computing cosine similarities using CPU implementation...");
    
    // 使用CPU版本的余弦相似度函数计算相似度
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> similarity_scores;
    similarity_scores.reserve(batch_embeddings.size());
    
    for (size_t i = 0; i < batch_embeddings.size(); ++i) {
        const auto &embedding = batch_embeddings[i];
        if (target_embedding.size() != embedding.size()) {
            spdlog::error("Embedding dimension mismatch: target {} vs candidate[{}] {}", 
                         target_embedding.size(), i, embedding.size());
            continue;
        }
        float similarity = neural_network::cpu::cosine_similarity(
            target_embedding.data(), embedding.data(), target_embedding.size());
        similarity_scores.push_back(similarity);
        spdlog::info("Individual similarity[{}]: {:.6f}", i, similarity);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    spdlog::info("CPU cosine similarity computation took {} ms", duration);
    spdlog::info("Final similarity results (Individual MPS embedding + CPU similarity):");
    
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
    
    spdlog::info("Individual MPS embedding + CPU similarity test completed successfully!");
    
    // 主要对比：与批量embedding结果的一致性验证（仅使用CPU余弦相似度计算）
    spdlog::info("Comparing individual vs batch embedding results (CPU cosine similarity only)...");
    auto batch_embeddings_for_comparison = model_set.text_embedding_model->embed(batch_text);
    
    // 计算批量embedding + CPU相似度作为参考
    std::vector<float> batch_cpu_similarity_scores;
    batch_cpu_similarity_scores.reserve(batch_embeddings_for_comparison.size());
    
    for (const auto &embedding : batch_embeddings_for_comparison) {
        if (target_embedding.size() == embedding.size()) {
            float similarity = neural_network::cpu::cosine_similarity(
                target_embedding.data(), embedding.data(), target_embedding.size());
            batch_cpu_similarity_scores.push_back(similarity);
        }
    }
    
    spdlog::info("=== Text Embedding Model Inference Comparison (CPU Cosine Similarity Only) ===");
    
    // 主要对比：Individual vs Batch embedding similarity results
    spdlog::info("Individual vs Batch Text Embedding Model Inference Results:");
    for (size_t i = 0; i < std::min(similarity_scores.size(), batch_cpu_similarity_scores.size()); ++i) {
        float diff = std::abs(similarity_scores[i] - batch_cpu_similarity_scores[i]);
        spdlog::info("  Text[{}] \"{}\": Individual_Embed+CPU_Sim={:.6f}, Batch_Embed+CPU_Sim={:.6f}, Diff={:.6f}", 
                    i, batch_text[i], similarity_scores[i], batch_cpu_similarity_scores[i], diff);
        
        // 验证个别推理和批量推理的embedding结果应该一致（允许很小的数值误差）
        EXPECT_LT(diff, 0.0001f) << "Individual and batch text embedding inference results differ significantly for text " << i;
    }
    
    // 总结比较结果
    spdlog::info("Summary: Text Embedding Model Individual vs Batch Inference Comparison:");
    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size() && 
         i < batch_cpu_similarity_scores.size(); ++i) {
        spdlog::info("  Text[{}] \"{}\": ", i, batch_text[i]);
        spdlog::info("    Individual_Embedding+CPU_Similarity: {:.6f}", similarity_scores[i]);
        spdlog::info("    Batch_Embedding+CPU_Similarity:      {:.6f}", batch_cpu_similarity_scores[i]);
    }
}

TEST(UnitTest, TestLibTorchTextEmbeddingVectorDifference) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();
    
    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    
    spdlog::info("=== LibTorch Text Embedding Vector Difference Analysis ===");
    
    // 计算单个推理的嵌入向量
    spdlog::info("Computing individual embeddings...");
    auto start_individual = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> individual_embeddings;
    individual_embeddings.reserve(batch_text.size());
    
    for (size_t i = 0; i < batch_text.size(); ++i) {
        auto embedding = model_set.text_embedding_model->embed(batch_text[i]);
        individual_embeddings.emplace_back(std::move(embedding));
        spdlog::info("Individual[{}] \"{}\" - embedding dim: {}", i, batch_text[i], individual_embeddings[i].size());
    }
    auto end_individual = std::chrono::high_resolution_clock::now();
    auto individual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_individual - start_individual).count();
    
    // 计算批量推理的嵌入向量
    spdlog::info("Computing batch embeddings...");
    auto start_batch = std::chrono::high_resolution_clock::now();
    auto batch_embeddings = model_set.text_embedding_model->embed(batch_text);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    
    spdlog::info("Timing comparison:");
    spdlog::info("  Individual inference: {} ms", individual_duration);
    spdlog::info("  Batch inference: {} ms", batch_duration);
    spdlog::info("  Speed ratio: {:.2f}x", static_cast<float>(individual_duration) / batch_duration);
    
    // 验证向量维度一致性
    ASSERT_EQ(individual_embeddings.size(), batch_embeddings.size());
    
    spdlog::info("=== Vector Difference Analysis ===");
    
    for (size_t i = 0; i < batch_text.size(); ++i) {
        const auto &individual_vec = individual_embeddings[i];
        const auto &batch_vec = batch_embeddings[i];
        
        ASSERT_EQ(individual_vec.size(), batch_vec.size()) 
            << "Embedding dimension mismatch for text " << i;
        
        // 计算向量差异统计
        std::vector<float> differences;
        differences.reserve(individual_vec.size());
        float sum_abs_diff = 0.0f;
        float max_diff = 0.0f;
        float sum_squared_diff = 0.0f;
        
        for (size_t j = 0; j < individual_vec.size(); ++j) {
            float diff = individual_vec[j] - batch_vec[j];
            differences.push_back(diff);
            float abs_diff = std::abs(diff);
            sum_abs_diff += abs_diff;
            max_diff = std::max(max_diff, abs_diff);
            sum_squared_diff += diff * diff;
        }
        
        float mean_abs_diff = sum_abs_diff / individual_vec.size();
        float rmse = std::sqrt(sum_squared_diff / individual_vec.size());
        
        spdlog::info("Text[{}] \"{}\": ", i, batch_text[i]);
        spdlog::info("  Vector dimension: {}", individual_vec.size());
        spdlog::info("  Mean absolute difference: {:.8f}", mean_abs_diff);
        spdlog::info("  Maximum absolute difference: {:.8f}", max_diff);
        spdlog::info("  Root Mean Square Error (RMSE): {:.8f}", rmse);
        
        // 输出前20个元素的详细对比
        spdlog::info("  First 20 elements comparison:");
        size_t first_elements_to_show = std::min(static_cast<size_t>(20), individual_vec.size());
        for (size_t j = 0; j < first_elements_to_show; ++j) {
            spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", 
                        j, individual_vec[j], batch_vec[j], differences[j]);
        }
        
        // 输出后20个元素的详细对比
        if (individual_vec.size() > 20) {
            spdlog::info("  Last 20 elements comparison:");
            size_t start_idx = individual_vec.size() - 20;
            for (size_t j = start_idx; j < individual_vec.size(); ++j) {
                spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", 
                            j, individual_vec[j], batch_vec[j], differences[j]);
            }
        }
        
        // 验证差异在可接受范围内（通常embedding应该是相同的）
        EXPECT_LT(mean_abs_diff, 1e-6f) 
            << "Mean absolute difference too large for text " << i;
        EXPECT_LT(max_diff, 1e-5f) 
            << "Maximum difference too large for text " << i;
    }
    
    spdlog::info("=== Summary ===");
    spdlog::info("LibTorch text embedding individual vs batch inference comparison completed.");
    spdlog::info("All vectors should be nearly identical with minimal numerical differences.");
    spdlog::info("Performance: Batch inference is {:.2f}x faster than individual inference.", 
                 static_cast<float>(individual_duration) / batch_duration);
}

TEST(UnitTest, TestLibTorchTokenEmbeddingNoMeanPoolingVectorDifference) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();
    
    // 创建 TextEmbeddingModel 实例（不带 mean pooling）
    neural_network::TextEmbeddingModel token_embedding_model("exported_model/text_embedding.pt", neural_network::Device::CPU);
    
    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    
    spdlog::info("=== LibTorch Token Embedding Vector Difference Analysis ===");
    
    // 使用 TextEmbeddingModel 计算单个推理的 token 嵌入向量
    spdlog::info("Computing individual token embeddings...");
    auto start_individual = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<float>>> individual_token_embeddings;
    individual_token_embeddings.reserve(batch_text.size());
    
    for (size_t i = 0; i < batch_text.size(); ++i) {
        // 使用 tokenizer 获取 token_ids 和 attention_mask
        auto token_ids = model_set.tokenizer_wrapper.encode(batch_text[i]);
        neural_network::attention_mask_list_t attention_mask(token_ids.size(), 1);
        
        // 使用非 mean pooling 的模型获取 token 级别嵌入
        auto token_embedding_matrix = token_embedding_model.embed(token_ids, attention_mask);
        individual_token_embeddings.emplace_back(std::move(token_embedding_matrix));
        
        spdlog::info("Individual[{}] \"{}\" - token count: {}, embedding dim per token: {}", 
                    i, batch_text[i], 
                    individual_token_embeddings[i].size(),
                    individual_token_embeddings[i].empty() ? 0 : individual_token_embeddings[i][0].size());
    }
    auto end_individual = std::chrono::high_resolution_clock::now();
    auto individual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_individual - start_individual).count();
    
    // 计算批量推理的 token 嵌入向量
    spdlog::info("Computing batch token embeddings...");
    auto start_batch = std::chrono::high_resolution_clock::now();
    
    // 准备批量输入
    std::vector<neural_network::token_id_list_t> batch_token_ids;
    std::vector<neural_network::attention_mask_list_t> batch_attention_masks;
    
    for (const auto &text : batch_text) {
        auto token_ids = model_set.tokenizer_wrapper.encode(text);
        neural_network::attention_mask_list_t attention_mask(token_ids.size(), 1);
        batch_token_ids.emplace_back(std::move(token_ids));
        batch_attention_masks.emplace_back(std::move(attention_mask));
    }
    
    auto batch_token_embeddings = token_embedding_model.embed(batch_token_ids, batch_attention_masks);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    
    spdlog::info("Timing comparison:");
    spdlog::info("  Individual inference: {} ms", individual_duration);
    spdlog::info("  Batch inference: {} ms", batch_duration);
    spdlog::info("  Speed ratio: {:.2f}x", static_cast<float>(individual_duration) / batch_duration);
    
    // 验证向量维度一致性
    ASSERT_EQ(individual_token_embeddings.size(), batch_token_embeddings.size());
    
    spdlog::info("=== Token Embedding Vector Difference Analysis ===");
    
    for (size_t i = 0; i < batch_text.size(); ++i) {
        const auto &individual_token_matrix = individual_token_embeddings[i];
        const auto &batch_token_matrix = batch_token_embeddings[i];
        
        ASSERT_EQ(individual_token_matrix.size(), batch_token_matrix.size()) 
            << "Token count mismatch for text " << i;
        
        spdlog::info("Text[{}] \"{}\": ", i, batch_text[i]);
        spdlog::info("  Token count: {}", individual_token_matrix.size());
        
        if (individual_token_matrix.empty()) {
            spdlog::info("  No tokens to compare");
            continue;
        }
        
        // 统计所有 token 的差异
        float total_sum_abs_diff = 0.0f;
        float total_max_diff = 0.0f;
        float total_sum_squared_diff = 0.0f;
        size_t total_elements = 0;
        
        for (size_t token_idx = 0; token_idx < individual_token_matrix.size(); ++token_idx) {
            const auto &individual_token_vec = individual_token_matrix[token_idx];
            const auto &batch_token_vec = batch_token_matrix[token_idx];
            
            ASSERT_EQ(individual_token_vec.size(), batch_token_vec.size()) 
                << "Token embedding dimension mismatch for text " << i << " token " << token_idx;
            
            // 计算当前 token 的向量差异
            float token_sum_abs_diff = 0.0f;
            float token_max_diff = 0.0f;
            float token_sum_squared_diff = 0.0f;
            
            for (size_t j = 0; j < individual_token_vec.size(); ++j) {
                float diff = individual_token_vec[j] - batch_token_vec[j];
                float abs_diff = std::abs(diff);
                token_sum_abs_diff += abs_diff;
                token_max_diff = std::max(token_max_diff, abs_diff);
                token_sum_squared_diff += diff * diff;
                
                total_sum_abs_diff += abs_diff;
                total_max_diff = std::max(total_max_diff, abs_diff);
                total_sum_squared_diff += diff * diff;
                total_elements++;
            }
            
            float token_mean_abs_diff = token_sum_abs_diff / individual_token_vec.size();
            float token_rmse = std::sqrt(token_sum_squared_diff / individual_token_vec.size());
            
            spdlog::info("    Token[{}]: dim={}, mean_abs_diff={:.8f}, max_diff={:.8f}, rmse={:.8f}", 
                        token_idx, individual_token_vec.size(), token_mean_abs_diff, token_max_diff, token_rmse);
            
            // 验证每个 token 的差异在可接受范围内
            EXPECT_LT(token_mean_abs_diff, 1e-6f) 
                << "Token embedding mean absolute difference too large for text " << i << " token " << token_idx;
            EXPECT_LT(token_max_diff, 1e-5f) 
                << "Token embedding maximum difference too large for text " << i << " token " << token_idx;
        }
        
        // 计算整个文本的总体差异统计
        if (total_elements > 0) {
            float overall_mean_abs_diff = total_sum_abs_diff / total_elements;
            float overall_rmse = std::sqrt(total_sum_squared_diff / total_elements);
            
            spdlog::info("  Overall statistics:");
            spdlog::info("    Total elements: {}", total_elements);
            spdlog::info("    Overall mean absolute difference: {:.8f}", overall_mean_abs_diff);
            spdlog::info("    Overall maximum absolute difference: {:.8f}", total_max_diff);
            spdlog::info("    Overall Root Mean Square Error (RMSE): {:.8f}", overall_rmse);
            
            // 验证整体差异在可接受范围内
            EXPECT_LT(overall_mean_abs_diff, 1e-6f) 
                << "Overall token embedding mean absolute difference too large for text " << i;
            EXPECT_LT(total_max_diff, 1e-5f) 
                << "Overall token embedding maximum difference too large for text " << i;
        }
        
        // 输出第一个和最后一个 token 的详细对比（前10个元素）
        if (!individual_token_matrix.empty()) {
            spdlog::info("  First token detailed comparison (first 10 elements):");
            const auto &first_individual = individual_token_matrix[0];
            const auto &first_batch = batch_token_matrix[0];
            size_t elements_to_show = std::min(static_cast<size_t>(10), first_individual.size());
            for (size_t j = 0; j < elements_to_show; ++j) {
                float diff = first_individual[j] - first_batch[j];
                spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", 
                            j, first_individual[j], first_batch[j], diff);
            }
            
            if (individual_token_matrix.size() > 1) {
                spdlog::info("  Last token detailed comparison (first 10 elements):");
                const auto &last_individual = individual_token_matrix.back();
                const auto &last_batch = batch_token_matrix.back();
                elements_to_show = std::min(static_cast<size_t>(10), last_individual.size());
                for (size_t j = 0; j < elements_to_show; ++j) {
                    float diff = last_individual[j] - last_batch[j];
                    spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", 
                                j, last_individual[j], last_batch[j], diff);
                }
            }
        }
    }
    
    spdlog::info("=== Summary ===");
    spdlog::info("LibTorch token embedding individual vs batch inference comparison completed.");
    spdlog::info("All token embeddings should be nearly identical with minimal numerical differences.");
    spdlog::info("Performance: Batch inference is {:.2f}x faster than individual inference.", 
                 static_cast<float>(individual_duration) / batch_duration);
}



#endif