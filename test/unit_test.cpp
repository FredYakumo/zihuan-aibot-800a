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
TEST(UnitTest, TestCosineSimilarityONNX) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("embedding.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";
    
    spdlog::info("Computing embeddings for similarity test...");
    
    // 计算目标文本的嵌入向量
    neural_network::token_id_vec_with_mask_t target_token = tokenizer_wrapper.encode_with_mask(target_text);
    auto target_embedding = embedder.embed(target_token.first, target_token.second);
    spdlog::info("Target embedding dim: {}", target_embedding.size());
    
    // 计算候选文本的嵌入向量
    auto start_embed = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> batch_embeddings;
    for (const auto &text : batch_text) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        auto embedding = embedder.embed(token_mask.first, token_mask.second);
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
    
    // 验证相似度范围在[-1, 1]之间
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f);
        EXPECT_LE(score, 1.0f);
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
    
    // 额外验证：与LibTorch模型结果对比（如果存在）
    if (model_set.cosine_similarity_model) {
        spdlog::info("Comparing individual embedding results with LibTorch cosine similarity model...");
        auto libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
        
        spdlog::info("Individual vs LibTorch cosine similarity comparison:");
        for (size_t i = 0; i < std::min(similarity_scores.size(), libtorch_scores.size()); ++i) {
            float diff = std::abs(similarity_scores[i] - libtorch_scores[i]);
            spdlog::info("  Text[{}]: Individual+CPU={:.6f}, LibTorch={:.6f}, Diff={:.6f}", 
                        i, similarity_scores[i], libtorch_scores[i], diff);
            
            // 允许一定的数值误差（约0.001）
            EXPECT_LT(diff, 0.001f) << "Significant difference between individual CPU and LibTorch results for text " << i;
        }
    }
    
    // 额外对比：与批量embedding结果的一致性验证
    spdlog::info("Comparing individual vs batch embedding results...");
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
    
    spdlog::info("=== Detailed Comparison Results ===");
    
    // 1. Individual vs Batch embedding similarity results
    spdlog::info("1. Individual vs Batch embedding similarity results:");
    for (size_t i = 0; i < std::min(similarity_scores.size(), batch_cpu_similarity_scores.size()); ++i) {
        float diff = std::abs(similarity_scores[i] - batch_cpu_similarity_scores[i]);
        spdlog::info("  Text[{}] \"{}\": Individual+CPU={:.6f}, Batch+CPU={:.6f}, Diff={:.6f}", 
                    i, batch_text[i], similarity_scores[i], batch_cpu_similarity_scores[i], diff);
        
        // 验证个别计算和批量计算的相似度结果应该一致（允许很小的数值误差）
        EXPECT_LT(diff, 0.0001f) << "Individual and batch embedding similarity results differ significantly for text " << i;
    }
    
    // 2. Individual embedding vs LibTorch cosine similarity (if available)
    if (model_set.cosine_similarity_model) {
        auto individual_libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
        spdlog::info("2. Individual embedding vs LibTorch cosine similarity:");
        for (size_t i = 0; i < std::min(similarity_scores.size(), individual_libtorch_scores.size()); ++i) {
            float diff = std::abs(similarity_scores[i] - individual_libtorch_scores[i]);
            spdlog::info("  Text[{}] \"{}\": Individual+CPU={:.6f}, Individual+LibTorch={:.6f}, Diff={:.6f}", 
                        i, batch_text[i], similarity_scores[i], individual_libtorch_scores[i], diff);
        }
    }
    
    // 3. Batch embedding vs LibTorch cosine similarity (if available)
    if (model_set.cosine_similarity_model) {
        auto batch_libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings_for_comparison);
        spdlog::info("3. Batch embedding vs LibTorch cosine similarity:");
        for (size_t i = 0; i < std::min(batch_cpu_similarity_scores.size(), batch_libtorch_scores.size()); ++i) {
            float diff = std::abs(batch_cpu_similarity_scores[i] - batch_libtorch_scores[i]);
            spdlog::info("  Text[{}] \"{}\": Batch+CPU={:.6f}, Batch+LibTorch={:.6f}, Diff={:.6f}", 
                        i, batch_text[i], batch_cpu_similarity_scores[i], batch_libtorch_scores[i], diff);
        }
    }
    
    // 4. Summary of all methods
    spdlog::info("4. Summary comparison of all methods:");
    if (model_set.cosine_similarity_model) {
        auto individual_libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
        auto batch_libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings_for_comparison);
        
        for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size() && 
             i < batch_cpu_similarity_scores.size() && i < individual_libtorch_scores.size() && 
             i < batch_libtorch_scores.size(); ++i) {
            spdlog::info("  Text[{}] \"{}\": ", i, batch_text[i]);
            spdlog::info("    Individual+CPU:     {:.6f}", similarity_scores[i]);
            spdlog::info("    Batch+CPU:          {:.6f}", batch_cpu_similarity_scores[i]);
            spdlog::info("    Individual+LibTorch: {:.6f}", individual_libtorch_scores[i]);
            spdlog::info("    Batch+LibTorch:      {:.6f}", batch_libtorch_scores[i]);
        }
    } else {
        for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size() && 
             i < batch_cpu_similarity_scores.size(); ++i) {
            spdlog::info("  Text[{}] \"{}\": ", i, batch_text[i]);
            spdlog::info("    Individual+CPU: {:.6f}", similarity_scores[i]);
            spdlog::info("    Batch+CPU:      {:.6f}", batch_cpu_similarity_scores[i]);
        }
    }
}

#endif