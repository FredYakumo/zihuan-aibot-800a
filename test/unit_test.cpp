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
TEST(UnitTest, TestCosineSimilarityLibTorch) {
    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.pt");
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

    spdlog::info("正在拷贝额外的向量 (LibTorch)");
    const auto emb = batch_embedding[0];
    for (size_t i = 0; i < 400000; ++i) {
        batch_embedding.push_back(emb);
    }

    spdlog::info("开始进行推理 (LibTorch)");

    neural_network::token_id_vec_with_mask_t target_token = tokenizer_wrapper.encode_with_mask("杀猪盘");
    auto target_embedding = embedder.embed(target_token.first, target_token.second);

    neural_network::CosineSimilarityModel cpu_model{"exported_model/cosine_sim.bin", neural_network::Device::CPU};

    auto start = std::chrono::high_resolution_clock::now();
    auto res = cpu_model.inference(target_embedding, batch_embedding);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // spdlog::info("Similarity (LibTorch):");
    // for (const auto r : res) {
    //     spdlog::info(r);
    // }

    spdlog::info("Inference time (LibTorch CPU): {} ms", cpu_duration);

    // Test GPU if available
    try {
        neural_network::CosineSimilarityModel cuda_model{"exported_model/cosine_sim.bin", neural_network::Device::CUDA};
        
        start = std::chrono::high_resolution_clock::now();
        auto res_cuda = cuda_model.inference(target_embedding, batch_embedding);
        end = std::chrono::high_resolution_clock::now();
        auto cuda_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        spdlog::info("Inference time (LibTorch CUDA): {} ms", cuda_duration);
        
        // Verify results are similar
        EXPECT_EQ(res.size(), res_cuda.size());
        for (size_t i = 0; i < std::min(res.size(), static_cast<size_t>(10)); ++i) {
            EXPECT_NEAR(res[i], res_cuda[i], 0.01f) << "Mismatch at index " << i;
        }
    } catch (const std::exception &e) {
        spdlog::warn("CUDA test skipped: {}", e.what());
    }

    // Test MPS if available (for macOS)
    try {
        neural_network::CosineSimilarityModel mps_model{"exported_model/cosine_sim.bin", neural_network::Device::MPS};
        
        start = std::chrono::high_resolution_clock::now();
        auto res_mps = mps_model.inference(target_embedding, batch_embedding);
        end = std::chrono::high_resolution_clock::now();
        auto mps_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        spdlog::info("Inference time (LibTorch MPS): {} ms", mps_duration);
        
        // Verify results are similar
        EXPECT_EQ(res.size(), res_mps.size());
        for (size_t i = 0; i < std::min(res.size(), static_cast<size_t>(10)); ++i) {
            EXPECT_NEAR(res[i], res_mps[i], 0.01f) << "Mismatch at index " << i;
        }
    } catch (const std::exception &e) {
        spdlog::warn("MPS test skipped: {}", e.what());
    }
}

TEST(UnitTest, TestTextEmbeddingLibTorch) {
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

TEST(UnitTest, TestTextEmbeddingWithMeanPoolingLibTorch) {
    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.pt");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> test_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    
    // Test single text embedding with mean pooling
    for (const auto &text : test_texts) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        auto sentence_embedding = embedder.embed(token_mask.first, token_mask.second);
        
        EXPECT_GT(sentence_embedding.size(), 0) << "Sentence embedding should not be empty for text: " << text;
        spdlog::info("Text: '{}', Sentence embedding dim: {}", text, sentence_embedding.size());
    }

    // Test batch text embedding with mean pooling
    std::vector<neural_network::token_id_list_t> batch_token_ids;
    std::vector<neural_network::attention_mask_list_t> batch_attention_masks;
    
    for (const auto &text : test_texts) {
        auto token_mask = tokenizer_wrapper.encode_with_mask(text);
        batch_token_ids.push_back(token_mask.first);
        batch_attention_masks.push_back(token_mask.second);
    }
    
    auto batch_sentence_embeddings = embedder.embed(batch_token_ids, batch_attention_masks);
    EXPECT_EQ(batch_sentence_embeddings.size(), test_texts.size()) << "Batch sentence embedding count should match input count";
    
    for (size_t i = 0; i < batch_sentence_embeddings.size(); ++i) {
        EXPECT_GT(batch_sentence_embeddings[i].size(), 0) << "Batch sentence embedding " << i << " should not be empty";
    }
    
    spdlog::info("Batch sentence embeddings test passed for {} texts", test_texts.size());
    
    // Test string-based API
    auto simple_embedding = embedder.embed("测试文本");
    EXPECT_GT(simple_embedding.size(), 0) << "Simple text embedding should not be empty";
    
    auto batch_simple_embeddings = embedder.embed(test_texts);
    EXPECT_EQ(batch_simple_embeddings.size(), test_texts.size()) << "Simple batch embedding count should match";
    
    spdlog::info("String-based API test passed");
}

TEST(UnitTest, TestModelSetLibTorch) {
    spdlog::info("Testing ModelSet with LibTorch backend");
    
    try {
        neural_network::init_model_set(neural_network::Device::CPU);
        auto &model_set = neural_network::get_model_set();
        
        // Test text embedding model
        EXPECT_NE(model_set.text_embedding_model, nullptr) << "Text embedding model should be initialized";
        
        // Test cosine similarity model
        EXPECT_NE(model_set.cosine_similarity_model, nullptr) << "Cosine similarity model should be initialized";
        
        // Test tokenizer
        EXPECT_NE(model_set.tokenizer, nullptr) << "Tokenizer should be initialized";
        
        // Test a complete workflow
        const std::string test_text = "这是一个测试文本";
        auto target_embedding = model_set.text_embedding_model->embed(test_text);
        EXPECT_GT(target_embedding.size(), 0) << "Target embedding should not be empty";
        
        // Create some reference embeddings
        std::vector<std::string> reference_texts = {
            "这是另一个测试文本",
            "完全不同的内容",
            "这是一个测试文本"  // Same as target
        };
        
        neural_network::emb_mat_t reference_embeddings;
        for (const auto &text : reference_texts) {
            auto emb = model_set.text_embedding_model->embed(text);
            reference_embeddings.push_back(emb);
        }
        
        // Test cosine similarity
        auto similarities = model_set.cosine_similarity_model->inference(target_embedding, reference_embeddings);
        EXPECT_EQ(similarities.size(), reference_texts.size()) << "Similarity count should match reference count";
        
        // The similarity with the same text should be highest
        auto max_sim_idx = std::distance(similarities.begin(), 
                                       std::max_element(similarities.begin(), similarities.end()));
        EXPECT_EQ(max_sim_idx, 2) << "Highest similarity should be with the same text";
        
        spdlog::info("ModelSet integration test passed");
        spdlog::info("Similarities: [{}, {}, {}]", similarities[0], similarities[1], similarities[2]);
        
    } catch (const std::exception &e) {
        FAIL() << "ModelSet test failed with exception: " << e.what();
    }
}
#endif