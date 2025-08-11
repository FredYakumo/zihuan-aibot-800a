#include "config.h"
#include "neural_network/model_set.h"
#include "neural_network/text_model/text_embedding_model.h"
#include "neural_network/text_model/text_embedding_with_mean_pooling_model.h"
#include "neural_network/text_model/tokenizer_wrapper.h"
#include "utils.h"
#include <chrono>
#include <fstream>
#include <general-wheel-cpp/string_utils.hpp>
#include <gtest/gtest.h>
#include <iterator>
#include <neural_network/nn.h>
#include <string>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.
#include <fstream>
// for std::remove
#include <cstdio>

// LAC components (headers are no-ops unless __USE_PADDLE_INFERENCE__ is defined)
#include "neural_network/text_model/lac/ahocorasick.h"
#include "neural_network/text_model/lac/lac_custom.h"
#include "neural_network/text_model/lac/lac_util.h"

using namespace wheel;

TEST(BasicSetup, FrameworkInitialization) { GTEST_LOG_(INFO) << "Basic test framework initialization"; }

#ifdef __USE_ONNX_RUNTIME__
TEST(ONNXRuntime, ProviderAvailabilityCheck) {
    spdlog::info(Ort::GetVersionString());
    for (const auto &provider : Ort::GetAvailableProviders()) {
        spdlog::info("Available provider: {}", provider);
    }
}
#endif

TEST(BotAdapter, WebSocketConnectionSetup) {
    spdlog::set_level(spdlog::level::debug);
    // bot_adapter::BotAdapter adapter{"ws://localhost:13378/all"};
    // const auto test_sender_id = 3507578481;
    // adapter.register_event<bot_adapter::GroupMessageEvent>([&adapter](const bot_adapter::GroupMessageEvent &e) {
    //     spdlog::info("Message received from sender: {}, group: {}", e.sender.id, e.group.name);
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

TEST(StringUtils, KeywordReplacementOperations) {
    std::string str0 = "#联网 abc";
    std::string new_str0 = replace_str(str0, "#联网", "");
    EXPECT_EQ(new_str0, " abc");

    std::string str = "abc #联网(123)";
    std::string keyword = "#联网";
    std::string new_str = replace_keyword_and_parentheses_content(str, keyword, "");
    EXPECT_EQ(new_str, "abc ");
}

constexpr std::string_view test_url = "ws://localhost:13378/all";

TEST(BotAdapter, MessageIdRetrieval) {
    // spdlog::set_level(spdlog::level::debug);
    // bot_adapter::BotAdapter adapter{test_url};
    // adapter.get_message_id(2259, 790544814,
    //                        [](const nlohmann::json &data) { spdlog::debug("message data: {}", data.dump()); });

    // adapter.start();
}

// TEST(BotAdapter, LongTextMessageSending) {
//     spdlog::set_level(spdlog::level::debug);
//     bot_adapter::BotAdapter adapter{test_url};
//     adapter.send_long_plain_text_replay(
//         bot_adapter::GroupSender(3507578481, "FredYakumo", std::nullopt, "", std::nullopt,
//                                  std::chrono::system_clock::now(),
//                                  bot_adapter::Group(790544814, "AIBot-800b grayscale test", "")),
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
TEST(ONNXRuntime, CosineSimilarityInference) {
    neural_network::init_onnx_runtime();

    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.onnx");
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";

    spdlog::info("Computing embeddings for similarity test...");

    // Calculate embedding vector for target text
    auto target_embedding = embedder.embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());

    // Calculate embedding vectors for candidate texts
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

    // Compute similarity using ONNX cosine similarity model
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

    // Validate results
    EXPECT_EQ(similarity_scores.size(), batch_text.size());

    // Validate similarity range within [-1, 1] (with small tolerance)
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f - 1e-6);
        EXPECT_LE(score, 1.0f + 1e-6);
    }

    // Validate that "杀猪" has the highest similarity with itself
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");

    spdlog::info("ONNX cosine similarity test completed successfully!");
}
#endif

#ifdef __USE_ONNX_RUNTIME__
TEST(ONNXRuntime, BatchTextEmbeddingInference) {
    neural_network::init_onnx_runtime();
    neural_network::init_model_set(neural_network::Device::CPU);
    neural_network::TextEmbeddingWithMeanPoolingModel embedder("exported_model/text_embedding_mean_pooling.onnx");
    const auto tokenizer = neural_network::load_tokenizers("exported_model/tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";

    spdlog::info("Computing embeddings for similarity test...");

    // Calculate embedding vector for target text
    auto target_embedding = embedder.embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());

    // Calculate embedding vectors for candidate texts
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

    // Compute similarity using ONNX cosine similarity model
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

    // Validate results
    EXPECT_EQ(similarity_scores.size(), batch_text.size());

    // Validate similarity range within [-1, 1] (with small tolerance)
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f - 1e-6);
        EXPECT_LE(score, 1.0f + 1e-6);
    }

    // Validate that "杀猪" has the highest similarity with itself
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");

    spdlog::info("ONNX cosine similarity test completed successfully!");
}

#endif

#ifdef __USE_LIBTORCH__

TEST(LibTorchMPS, BatchVsSingleInferenceSpeedComparison) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();

    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 5; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }

    spdlog::info("=== MPS Text Embedding Speed Comparison: Batch vs Single Inference ===");
    spdlog::info("Total test texts: {}", test_texts.size());

    spdlog::info("Testing single inference...");
    auto start_single = std::chrono::high_resolution_clock::now();
    for (const auto &text : test_texts) {
        model_set.text_embedding_model->embed(text);
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    spdlog::info("Single embedding total time: {} ms", duration_single);

    spdlog::info("Testing batch inference...");
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);

    // Compute speed ratio
    if (duration_batch > 0) {
        float speed_ratio = static_cast<float>(duration_single) / duration_batch;
        spdlog::info("=== Speed Comparison Results ===");
        spdlog::info("Batch inference is {:.2f}x faster than single inference", speed_ratio);
        spdlog::info("Performance improvement: {:.1f}%", (speed_ratio - 1.0f) * 100.0f);
    } else {
        spdlog::warn("Batch inference time too small to calculate meaningful ratio");
    }
}

TEST(LibTorchMPS, BatchVsSingleInferenceSpeedComparisonSize100) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();

    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 20; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }

    spdlog::info("=== MPS Text Embedding Speed Comparison: Batch vs Single Inference ===");
    spdlog::info("Total test texts: {}", test_texts.size());

    spdlog::info("Testing single inference...");
    auto start_single = std::chrono::high_resolution_clock::now();
    for (const auto &text : test_texts) {
        model_set.text_embedding_model->embed(text);
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    spdlog::info("Single embedding total time: {} ms", duration_single);

    spdlog::info("Testing batch inference...");
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);

    // Compute speed ratio
    if (duration_batch > 0) {
        float speed_ratio = static_cast<float>(duration_single) / duration_batch;
        spdlog::info("=== Speed Comparison Results ===");
        spdlog::info("Batch inference is {:.2f}x faster than single inference", speed_ratio);
        spdlog::info("Performance improvement: {:.1f}%", (speed_ratio - 1.0f) * 100.0f);
    } else {
        spdlog::warn("Batch inference time too small to calculate meaningful ratio");
    }
}

TEST(LibTorchPerformance, MPSVsCPUInferenceSpeedComparison) {
    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 10; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }

    spdlog::info("=== MPS vs CPU Text Embedding Speed Comparison ===");
    spdlog::info("Total test texts: {}", test_texts.size());

    // ===================
    // MPS Performance Test
    // ===================
    spdlog::info("=== Testing MPS Performance ===");
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &mps_model_set = neural_network::get_model_set();

    // MPS Single Inference
    spdlog::info("MPS - Testing single inference...");
    auto mps_start_single = std::chrono::high_resolution_clock::now();
    for (const auto &text : test_texts) {
        mps_model_set.text_embedding_model->embed(text);
    }
    auto mps_end_single = std::chrono::high_resolution_clock::now();
    auto mps_duration_single =
        std::chrono::duration_cast<std::chrono::milliseconds>(mps_end_single - mps_start_single).count();
    spdlog::info("MPS Single embedding total time: {} ms", mps_duration_single);

    // MPS Batch Inference
    spdlog::info("MPS - Testing batch inference...");
    auto mps_start_batch = std::chrono::high_resolution_clock::now();
    mps_model_set.text_embedding_model->embed(test_texts);
    auto mps_end_batch = std::chrono::high_resolution_clock::now();
    auto mps_duration_batch =
        std::chrono::duration_cast<std::chrono::milliseconds>(mps_end_batch - mps_start_batch).count();
    spdlog::info("MPS Batch embedding total time: {} ms", mps_duration_batch);

    // ===================
    // CPU Performance Test
    // ===================
    spdlog::info("=== Testing CPU Performance ===");
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &cpu_model_set = neural_network::get_model_set();

    // CPU Single Inference
    spdlog::info("CPU - Testing single inference...");
    auto cpu_start_single = std::chrono::high_resolution_clock::now();
    for (const auto &text : test_texts) {
        cpu_model_set.text_embedding_model->embed(text);
    }
    auto cpu_end_single = std::chrono::high_resolution_clock::now();
    auto cpu_duration_single =
        std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_single - cpu_start_single).count();
    spdlog::info("CPU Single embedding total time: {} ms", cpu_duration_single);

    // CPU Batch Inference
    spdlog::info("CPU - Testing batch inference...");
    auto cpu_start_batch = std::chrono::high_resolution_clock::now();
    cpu_model_set.text_embedding_model->embed(test_texts);
    auto cpu_end_batch = std::chrono::high_resolution_clock::now();
    auto cpu_duration_batch =
        std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_batch - cpu_start_batch).count();
    spdlog::info("CPU Batch embedding total time: {} ms", cpu_duration_batch);

    // ===================
    // Performance Analysis
    // ===================
    spdlog::info("=== Performance Comparison Results ===");

    // Single Inference Comparison
    if (cpu_duration_single > 0) {
        float single_speedup = static_cast<float>(cpu_duration_single) / mps_duration_single;
        spdlog::info("Single Inference:");
        spdlog::info("  MPS: {} ms", mps_duration_single);
        spdlog::info("  CPU: {} ms", cpu_duration_single);
        spdlog::info("  MPS is {:.2f}x {} than CPU", single_speedup > 1.0f ? single_speedup : (1.0f / single_speedup),
                     single_speedup > 1.0f ? "faster" : "slower");
    }

    // Batch Inference Comparison
    if (cpu_duration_batch > 0) {
        float batch_speedup = static_cast<float>(cpu_duration_batch) / mps_duration_batch;
        spdlog::info("Batch Inference:");
        spdlog::info("  MPS: {} ms", mps_duration_batch);
        spdlog::info("  CPU: {} ms", cpu_duration_batch);
        spdlog::info("  MPS is {:.2f}x {} than CPU", batch_speedup > 1.0f ? batch_speedup : (1.0f / batch_speedup),
                     batch_speedup > 1.0f ? "faster" : "slower");
    }

    // Batch vs Single Analysis for each device
    spdlog::info("=== Batch vs Single Analysis ===");

    if (mps_duration_batch > 0) {
        float mps_batch_ratio = static_cast<float>(mps_duration_single) / mps_duration_batch;
        spdlog::info("MPS - Batch is {:.2f}x faster than single inference", mps_batch_ratio);
    }

    if (cpu_duration_batch > 0) {
        float cpu_batch_ratio = static_cast<float>(cpu_duration_single) / cpu_duration_batch;
        spdlog::info("CPU - Batch is {:.2f}x faster than single inference", cpu_batch_ratio);
    }

    // Performance per text analysis
    spdlog::info("=== Performance Per Text ===");
    size_t text_count = test_texts.size();
    if (text_count > 0) {
        spdlog::info("Average time per text (single inference):");
        spdlog::info("  MPS: {:.2f} ms/text", static_cast<float>(mps_duration_single) / text_count);
        spdlog::info("  CPU: {:.2f} ms/text", static_cast<float>(cpu_duration_single) / text_count);

        spdlog::info("Average time per text (batch inference):");
        spdlog::info("  MPS: {:.2f} ms/text", static_cast<float>(mps_duration_batch) / text_count);
        spdlog::info("  CPU: {:.2f} ms/text", static_cast<float>(cpu_duration_batch) / text_count);
    }

    spdlog::info("MPS vs CPU performance comparison completed!");
}

TEST(LibTorchMPS, LargeBatchTextEmbedding) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();

    std::vector<std::string> test_texts;
    const std::vector<std::string> base_texts{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    for (int i = 0; i < 2000; ++i) {
        test_texts.insert(test_texts.end(), base_texts.begin(), base_texts.end());
    }
    auto start_batch = std::chrono::high_resolution_clock::now();
    model_set.text_embedding_model->embed(test_texts, 200);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto duration_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    spdlog::info("Batch embedding total time: {} ms", duration_batch);
}

TEST(LibTorchCPU, TextEmbeddingPerformanceComparison) {
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

TEST(LibTorchCPU, CosineSimilarityInference) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";

    spdlog::info("Computing embeddings for similarity test...");

    // Compute target text embedding
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());

    // Compute candidate embeddings using batch embedding
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

    // Compute similarity using LibTorch cosine similarity model
    auto start_time = std::chrono::high_resolution_clock::now();
    auto similarity_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    spdlog::info("LibTorch cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results:");

    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size(); ++i) {
        spdlog::info("  \"{}\" <-> \"{}\": {:.6f}", target_text, batch_text[i], similarity_scores[i]);
    }

    // Validate results
    EXPECT_EQ(similarity_scores.size(), batch_text.size());

    // Validate similarity range within [-1, 1]
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f);
        EXPECT_LE(score, 1.0f);
    }

    // Validate that "杀猪" has the highest similarity with itself
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");

    spdlog::info("LibTorch cosine similarity test completed successfully!");
}

TEST(LibTorchHybrid, MPSEmbeddingWithCPUSimilarity) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";

    spdlog::info("Computing embeddings using MPS for CPU similarity verification...");

    // Compute target text embedding using MPS
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());

    // Compute candidate embeddings using MPS batch embedding
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

    // Compute similarity using CPU cosine similarity function
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> similarity_scores;
    similarity_scores.reserve(batch_embeddings.size());

    for (const auto &embedding : batch_embeddings) {
        if (target_embedding.size() != embedding.size()) {
            spdlog::error("Embedding dimension mismatch: target {} vs candidate {}", target_embedding.size(),
                          embedding.size());
            continue;
        }
        float similarity =
            neural_network::cpu::cosine_similarity(target_embedding.data(), embedding.data(), target_embedding.size());
        similarity_scores.push_back(similarity);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    spdlog::info("CPU cosine similarity computation took {} ms", duration);
    spdlog::info("Similarity results (MPS embedding + CPU similarity):");

    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size(); ++i) {
        spdlog::info("  \"{}\" <-> \"{}\": {:.6f}", target_text, batch_text[i], similarity_scores[i]);
    }

    // Validate results
    EXPECT_EQ(similarity_scores.size(), batch_text.size());

    // Validate similarity range within [-1, 1]
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f);
        EXPECT_LE(score, 1.0f);
    }

    // Validate that "杀猪" has the highest similarity with itself
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");

    spdlog::info("MPS embedding + CPU similarity test completed successfully!");

    // Additional check: compare with LibTorch cosine similarity model (if available)
    if (model_set.cosine_similarity_model) {
        spdlog::info("Comparing with LibTorch cosine similarity model results...");
        auto libtorch_scores = model_set.cosine_similarity_model->inference(target_embedding, batch_embeddings);

        for (size_t i = 0; i < std::min(similarity_scores.size(), libtorch_scores.size()); ++i) {
            float diff = std::abs(similarity_scores[i] - libtorch_scores[i]);
            spdlog::info("  Text[{}]: CPU={:.6f}, LibTorch={:.6f}, Diff={:.6f}", i, similarity_scores[i],
                         libtorch_scores[i], diff);

            // Allow small numerical error (~0.001)
            EXPECT_LT(diff, 0.001f) << "Significant difference between CPU and LibTorch results for text " << i;
        }
    }
}

TEST(LibTorchHybrid, MPSIndividualEmbeddingWithCPUSimilarity) {
    neural_network::init_model_set(neural_network::Device::MPS);
    auto &model_set = neural_network::get_model_set();

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};
    const std::string target_text = "杀猪";

    spdlog::info("Computing embeddings individually using MPS for CPU similarity verification...");

    // Compute target text embedding individually using MPS
    auto target_embedding = model_set.text_embedding_model->embed(target_text);
    spdlog::info("Target embedding dim: {}", target_embedding.size());

    // Compute candidate embeddings individually
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

    // Compute similarity using CPU cosine similarity function
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<float> similarity_scores;
    similarity_scores.reserve(batch_embeddings.size());

    for (size_t i = 0; i < batch_embeddings.size(); ++i) {
        const auto &embedding = batch_embeddings[i];
        if (target_embedding.size() != embedding.size()) {
            spdlog::error("Embedding dimension mismatch: target {} vs candidate[{}] {}", target_embedding.size(), i,
                          embedding.size());
            continue;
        }
        float similarity =
            neural_network::cpu::cosine_similarity(target_embedding.data(), embedding.data(), target_embedding.size());
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

    // Validate results
    EXPECT_EQ(similarity_scores.size(), batch_text.size());

    // Validate similarity range within [-1, 1]
    for (const auto &score : similarity_scores) {
        EXPECT_GE(score, -1.0f);
        EXPECT_LE(score, 1.0f);
    }

    // Validate that "杀猪" has the highest similarity with itself
    auto max_score_it = std::max_element(similarity_scores.begin(), similarity_scores.end());
    size_t max_index = std::distance(similarity_scores.begin(), max_score_it);
    EXPECT_EQ(batch_text[max_index], "杀猪");

    spdlog::info("Individual MPS embedding + CPU similarity test completed successfully!");

    // Primary comparison: consistency with batch embedding results (CPU cosine similarity only)
    spdlog::info("Comparing individual vs batch embedding results (CPU cosine similarity only)...");
    auto batch_embeddings_for_comparison = model_set.text_embedding_model->embed(batch_text);

    // Compute batch embedding + CPU similarity as reference
    std::vector<float> batch_cpu_similarity_scores;
    batch_cpu_similarity_scores.reserve(batch_embeddings_for_comparison.size());

    for (const auto &embedding : batch_embeddings_for_comparison) {
        if (target_embedding.size() == embedding.size()) {
            float similarity = neural_network::cpu::cosine_similarity(target_embedding.data(), embedding.data(),
                                                                      target_embedding.size());
            batch_cpu_similarity_scores.push_back(similarity);
        }
    }

    spdlog::info("=== Text Embedding Model Inference Comparison (CPU Cosine Similarity Only) ===");

    // Main comparison: Individual vs Batch embedding similarity results
    spdlog::info("Individual vs Batch Text Embedding Model Inference Results:");
    for (size_t i = 0; i < std::min(similarity_scores.size(), batch_cpu_similarity_scores.size()); ++i) {
        float diff = std::abs(similarity_scores[i] - batch_cpu_similarity_scores[i]);
        spdlog::info("  Text[{}] \"{}\": Individual_Embed+CPU_Sim={:.6f}, Batch_Embed+CPU_Sim={:.6f}, Diff={:.6f}", i,
                     batch_text[i], similarity_scores[i], batch_cpu_similarity_scores[i], diff);

        // Validate that individual and batch embeddings are consistent (tiny numerical difference allowed)
        EXPECT_LT(diff, 0.0001f)
            << "Individual and batch text embedding inference results differ significantly for text " << i;
    }

    // Summary of comparison results
    spdlog::info("Summary: Text Embedding Model Individual vs Batch Inference Comparison:");
    for (size_t i = 0; i < batch_text.size() && i < similarity_scores.size() && i < batch_cpu_similarity_scores.size();
         ++i) {
        spdlog::info("  Text[{}] \"{}\": ", i, batch_text[i]);
        spdlog::info("    Individual_Embedding+CPU_Similarity: {:.6f}", similarity_scores[i]);
        spdlog::info("    Batch_Embedding+CPU_Similarity:      {:.6f}", batch_cpu_similarity_scores[i]);
    }
}

TEST(LibTorchAccuracy, TextEmbeddingBatchVsIndividualConsistency) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};

    spdlog::info("=== LibTorch Text Embedding Vector Difference Analysis ===");

    // Compute embeddings individually
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
    auto individual_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_individual - start_individual).count();

    // Compute embeddings in batch
    spdlog::info("Computing batch embeddings...");
    auto start_batch = std::chrono::high_resolution_clock::now();
    auto batch_embeddings = model_set.text_embedding_model->embed(batch_text);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();

    spdlog::info("Timing comparison:");
    spdlog::info("  Individual inference: {} ms", individual_duration);
    spdlog::info("  Batch inference: {} ms", batch_duration);
    spdlog::info("  Speed ratio: {:.2f}x", static_cast<float>(individual_duration) / batch_duration);

    // Validate vector dimension consistency
    ASSERT_EQ(individual_embeddings.size(), batch_embeddings.size());

    spdlog::info("=== Vector Difference Analysis ===");

    for (size_t i = 0; i < batch_text.size(); ++i) {
        const auto &individual_vec = individual_embeddings[i];
        const auto &batch_vec = batch_embeddings[i];

        ASSERT_EQ(individual_vec.size(), batch_vec.size()) << "Embedding dimension mismatch for text " << i;

        // Compute vector difference statistics
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

        // Print detailed comparison for the first 20 elements
        spdlog::info("  First 20 elements comparison:");
        size_t first_elements_to_show = std::min(static_cast<size_t>(20), individual_vec.size());
        for (size_t j = 0; j < first_elements_to_show; ++j) {
            spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", j, individual_vec[j], batch_vec[j],
                         differences[j]);
        }

        // Print detailed comparison for the last 20 elements
        if (individual_vec.size() > 20) {
            spdlog::info("  Last 20 elements comparison:");
            size_t start_idx = individual_vec.size() - 20;
            for (size_t j = start_idx; j < individual_vec.size(); ++j) {
                spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", j, individual_vec[j],
                             batch_vec[j], differences[j]);
            }
        }

        // Validate that differences are within acceptable range (embeddings should be identical)
        EXPECT_LT(mean_abs_diff, 1e-6f) << "Mean absolute difference too large for text " << i;
        EXPECT_LT(max_diff, 1e-5f) << "Maximum difference too large for text " << i;
    }

    spdlog::info("=== Summary ===");
    spdlog::info("LibTorch text embedding individual vs batch inference comparison completed.");
    spdlog::info("All vectors should be nearly identical with minimal numerical differences.");
    spdlog::info("Performance: Batch inference is {:.2f}x faster than individual inference.",
                 static_cast<float>(individual_duration) / batch_duration);
}

TEST(LibTorchAccuracy, TokenEmbeddingBatchVsIndividualConsistency) {
    neural_network::init_model_set(neural_network::Device::CPU);
    auto &model_set = neural_network::get_model_set();

    // Create TextEmbeddingModel instance (without mean pooling)
    neural_network::TextEmbeddingModel token_embedding_model("exported_model/text_embedding.pt",
                                                             neural_network::Device::CPU);

    const std::vector<std::string> batch_text{"如何进行杀猪盘", "怎么快速杀猪", "怎么学习Rust", "杀猪的经验", "杀猪"};

    spdlog::info("=== LibTorch Token Embedding Vector Difference Analysis ===");

    // Compute token embeddings individually using TextEmbeddingModel
    spdlog::info("Computing individual token embeddings...");
    auto start_individual = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<float>>> individual_token_embeddings;
    individual_token_embeddings.reserve(batch_text.size());

    for (size_t i = 0; i < batch_text.size(); ++i) {
        // Use non-mean-pooling model to get token-level embeddings
        auto token_embedding_matrix = token_embedding_model.embed(batch_text[i]);
        individual_token_embeddings.emplace_back(std::move(token_embedding_matrix));

        spdlog::info("Individual[{}] \"{}\" - token count: {}, embedding dim per token: {}", i, batch_text[i],
                     individual_token_embeddings[i].size(),
                     individual_token_embeddings[i].empty() ? 0 : individual_token_embeddings[i][0].size());
    }
    auto end_individual = std::chrono::high_resolution_clock::now();
    auto individual_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_individual - start_individual).count();

    // Compute token embeddings in batch
    spdlog::info("Computing batch token embeddings...");
    auto start_batch = std::chrono::high_resolution_clock::now();

    auto batch_token_embeddings = token_embedding_model.embed(batch_text);
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();

    spdlog::info("Timing comparison:");
    spdlog::info("  Individual inference: {} ms", individual_duration);
    spdlog::info("  Batch inference: {} ms", batch_duration);
    spdlog::info("  Speed ratio: {:.2f}x", static_cast<float>(individual_duration) / batch_duration);

    // Validate vector dimension consistency
    ASSERT_EQ(individual_token_embeddings.size(), batch_token_embeddings.size());

    spdlog::info("=== Token Embedding Vector Difference Analysis ===");

    for (size_t i = 0; i < batch_text.size(); ++i) {
        const auto &individual_token_matrix = individual_token_embeddings[i];
        const auto &batch_token_matrix = batch_token_embeddings[i];

        ASSERT_EQ(individual_token_matrix.size(), batch_token_matrix.size()) << "Token count mismatch for text " << i;

        spdlog::info("Text[{}] \"{}\": ", i, batch_text[i]);
        spdlog::info("  Token count: {}", individual_token_matrix.size());

        if (individual_token_matrix.empty()) {
            spdlog::info("  No tokens to compare");
            continue;
        }

        // Aggregate differences across all tokens
        float total_sum_abs_diff = 0.0f;
        float total_max_diff = 0.0f;
        float total_sum_squared_diff = 0.0f;
        size_t total_elements = 0;

        for (size_t token_idx = 0; token_idx < individual_token_matrix.size(); ++token_idx) {
            const auto &individual_token_vec = individual_token_matrix[token_idx];
            const auto &batch_token_vec = batch_token_matrix[token_idx];

            ASSERT_EQ(individual_token_vec.size(), batch_token_vec.size())
                << "Token embedding dimension mismatch for text " << i << " token " << token_idx;

            // Compute vector differences for the current token
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

            spdlog::info("    Token[{}]: dim={}, mean_abs_diff={:.8f}, max_diff={:.8f}, rmse={:.8f}", token_idx,
                         individual_token_vec.size(), token_mean_abs_diff, token_max_diff, token_rmse);

            // Validate per-token differences are within acceptable range
            EXPECT_LT(token_mean_abs_diff, 1e-6f)
                << "Token embedding mean absolute difference too large for text " << i << " token " << token_idx;
            EXPECT_LT(token_max_diff, 1e-5f)
                << "Token embedding maximum difference too large for text " << i << " token " << token_idx;
        }

        // Compute overall difference statistics for the entire text
        if (total_elements > 0) {
            float overall_mean_abs_diff = total_sum_abs_diff / total_elements;
            float overall_rmse = std::sqrt(total_sum_squared_diff / total_elements);

            spdlog::info("  Overall statistics:");
            spdlog::info("    Total elements: {}", total_elements);
            spdlog::info("    Overall mean absolute difference: {:.8f}", overall_mean_abs_diff);
            spdlog::info("    Overall maximum absolute difference: {:.8f}", total_max_diff);
            spdlog::info("    Overall Root Mean Square Error (RMSE): {:.8f}", overall_rmse);

            // Validate overall differences are within acceptable range
            EXPECT_LT(overall_mean_abs_diff, 1e-6f)
                << "Overall token embedding mean absolute difference too large for text " << i;
            EXPECT_LT(total_max_diff, 1e-5f) << "Overall token embedding maximum difference too large for text " << i;
        }

        // Print detailed comparison for the first and last token (first 10 elements)
        if (!individual_token_matrix.empty()) {
            spdlog::info("  First token detailed comparison (first 10 elements):");
            const auto &first_individual = individual_token_matrix[0];
            const auto &first_batch = batch_token_matrix[0];
            size_t elements_to_show = std::min(static_cast<size_t>(10), first_individual.size());
            for (size_t j = 0; j < elements_to_show; ++j) {
                float diff = first_individual[j] - first_batch[j];
                spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", j, first_individual[j],
                             first_batch[j], diff);
            }

            if (individual_token_matrix.size() > 1) {
                spdlog::info("  Last token detailed comparison (first 10 elements):");
                const auto &last_individual = individual_token_matrix.back();
                const auto &last_batch = batch_token_matrix.back();
                elements_to_show = std::min(static_cast<size_t>(10), last_individual.size());
                for (size_t j = 0; j < elements_to_show; ++j) {
                    float diff = last_individual[j] - last_batch[j];
                    spdlog::info("    [{}]: Individual={:.8f}, Batch={:.8f}, Diff={:.8f}", j, last_individual[j],
                                 last_batch[j], diff);
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

#ifdef __USE_PADDLE_INFERENCE__

TEST(LACUtil, Utf8SplitWords) {
    // UTF-8 splitting of mixed Chinese + ASCII
    const std::string text = "如何进行杀猪盘123";
    std::vector<std::string> words;
    auto r =  neural_network::lac::split_words(text, neural_network::lac::CODE_UTF8, words);
    ASSERT_EQ(r, neural_network::lac::SUCCESS);
    ASSERT_EQ(words.size(), 10u); // 7 Chinese chars + 3 ASCII digits
    EXPECT_EQ(words[0], "如");
    EXPECT_EQ(words[1], "何");
    EXPECT_EQ(words[2], "进");
    EXPECT_EQ(words[3], "行");
    EXPECT_EQ(words[4], "杀");
    EXPECT_EQ(words[5], "猪");
    EXPECT_EQ(words[6], "盘");
    EXPECT_EQ(words[7], "1");
    EXPECT_EQ(words[8], "2");
    EXPECT_EQ(words[9], "3");
}

TEST(LACAhoCorasick, BasicMatch) {
    // Build automaton for pattern "杀猪"
    std::vector<std::string> pattern_chars;
    ASSERT_EQ( neural_network::lac::split_words("杀猪", neural_network::lac::CODE_UTF8, pattern_chars), neural_network::lac::SUCCESS);

     neural_network::lac::AhoCorasick ac;
    ac.insert(pattern_chars, 0);
    ac.make_fail();

    // Sentence contains one occurrence
    std::vector<std::string> sent_chars;
    ASSERT_EQ( neural_network::lac::split_words("怎么快速杀猪", neural_network::lac::CODE_UTF8, sent_chars), neural_network::lac::SUCCESS);

    std::vector<std::pair<int, int>> res;
    int count = ac.search(sent_chars, res);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].second, 0);
}

TEST(LACCustomization, LoadAndApply) {
    // Create a minimal customization dict mapping "杀猪" to tag n
    const std::string dict_path = "lac_custom_test.dic";
    {
        std::ofstream ofs(dict_path, std::ios::trunc);
        ASSERT_TRUE(ofs.good());
        ofs << "杀猪/n\n";
    }

    neural_network::lac::Customization custom(dict_path);

    std::vector<std::string> chars;
    ASSERT_EQ(neural_network::lac::split_words("今天要去杀猪", neural_network::lac::CODE_UTF8, chars), neural_network::lac::SUCCESS);
    std::vector<std::string> tag_ids(chars.size(), "O-I");

    // Apply customization
    auto ret = custom.parse_customization(chars, tag_ids);
    ASSERT_EQ(ret, neural_network::lac::SUCCESS);

    // Find start index of "杀猪"
    size_t start = std::string::npos;
    for (size_t i = 0; i + 1 < chars.size(); ++i) {
        if (chars[i] == "杀" && chars[i + 1] == "猪") {
            start = i;
            break;
        }
    }
    ASSERT_NE(start, std::string::npos);

    // Expect tags changed to n-B, n-I for the two characters
    EXPECT_TRUE(tag_ids[start].rfind("n-", 0) == 0);
    EXPECT_EQ(tag_ids[start].back(), 'B');
    EXPECT_TRUE(tag_ids[start + 1].rfind("n-", 0) == 0);
    EXPECT_EQ(tag_ids[start + 1].back(), 'I');

    // Cleanup temporary file
    std::remove(dict_path.c_str());
}

constexpr const char *lac_model_path = "/Users/fredyakumo/models_general/lac_model";

TEST(LACModel, ChineseSegmentationAndPOSTagging) {
    // Initialize LAC segmenter
    neural_network::lac::LAC lac_model(lac_model_path);

    // Test basic segmentation and POS tagging
    const std::string test_text = "原神是一款开放世界冒险游戏";
    auto result = lac_model.run(test_text);

    // Validate the result is not empty
    ASSERT_FALSE(result.empty());

    // Print segmentation and POS tagging result
    std::string result_str;
    for (const auto &item : result) {
        result_str += item.word;
        if (!item.tag.empty()) {
            result_str += "/" + item.tag + " ";
        } else {
            result_str += " ";
        }
    }
    spdlog::info("LAC segmentation result: {}", result_str);

    // Validate that the segmentation contains the words "原神" and "游戏"
    // bool has_genshin = false;
    bool has_game = false;
    for (const auto &item : result) {
        // if (item.word == "原神") {
        //     has_genshin = true;
        // }
        if (item.word == "游戏") {
            has_game = true;
        }
    }
    // EXPECT_TRUE(has_genshin) << "Segmentation should contain '原神'";
    EXPECT_TRUE(has_game) << "Segmentation should contain '游戏'";

    // Test batch processing
    std::vector<std::string> batch_texts = {"璃月港的钟离是岩神巴巴托斯", "雷电将军统治着稻妻城"};
    auto batch_results = lac_model.run(batch_texts);

    // Validate batch results
    ASSERT_EQ(batch_results.size(), batch_texts.size());
    for (size_t i = 0; i < batch_results.size(); ++i) {
        ASSERT_FALSE(batch_results[i].empty());

        std::string batch_result_str;
        for (const auto &item : batch_results[i]) {
            batch_result_str += item.word;
            if (!item.tag.empty()) {
                batch_result_str += "/" + item.tag + " ";
            } else {
                batch_result_str += " ";
            }
        }
        spdlog::info("LAC batch result {}: {}", i, batch_result_str);
    }

    // Test custom dictionary feature
    const std::string dict_path = "lac_model_test.dic";
    {
        std::ofstream ofs(dict_path, std::ios::trunc);
        ASSERT_TRUE(ofs.good());
        ofs << "开放世界冒险/nz\n"; // Treat "开放世界冒险" as a special term
    }

    // Load custom dictionary
    int ret = lac_model.load_customization(dict_path);
    ASSERT_EQ(ret,  neural_network::lac::SUCCESS);

    // Re-run segmentation with custom dictionary
    auto custom_result = lac_model.run(test_text);

    // Validate custom dictionary effectiveness
    bool has_custom_term = false;
    for (const auto &item : custom_result) {
        if (item.word == "开放世界冒险" && item.tag == "nz") {
            has_custom_term = true;
            break;
        }
    }
    EXPECT_TRUE(has_custom_term) << "After custom dictionary, should recognize '开放世界冒险' as a special term";

    // Print segmentation result with custom dictionary
    std::string custom_result_str;
    for (const auto &item : custom_result) {
        custom_result_str += item.word;
        if (!item.tag.empty()) {
            custom_result_str += "/" + item.tag + " ";
        } else {
            custom_result_str += " ";
        }
    }
    spdlog::info("LAC custom dictionary segmentation result: {}", custom_result_str);

    // Cleanup temporary file
    std::remove(dict_path.c_str());
}

TEST(LACModel, InteractiveTest) {
    // Check LAC model path; skip test if files not present
    std::ifstream model_check(std::string(lac_model_path) + "/conf/word.dic");
    if (!model_check.good()) {
        GTEST_SKIP() << "LAC model files not found at " << lac_model_path << ", skipping test";
    }
    model_check.close();

    // Initialize LAC segmenter
     neural_network::lac::LAC lac_model(lac_model_path);

    // Define test cases with Genshin Impact related sentences
    const std::vector<std::string> test_cases = {"原神是一款开放世界冒险游戏", "胡桃是璃月港最受欢迎的角色之一",
                                                 "旅行者与派蒙一起冒险探索提瓦特大陆", "雷电将军统治着稻妻城",
                                                 "甘雨和魈都是璃月的仙人"};

    spdlog::info("Starting LAC interactive segmentation test (simulate Baidu LAC example):");
    for (const auto &query : test_cases) {
        spdlog::info("Input: {}", query);

        // Run segmentation
        auto result = lac_model.run(query);

        // Print segmentation and POS tagging results (formatted like the Baidu LAC example)
        std::string output;
        for (const auto &item : result) {
            if (item.tag.empty()) {
                output += item.word + " ";
            } else {
                output += item.word + "/" + item.tag + " ";
            }
        }
        spdlog::info("Segmentation: {}", output);

        // Validate result is not empty
        ASSERT_FALSE(result.empty());
    }

    // Test custom dictionary
    const std::string dict_path = "lac_interactive_test.dic";
    {
        std::ofstream ofs(dict_path, std::ios::trunc);
        ASSERT_TRUE(ofs.good());
        ofs << "提瓦特大陆/ns\n"; // Define "提瓦特大陆" as a place name
        ofs << "原神/nz\n";       // Define "原神" as a proper noun
    }

    // Load custom dictionary
    int ret = lac_model.load_customization(dict_path);
    ASSERT_EQ(ret,  neural_network::lac::SUCCESS);

    spdlog::info("\nLAC interactive segmentation test with custom dictionary:");
    for (const auto &query : test_cases) {
        spdlog::info("Input: {}", query);

        // Run segmentation
        auto result = lac_model.run(query);

        // Print segmentation and POS tagging results
        std::string output;
        for (const auto &item : result) {
            if (item.tag.empty()) {
                output += item.word + " ";
            } else {
                output += item.word + "/" + item.tag + " ";
            }
        }
        spdlog::info("Segmentation: {}", output);

        // Validate result is not empty
        ASSERT_FALSE(result.empty());
    }

    // Cleanup temporary file
    std::remove(dict_path.c_str());
}

TEST(LACModel, CustomDictSegmentationAndBatchPerformance) {
    // Check LAC model path; skip test if files not present
    std::ifstream model_check(std::string(lac_model_path) + "/conf/word.dic");
    if (!model_check.good()) {
        GTEST_SKIP() << "LAC model files not found at " << lac_model_path << ", skipping test";
    }
    model_check.close();

    // Initialize LAC segmenter
     neural_network::lac::LAC lac_model(lac_model_path);

    // Create a custom dictionary with project-specific terms
    const std::string dict_path = "lac_custom_dict_test.dic";
    {
        std::ofstream ofs(dict_path, std::ios::trunc);
        ASSERT_TRUE(ofs.good());
        ofs << "胡桃/PER\n";
        ofs << "甘雨/PER\n";
        ofs << "雷电将军/PER\n";
        ofs << "钟离/PER\n";
        ofs << "万叶/PER\n";
        ofs << "宵宫/PER\n";
        ofs << "班尼特/PER\n";
        ofs << "火系/n\n";
        ofs << "队伍/n\n";
        ofs << "输出/v\n";
        ofs << "辅助/n\n";
        ofs << "角色/n\n";
    }

    // Load custom dictionary
    int ret = lac_model.load_customization(dict_path);
    ASSERT_EQ(ret,  neural_network::lac::SUCCESS);

    // Define test cases simulating real conversation scenarios
    struct TestCase {
        std::string query;
        std::vector<std::string> expected_words;                             // Expected keywords in segmentation result
        std::vector<std::pair<std::string, std::string>> expected_word_tags; // Expected word-tag pairs
    };

    std::vector<TestCase> test_cases = {
        {"胡桃和甘雨哪个输出更高？", {"胡桃", "甘雨", "输出"}, {{"胡桃", "PER"}, {"甘雨", "PER"}}},
        {"雷电将军和钟离是最强的辅助角色",
         {"雷电将军", "钟离", "辅助", "角色"},
         {{"雷电将军", "PER"}, {"钟离", "PER"}}},
        {
            "万叶可以搭配宵宫和班尼特组成强力火系队伍",
            {"万叶", "宵宫", "班尼特", "火系", "队伍"},
            {} // Do not validate specific POS tags
        }};

    for (const auto &tc : test_cases) {
        spdlog::info("Test sentence: {}", tc.query);

        // Run segmentation
        auto result = lac_model.run(tc.query);

        // Print segmentation and POS tagging results
        std::string output;
        for (const auto &item : result) {
            if (item.tag.empty()) {
                output += item.word + " ";
            } else {
                output += item.word + "/" + item.tag + " ";
            }
        }
        spdlog::info("Segmentation: {}", output);

        // Validate the segmentation contains expected keywords
        for (const auto &expected_word : tc.expected_words) {
            bool found = false;
            for (const auto &item : result) {
                if (item.word == expected_word) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Segmentation should contain keyword '" << expected_word << "'";
        }

        // Validate word-tag pairs
        for (const auto &expected_pair : tc.expected_word_tags) {
            bool found = false;
            for (const auto &item : result) {
                if (item.word == expected_pair.first && item.tag == expected_pair.second) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Segmentation should contain word-tag pair '" << expected_pair.first << "/"
                               << expected_pair.second << "'";
        }
    }

    // Performance test: process multiple sentences in batch and measure time
    const int BATCH_SIZE = 100;
    std::vector<std::string> batch_queries;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        batch_queries.push_back(test_cases[i % test_cases.size()].query);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto batch_results = lac_model.run(batch_queries);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    spdlog::info("Processed {} sentences in {} ms, average per sentence: {:.2f} ms", BATCH_SIZE, duration,
                 static_cast<float>(duration) / BATCH_SIZE);

    // Validate batch results
    ASSERT_EQ(batch_results.size(), batch_queries.size());

    // Cleanup temporary file
    std::remove(dict_path.c_str());
}

#endif // __USE_PADDLE_INFERENCE__