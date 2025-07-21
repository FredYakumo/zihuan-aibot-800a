#include "neural_network/text_model.h"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"

#ifdef __USE_LIBTORCH__
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#endif

namespace neural_network {

#ifndef __USE_LIBTORCH__

    TextEmbeddingModel::TextEmbeddingModel(const std::string &model_path, Device device)
        : m_session(get_onnx_runtime(), model_path.c_str(), get_session_options(device)) {

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        m_input_names.reserve(2);
        for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
            Ort::AllocatedStringPtr name_ptr = m_session.GetInputNameAllocated(i, allocator);
            m_input_names.push_back(name_ptr.get());
            m_input_names_ptr.emplace_back(std::move(name_ptr));
        }

        m_output_names.reserve(1);
        for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
            Ort::AllocatedStringPtr name_ptr = m_session.GetOutputNameAllocated(i, allocator);
            m_output_names.push_back(name_ptr.get());
            m_output_names_ptr.emplace_back(std::move(name_ptr));
        }
    }

    /**
     * @brief Get the token embeddings for a given text.
     * @param text Input text
     * @return A matrix representing the embeddings for each token.
     */
    emb_mat_t TextEmbeddingModel::embed(const std::string &text) {
        auto token_ids = get_model_set().tokenizer_wrapper.encode(text);
        attention_mask_list_t attention_mask(token_ids.size(), 1);
        return embed(token_ids, attention_mask);
    }

    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        std::vector<token_id_list_t> token_ids = get_model_set().tokenizer_wrapper.encode_batch(texts);
        std::vector<attention_mask_list_t> attention_mask_list;
        attention_mask_list.reserve(token_ids.size());
        for (const auto &ids : token_ids) {
            attention_mask_list_t mask(ids.size(), 1);
            attention_mask_list.push_back(std::move(mask));
        }
        return embed(token_ids, attention_mask_list, max_batch_size);
    }

    emb_mat_t TextEmbeddingModel::embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask) {
        assert(token_ids.size() == attention_mask.size());

        // Truncate input to maximum length of EMBEDDING_MAX_INPUT_LENGTH
        const size_t max_len = EMBEDDING_MAX_INPUT_LENGTH;
        auto token_end = token_ids.size() > max_len ? token_ids.begin() + max_len : token_ids.end();
        auto mask_end = attention_mask.size() > max_len ? attention_mask.begin() + max_len : attention_mask.end();

        const size_t seq_len = std::distance(token_ids.begin(), token_end);
        const std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};

        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_ids(token_ids.begin(), token_end);
        std::vector<int64_t> masks(attention_mask.begin(), mask_end);

        // Create input_ids, attention_mask tensor
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(m_memory_info, input_ids.data(), input_ids.size(),
                                                                     input_shape.data(), input_shape.size()));

        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(m_memory_info, masks.data(), masks.size(),
                                                                     input_shape.data(), input_shape.size()));

        // inference
        auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                            input_tensors.size(), m_output_names.data(), m_output_names.size());

        // get token embeddings [batch_size, seq_len, hidden_size]
        float *token_embeddings = output_tensors[0].GetTensorMutableData<float>();
        auto embeddings_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        const int64_t hidden_size = embeddings_shape[2];

        emb_mat_t result;
        result.reserve(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            const float *start = token_embeddings + i * hidden_size;
            result.emplace_back(start, start + hidden_size);
        }
        return result;
    }

    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<token_id_list_t> &token_ids,
                                                     const std::vector<attention_mask_list_t> &attention_mask, size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto batch_size = token_ids.size();
        if (batch_size == 0) {
            return {};
        }

        // If batch size is within limit, process normally
        if (batch_size <= max_batch_size) {
            size_t max_seq_len = 0;
            for (const auto &ids : token_ids) {
                if (ids.size() > max_seq_len) {
                    max_seq_len = ids.size();
                }
            }

            std::vector<int64_t> input_ids_flat;
            input_ids_flat.reserve(batch_size * max_seq_len);
            std::vector<int64_t> masks_flat;
            masks_flat.reserve(batch_size * max_seq_len);

            for (size_t i = 0; i < batch_size; ++i) {
                const auto &current_ids = token_ids[i];
                const size_t trunc_len = std::min(current_ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                for (size_t j = 0; j < trunc_len; ++j) {
                    input_ids_flat.push_back(current_ids[j]);
                }
                input_ids_flat.insert(input_ids_flat.end(), EMBEDDING_MAX_INPUT_LENGTH - trunc_len, 0LL);

                const auto &current_mask = attention_mask[i];
                const size_t mask_trunc_len = std::min(current_mask.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                for (size_t j = 0; j < mask_trunc_len; ++j) {
                    masks_flat.push_back(current_mask[j]);
                }
                masks_flat.insert(masks_flat.end(), EMBEDDING_MAX_INPUT_LENGTH - mask_trunc_len, 0LL);
            }

            const std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len)};

            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                m_memory_info, input_ids_flat.data(), input_ids_flat.size(), input_shape.data(), input_shape.size()));

            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                m_memory_info, masks_flat.data(), masks_flat.size(), input_shape.data(), input_shape.size()));

            auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                                input_tensors.size(), m_output_names.data(), m_output_names.size());

            float *token_embeddings = output_tensors[0].GetTensorMutableData<float>();
            auto embeddings_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            const int64_t hidden_size = embeddings_shape[2];

            std::vector<emb_mat_t> result;
            result.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                emb_mat_t current_emb_mat;
                const size_t original_seq_len = token_ids[i].size();
                current_emb_mat.reserve(original_seq_len);
                for (size_t j = 0; j < original_seq_len; ++j) {
                    const float *start = token_embeddings + i * max_seq_len * hidden_size + j * hidden_size;
                    current_emb_mat.emplace_back(start, start + hidden_size);
                }
                result.emplace_back(std::move(current_emb_mat));
            }
            return result;
        }

        // Split into batches and process iteratively
        std::vector<emb_mat_t> result;
        result.reserve(batch_size);

        for (size_t start = 0; start < batch_size; start += max_batch_size) {
            const size_t end = std::min(start + max_batch_size, batch_size);
            
            std::vector<token_id_list_t> batch_token_ids(token_ids.begin() + start, token_ids.begin() + end);
            std::vector<attention_mask_list_t> batch_attention_mask(attention_mask.begin() + start, attention_mask.begin() + end);
            
            auto batch_result = embed(batch_token_ids, batch_attention_mask, max_batch_size);
            result.insert(result.end(), std::make_move_iterator(batch_result.begin()), std::make_move_iterator(batch_result.end()));
        }

        return result;
    }

    TextEmbeddingWithMeanPoolingModel::TextEmbeddingWithMeanPoolingModel(const std::string &model_path, Device device)
        : m_session(get_onnx_runtime(), model_path.c_str(), get_session_options(device)) {

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        m_input_names.reserve(2);
        for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
            Ort::AllocatedStringPtr name_ptr = m_session.GetInputNameAllocated(i, allocator);
            m_input_names.push_back(name_ptr.get());
            m_input_names_ptr.emplace_back(std::move(name_ptr));
        }

        m_output_names.reserve(1);
        for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
            Ort::AllocatedStringPtr name_ptr = m_session.GetOutputNameAllocated(i, allocator);
            m_output_names.push_back(name_ptr.get());
            m_output_names_ptr.emplace_back(std::move(name_ptr));
        }
    }

    /**
     * @brief Get the token embeddings for a given text.
     * @param text Input text
     * @return A matrix representing the embeddings for each token.
     */
    emb_vec_t TextEmbeddingWithMeanPoolingModel::embed(const std::string &text) {
        auto token_ids = get_model_set().tokenizer_wrapper.encode(text);
        attention_mask_list_t attention_mask(token_ids.size(), 1);
        return embed(token_ids, attention_mask);
    }

    emb_mat_t TextEmbeddingWithMeanPoolingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        std::vector<token_id_list_t> token_ids = get_model_set().tokenizer_wrapper.encode_batch(texts);
        std::vector<attention_mask_list_t> attention_mask_list;
        attention_mask_list.reserve(token_ids.size());
        for (const auto &ids : token_ids) {
            attention_mask_list_t mask(ids.size(), 1);
            attention_mask_list.push_back(std::move(mask));
        }
        return embed(token_ids, attention_mask_list, max_batch_size);
    }

    emb_vec_t TextEmbeddingWithMeanPoolingModel::embed(const token_id_list_t &token_ids,
                                                       const attention_mask_list_t &attention_mask) {
        assert(token_ids.size() == attention_mask.size());

        // Truncate input to maximum length of EMBEDDING_MAX_INPUT_LENGTH
        const size_t max_len = EMBEDDING_MAX_INPUT_LENGTH;
        auto token_end = token_ids.size() > max_len ? token_ids.begin() + max_len : token_ids.end();
        auto mask_end = attention_mask.size() > max_len ? attention_mask.begin() + max_len : attention_mask.end();

        const size_t seq_len = std::distance(token_ids.begin(), token_end);
        const std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};

        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_ids(token_ids.begin(), token_end);
        std::vector<int64_t> masks(attention_mask.begin(), mask_end);

        // Create input_ids, attention_mask tensor
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(m_memory_info, input_ids.data(), input_ids.size(),
                                                                     input_shape.data(), input_shape.size()));

        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(m_memory_info, masks.data(), masks.size(),
                                                                     input_shape.data(), input_shape.size()));

        // inference
        auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                            input_tensors.size(), m_output_names.data(), m_output_names.size());

        // get sentence embedding [1, hidden_size]
        float *sentence_embedding = output_tensors[0].GetTensorMutableData<float>();
        auto embeddings_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        const int64_t hidden_size = embeddings_shape[1];

        return {sentence_embedding, sentence_embedding + hidden_size};
    }

    emb_mat_t TextEmbeddingWithMeanPoolingModel::embed(const std::vector<token_id_list_t> &token_ids,
                                                       const std::vector<attention_mask_list_t> &attention_mask, size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto batch_size = token_ids.size();
        if (batch_size == 0) {
            return {};
        }

        // If batch size is within limit, process normally
        if (batch_size <= max_batch_size) {
            size_t max_seq_len = 0;
            for (const auto &ids : token_ids) {
                const size_t current_len = std::min(ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                if (current_len > max_seq_len) {
                    max_seq_len = current_len;
                }
            }

            std::vector<int64_t> input_ids_flat;
            input_ids_flat.reserve(batch_size * EMBEDDING_MAX_INPUT_LENGTH);
            std::vector<int64_t> masks_flat;
            masks_flat.reserve(batch_size * EMBEDDING_MAX_INPUT_LENGTH);

            for (size_t i = 0; i < batch_size; ++i) {
                const auto &current_ids = token_ids[i];
                const size_t trunc_len = std::min(current_ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                for (size_t j = 0; j < trunc_len; ++j) {
                    input_ids_flat.push_back(current_ids[j]);
                }
                input_ids_flat.insert(input_ids_flat.end(), EMBEDDING_MAX_INPUT_LENGTH - trunc_len, 0LL);

                const auto &current_mask = attention_mask[i];
                const size_t mask_trunc_len = std::min(current_mask.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                for (size_t j = 0; j < mask_trunc_len; ++j) {
                    masks_flat.push_back(current_mask[j]);
                }
                masks_flat.insert(masks_flat.end(), EMBEDDING_MAX_INPUT_LENGTH - mask_trunc_len, 0LL);
            }

            const std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), EMBEDDING_MAX_INPUT_LENGTH};

            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                m_memory_info, input_ids_flat.data(), input_ids_flat.size(), input_shape.data(), input_shape.size()));

            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                m_memory_info, masks_flat.data(), masks_flat.size(), input_shape.data(), input_shape.size()));

            auto output_tensors = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), input_tensors.data(),
                                                input_tensors.size(), m_output_names.data(), m_output_names.size());

            // get sentence embeddings [batch_size, hidden_size]
            float *sentence_embeddings = output_tensors[0].GetTensorMutableData<float>();
            auto embeddings_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            const int64_t hidden_size = embeddings_shape[1];

            emb_mat_t result;
            result.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                const float *start = sentence_embeddings + i * hidden_size;
                result.emplace_back(start, start + hidden_size);
            }
            return result;
        }

        // Split into batches and process iteratively
        emb_mat_t result;
        result.reserve(batch_size);

        for (size_t start = 0; start < batch_size; start += max_batch_size) {
            const size_t end = std::min(start + max_batch_size, batch_size);
            
            std::vector<token_id_list_t> batch_token_ids(token_ids.begin() + start, token_ids.begin() + end);
            std::vector<attention_mask_list_t> batch_attention_mask(attention_mask.begin() + start, attention_mask.begin() + end);
            
            auto batch_result = embed(batch_token_ids, batch_attention_mask, max_batch_size);
            result.insert(result.end(), std::make_move_iterator(batch_result.begin()), std::make_move_iterator(batch_result.end()));
        }

        return result;
    }
#endif // __USE_LIBTORCH__

#ifdef __USE_LIBTORCH__

    TextEmbeddingModel::TextEmbeddingModel(const std::string &model_path, Device device) {
        try {
            m_module = torch::jit::load(model_path, get_torch_device(device));
            m_module.eval(); // Set to evaluation mode
            spdlog::info("Successfully loaded PyTorch TextEmbeddingModel from: {}", model_path);
        } catch (const std::exception &e) {
            spdlog::error("Failed to load PyTorch TextEmbeddingModel from {}: {}", model_path, e.what());
            throw;
        }
    }

    /**
     * @brief Get the token embeddings for a given text.
     * @param text Input text
     * @return A matrix representing the embeddings for each token.
     */
    emb_mat_t TextEmbeddingModel::embed(const std::string &text) {
        auto token_ids = get_model_set().tokenizer_wrapper.encode(text);
        attention_mask_list_t attention_mask(token_ids.size(), 1);
        return embed(token_ids, attention_mask);
    }

    /**
     * @brief Get the token embeddings for multiple texts.
     * @param texts List of input texts
     * @return A list of matrices representing the embeddings for each text.
     */
    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        std::vector<token_id_list_t> token_ids = get_model_set().tokenizer_wrapper.encode_batch(texts);
        std::vector<attention_mask_list_t> attention_mask_list;
        attention_mask_list.reserve(token_ids.size());
        for (const auto &ids : token_ids) {
            attention_mask_list_t mask(ids.size(), 1);
            attention_mask_list.push_back(std::move(mask));
        }
        return embed(token_ids, attention_mask_list, max_batch_size);
    }

    /**
     * @brief Get the token embeddings for a given tokenized text.
     * @return A matrix representing the embeddings for each token.
     */
    emb_mat_t TextEmbeddingModel::embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask) {
        assert(token_ids.size() == attention_mask.size());

        // Truncate input to maximum length of EMBEDDING_MAX_INPUT_LENGTH
        const size_t max_len = EMBEDDING_MAX_INPUT_LENGTH;
        auto token_end = token_ids.size() > max_len ? token_ids.begin() + max_len : token_ids.end();
        auto mask_end = attention_mask.size() > max_len ? attention_mask.begin() + max_len : attention_mask.end();

        const size_t seq_len = std::distance(token_ids.begin(), token_end);

        // Convert to tensors
        std::vector<int64_t> input_ids_vec(token_ids.begin(), token_end);
        std::vector<int64_t> attention_mask_vec(attention_mask.begin(), mask_end);

        auto device = (*m_module.parameters().begin()).device();
        torch::Tensor input_ids_tensor =
            torch::tensor(input_ids_vec, torch::dtype(torch::kLong)).unsqueeze(0).to(device);
        torch::Tensor attention_mask_tensor =
            torch::tensor(attention_mask_vec, torch::dtype(torch::kLong)).unsqueeze(0).to(device);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids_tensor);
        inputs.push_back(attention_mask_tensor);

        torch::jit::IValue output = m_module.forward(inputs);
        
        // Model now always returns tensor embeddings directly
        torch::Tensor token_embeddings_tensor = output.toTensor();

        // Convert back to matrix [seq_len, hidden_size]
        token_embeddings_tensor = token_embeddings_tensor.to(torch::kCPU).squeeze(0);
        auto embeddings_ptr = token_embeddings_tensor.data_ptr<float>();
        auto hidden_size = token_embeddings_tensor.size(1);

        emb_mat_t result;
        result.reserve(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            const float *start = embeddings_ptr + i * hidden_size;
            result.emplace_back(start, start + hidden_size);
        }
        return result;
    }

    /**
     * @brief Get the token embeddings for multiple tokenized texts.
     * @param token_ids List of token ID sequences
     * @param attention_mask List of attention mask sequences
     * @return A list of matrices representing the embeddings for each text
     */
    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<token_id_list_t> &token_ids,
                                                     const std::vector<attention_mask_list_t> &attention_mask, size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto batch_size = token_ids.size();
        if (batch_size == 0) {
            return {};
        }

        // If batch size is within limit, process normally
        if (batch_size <= max_batch_size) {
            // For small batches, use sequential processing to save memory
            if (batch_size <= 4) {
                std::vector<emb_mat_t> result;
                result.reserve(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    result.emplace_back(embed(token_ids[i], attention_mask[i]));
                }
                return result;
            }

            // Find actual max sequence length in this batch (capped at EMBEDDING_MAX_INPUT_LENGTH)
            size_t max_seq_len = 0;
            for (const auto &ids : token_ids) {
                max_seq_len = std::max(max_seq_len, std::min(ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH)));
            }
            
            auto device = (*m_module.parameters().begin()).device();

            // Create tensors with minimal required size
            torch::Tensor input_ids_tensor = torch::zeros({static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len)}, 
                                                          torch::dtype(torch::kLong).device(torch::kCPU));
            torch::Tensor attention_mask_tensor = torch::zeros({static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len)}, 
                                                               torch::dtype(torch::kLong).device(torch::kCPU));

            // Use tensor accessors to fill data efficiently on CPU
            auto input_ids_accessor = input_ids_tensor.accessor<int64_t, 2>();
            auto attention_mask_accessor = attention_mask_tensor.accessor<int64_t, 2>();

            for (size_t i = 0; i < batch_size; ++i) {
                const auto &current_ids = token_ids[i];
                const size_t trunc_len = std::min(current_ids.size(), max_seq_len);
                
                for (size_t j = 0; j < trunc_len; ++j) {
                    input_ids_accessor[i][j] = static_cast<int64_t>(current_ids[j]);
                }

                const auto &current_mask = attention_mask[i];
                const size_t mask_trunc_len = std::min(current_mask.size(), max_seq_len);
                
                for (size_t j = 0; j < mask_trunc_len; ++j) {
                    attention_mask_accessor[i][j] = static_cast<int64_t>(current_mask[j]);
                }
            }

            // Transfer to device only once
            input_ids_tensor = input_ids_tensor.to(device);
            attention_mask_tensor = attention_mask_tensor.to(device);

            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(std::move(input_ids_tensor));
            inputs.emplace_back(std::move(attention_mask_tensor));

            torch::jit::IValue output;
            {
                torch::NoGradGuard no_grad;
                output = m_module.forward(inputs);
            }
            
            // Model now always returns tensor embeddings directly
            torch::Tensor token_embeddings_tensor = output.toTensor();

            // Convert back to list of matrices [batch_size, seq_len, hidden_size]
            token_embeddings_tensor = token_embeddings_tensor.to(torch::kCPU);
            auto embeddings_ptr = token_embeddings_tensor.data_ptr<float>();
            auto hidden_size = token_embeddings_tensor.size(2);

            std::vector<emb_mat_t> result;
            result.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                emb_mat_t current_emb_mat;
                const size_t original_seq_len = std::min(token_ids[i].size(), max_seq_len);
                current_emb_mat.reserve(original_seq_len);
                for (size_t j = 0; j < original_seq_len; ++j) {
                    const float *start = embeddings_ptr + i * max_seq_len * hidden_size + j * hidden_size;
                    current_emb_mat.emplace_back(start, start + hidden_size);
                }
                result.emplace_back(std::move(current_emb_mat));
            }
            return result;
        }

        // Split into batches and process iteratively
        std::vector<emb_mat_t> result;
        result.reserve(batch_size);

        for (size_t start = 0; start < batch_size; start += max_batch_size) {
            const size_t end = std::min(start + max_batch_size, batch_size);
            
            std::vector<token_id_list_t> batch_token_ids(token_ids.begin() + start, token_ids.begin() + end);
            std::vector<attention_mask_list_t> batch_attention_mask(attention_mask.begin() + start, attention_mask.begin() + end);
            
            auto batch_result = embed(batch_token_ids, batch_attention_mask, max_batch_size);
            result.insert(result.end(), std::make_move_iterator(batch_result.begin()), std::make_move_iterator(batch_result.end()));
        }

        return result;
    }

    TextEmbeddingWithMeanPoolingModel::TextEmbeddingWithMeanPoolingModel(const std::string &model_path, Device device) {
        try {
            m_module = torch::jit::load(model_path, get_torch_device(device));
            m_module.eval(); // Set to evaluation mode
            spdlog::info("Successfully loaded PyTorch model from: {}", model_path);
        } catch (const std::exception &e) {
            spdlog::error("Failed to load PyTorch model from {}: {}", model_path, e.what());
            throw;
        }
    }

    /**
     * @brief Get the sentence embedding for a given text.
     * @param text Input text
     * @return A vector representing the sentence embedding.
     */
    emb_vec_t TextEmbeddingWithMeanPoolingModel::embed(const std::string &text) {
        auto token_ids = get_model_set().tokenizer_wrapper.encode(text);
        attention_mask_list_t attention_mask(token_ids.size(), 1);
        return embed(token_ids, attention_mask);
    }

    /**
     * @brief Get the sentence embeddings for multiple texts.
     * @param texts List of input texts
     * @return A matrix where each row is a sentence embedding.
     */
    emb_mat_t TextEmbeddingWithMeanPoolingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        if (texts.empty()) {
            return {};
        }

        // TODO: 这里如果使用了batch encode，批量embedding的结果会有问题
        // std::vector<token_id_list_t> token_ids = get_model_set().tokenizer_wrapper.encode_batch(texts);
        std::vector<token_id_list_t> token_ids;
        for (const auto &text : texts) {
            token_ids.push_back(get_model_set().tokenizer_wrapper.encode(text));
        }
        std::vector<attention_mask_list_t> attention_mask_list;
        attention_mask_list.reserve(token_ids.size());
        for (const auto &ids : token_ids) {
            attention_mask_list_t mask(ids.size(), 1);
            attention_mask_list.push_back(std::move(mask));
        }

        // Delegate to the optimized batch function
        return embed(token_ids, attention_mask_list, max_batch_size);
    }

    /**
     * @brief Get the sentence embedding for a given tokenized text.
     * @return A vector representing the sentence embedding.
     */
    emb_vec_t TextEmbeddingWithMeanPoolingModel::embed(const token_id_list_t &token_ids,
                                                       const attention_mask_list_t &attention_mask) {
        assert(token_ids.size() == attention_mask.size());

        // Truncate and pad input to fixed length of EMBEDDING_MAX_INPUT_LENGTH
        const size_t max_len = EMBEDDING_MAX_INPUT_LENGTH;
        auto token_end = token_ids.size() > max_len ? token_ids.begin() + max_len : token_ids.end();
        auto mask_end = attention_mask.size() > max_len ? attention_mask.begin() + max_len : attention_mask.end();

        const size_t seq_len = std::distance(token_ids.begin(), token_end);

        // Convert to tensors with padding to EMBEDDING_MAX_INPUT_LENGTH
        std::vector<int64_t> input_ids_vec(token_ids.begin(), token_end);
        std::vector<int64_t> attention_mask_vec(attention_mask.begin(), mask_end);
        
        // Pad to EMBEDDING_MAX_INPUT_LENGTH length
        input_ids_vec.resize(max_len, 0LL);
        attention_mask_vec.resize(max_len, 0LL);

        auto device = (*m_module.parameters().begin()).device();
        torch::Tensor input_ids_tensor =
            torch::tensor(input_ids_vec, torch::dtype(torch::kLong)).unsqueeze(0).to(device);
        torch::Tensor attention_mask_tensor =
            torch::tensor(attention_mask_vec, torch::dtype(torch::kLong)).unsqueeze(0).to(device);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids_tensor);
        inputs.push_back(attention_mask_tensor);

        torch::jit::IValue output = m_module.forward(inputs);
        torch::Tensor embedding_tensor = output.toTensor();

        // Convert back to vector
        embedding_tensor = embedding_tensor.to(torch::kCPU).squeeze(0);
        auto embedding_ptr = embedding_tensor.data_ptr<float>();
        auto hidden_size = embedding_tensor.size(0);

        return emb_vec_t(embedding_ptr, embedding_ptr + hidden_size);
    }

    /**
     * @brief Get the sentence embeddings for multiple tokenized texts.
     * @param token_ids List of token ID sequences
     * @param attention_mask List of attention mask sequences
     * @return A matrix where each row is a sentence embedding.
     */
    emb_mat_t TextEmbeddingWithMeanPoolingModel::embed(const std::vector<token_id_list_t> &token_ids,
                                                       const std::vector<attention_mask_list_t> &attention_mask, size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto batch_size = token_ids.size();
        if (batch_size == 0) {
            return {};
        }

        // If batch size is within limit, process normally
        if (batch_size <= max_batch_size) {
            // For small batches, use sequential processing to save memory
            if (batch_size <= 4) {
                emb_mat_t result;
                result.reserve(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    result.emplace_back(embed(token_ids[i], attention_mask[i]));
                }
                return result;
            }

            auto device = (*m_module.parameters().begin()).device();

            // Use fixed EMBEDDING_MAX_INPUT_LENGTH for batching (all sequences padded to same length)
            const size_t fixed_seq_len = EMBEDDING_MAX_INPUT_LENGTH;

            // Create tensors with fixed size for efficient batching
            torch::Tensor input_ids_tensor = torch::zeros({static_cast<int64_t>(batch_size), static_cast<int64_t>(fixed_seq_len)}, 
                                                          torch::dtype(torch::kLong).device(torch::kCPU));
            torch::Tensor attention_mask_tensor = torch::zeros({static_cast<int64_t>(batch_size), static_cast<int64_t>(fixed_seq_len)}, 
                                                               torch::dtype(torch::kLong).device(torch::kCPU));

            // Use tensor accessors to fill data efficiently on CPU
            auto input_ids_accessor = input_ids_tensor.accessor<int64_t, 2>();
            auto attention_mask_accessor = attention_mask_tensor.accessor<int64_t, 2>();

            for (size_t i = 0; i < batch_size; ++i) {
                const auto &current_ids = token_ids[i];
                const size_t trunc_len = std::min(current_ids.size(), fixed_seq_len);
                
                for (size_t j = 0; j < trunc_len; ++j) {
                    input_ids_accessor[i][j] = static_cast<int64_t>(current_ids[j]);
                }

                const auto &current_mask = attention_mask[i];
                const size_t mask_trunc_len = std::min(current_mask.size(), fixed_seq_len);
                
                for (size_t j = 0; j < mask_trunc_len; ++j) {
                    attention_mask_accessor[i][j] = static_cast<int64_t>(current_mask[j]);
                }
            }

            // Transfer to device only once
            input_ids_tensor = input_ids_tensor.to(device);
            attention_mask_tensor = attention_mask_tensor.to(device);

            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(std::move(input_ids_tensor));
            inputs.emplace_back(std::move(attention_mask_tensor));

            torch::jit::IValue output;
            {
                torch::NoGradGuard no_grad;
                output = m_module.forward(inputs);
            }
            torch::Tensor sentence_embeddings_tensor = output.toTensor();

            // Convert back to matrix [batch_size, hidden_size]
            sentence_embeddings_tensor = sentence_embeddings_tensor.to(torch::kCPU);
            auto embeddings_ptr = sentence_embeddings_tensor.data_ptr<float>();
            auto hidden_size = sentence_embeddings_tensor.size(1);

            emb_mat_t result;
            result.reserve(batch_size);

            for (size_t i = 0; i < batch_size; ++i) {
                const float *start = embeddings_ptr + i * hidden_size;
                result.emplace_back(start, start + hidden_size);
            }
            return result;
        }

        // Split into batches and process iteratively
        emb_mat_t result;
        result.reserve(batch_size);

        for (size_t start = 0; start < batch_size; start += max_batch_size) {
            const size_t end = std::min(start + max_batch_size, batch_size);
            
            std::vector<token_id_list_t> batch_token_ids(token_ids.begin() + start, token_ids.begin() + end);
            std::vector<attention_mask_list_t> batch_attention_mask(attention_mask.begin() + start, attention_mask.begin() + end);
            
            auto batch_result = embed(batch_token_ids, batch_attention_mask, max_batch_size);
            result.insert(result.end(), std::make_move_iterator(batch_result.begin()), std::make_move_iterator(batch_result.end()));
        }

        return result;
    }

#endif // __USE_LIBTORCH__

} // namespace neural_network