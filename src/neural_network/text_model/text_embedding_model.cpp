#include "neural_network/text_model/text_embedding_model.h"
#include "neural_network/model_set.h"
#include "neural_network/nn.h"

#ifdef __USE_LIBTORCH__
#include <torch/csrc/jit/api/module.h>
#include <torch/cuda.h>
#include <torch/script.h>
#endif

namespace neural_network {

#ifdef __USE_ONNX_RUNTIME__

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
        auto [token_ids, attention_mask] = get_model_set().tokenizer_wrapper.encode(text);
        return embed(token_ids, attention_mask);
    }

    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        std::vector<token_id_vec_t> token_ids;
        std::vector<attention_mask_list_t> attention_masks;
        token_ids.reserve(texts.size());
        attention_masks.reserve(texts.size());
        for (const auto &text : texts) {
            auto [ids, mask] = get_model_set().tokenizer_wrapper.encode(text);
            token_ids.emplace_back(std::move(ids));
            attention_masks.emplace_back(std::move(mask));
        }
        return embed(token_ids, attention_masks);
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
                                                     const std::vector<attention_mask_list_t> &attention_mask,
                                                     size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto batch_size = token_ids.size();
        if (batch_size == 0) {
            return {};
        }

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
            const size_t mask_trunc_len =
                std::min(current_mask.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
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

#endif

#ifdef __USE_LIBTORCH__

    TextEmbeddingModel::TextEmbeddingModel(const std::string &model_path, Device device)
        : m_device(get_torch_device(device)) {
        try {
            m_module = torch::jit::load(model_path, m_device);
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
        auto [token_ids, attention_mask] = get_model_set().tokenizer_wrapper.encode(text);
        return embed(token_ids, attention_mask);
    }

    /**
     * @brief Get the token embeddings for multiple texts.
     * @param texts List of input texts
     * @return A list of matrices representing the embeddings for each text.
     */
    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts, size_t max_batch_size) {
        std::vector<token_id_list_t> token_ids;
        std::vector<attention_mask_list_t> attention_mask_list;
        token_ids.reserve(texts.size());
        attention_mask_list.reserve(texts.size());
        for (const auto &text : texts) {
            auto [ids, mask] = get_model_set().tokenizer_wrapper.encode(text);
            token_ids.emplace_back(std::move(ids));
            attention_mask_list.emplace_back(std::move(mask));
        }
        return embed(token_ids, attention_mask_list);
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

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::tensor(std::vector<int32_t>(token_ids.begin(), token_end), torch::dtype(torch::kInt32))
                             .unsqueeze(0)
                             .to(m_device));
        inputs.push_back(
            torch::tensor(std::vector<int32_t>(attention_mask.begin(), mask_end), torch::dtype(torch::kInt32))
                .unsqueeze(0)
                .to(m_device));

        torch::jit::IValue output = m_module.forward(inputs);
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
                                                     const std::vector<attention_mask_list_t> &attention_mask,
                                                     size_t max_batch_size) {
        assert(token_ids.size() == attention_mask.size());
        const auto total_size = token_ids.size();
        if (total_size == 0) {
            return {};
        }

        std::vector<emb_mat_t> result;
        result.reserve(total_size);

        // Process in batches
        for (size_t start_idx = 0; start_idx < total_size; start_idx += max_batch_size) {
            const size_t end_idx = std::min(start_idx + max_batch_size, total_size);
            const size_t current_batch_size = end_idx - start_idx;

            // Find the maximum sequence length in current batch (after truncation)
            size_t max_seq_len = 0;
            for (size_t i = 0; i < current_batch_size; ++i) {
                const auto &current_ids = token_ids[start_idx + i];
                // Truncate each sequence to EMBEDDING_MAX_INPUT_LENGTH first
                const size_t trunc_len = std::min(current_ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                // Find the maximum among all truncated lengths
                max_seq_len = std::max(max_seq_len, trunc_len);
            }

            // Create tensors with the correct shape
            torch::Tensor input_ids_tensor =
                torch::zeros({static_cast<int64_t>(current_batch_size), static_cast<int64_t>(max_seq_len)},
                             torch::TensorOptions().dtype(torch::kInt32));
            torch::Tensor attention_mask_tensor =
                torch::zeros({static_cast<int64_t>(current_batch_size), static_cast<int64_t>(max_seq_len)},
                             torch::TensorOptions().dtype(torch::kInt32));

            // Get direct pointer to tensor data for batch copy operations
            int32_t *input_ids_data = input_ids_tensor.data_ptr<int32_t>();
            int32_t *attention_mask_data = attention_mask_tensor.data_ptr<int32_t>();

            // Store the original sequence lengths for each item in the batch
            std::vector<size_t> original_seq_lengths;
            original_seq_lengths.reserve(current_batch_size);

            for (size_t i = 0; i < current_batch_size; ++i) {
                const auto &current_ids = token_ids[start_idx + i];
                const auto &current_mask = attention_mask[start_idx + i];

                const size_t trunc_len = std::min(current_ids.size(), static_cast<size_t>(EMBEDDING_MAX_INPUT_LENGTH));
                original_seq_lengths.push_back(trunc_len);
                const size_t row_offset = i * max_seq_len;

                // Batch copy using std::copy (much faster than element-by-element)
                std::copy(current_ids.begin(), current_ids.begin() + trunc_len, input_ids_data + row_offset);
                std::copy(current_mask.begin(), current_mask.begin() + trunc_len, attention_mask_data + row_offset);
                // Remaining positions are already zero-initialized by torch::zeros
            }

            // Move to target device
            input_ids_tensor = input_ids_tensor.to(m_device);
            attention_mask_tensor = attention_mask_tensor.to(m_device);

            // Run inference with no_grad to save memory
            torch::Tensor token_embeddings_tensor;
            {
                torch::NoGradGuard no_grad; // Disable gradient computation to save memory
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_ids_tensor);
                inputs.push_back(attention_mask_tensor);

                torch::jit::IValue output = m_module.forward(inputs);
                token_embeddings_tensor = output.toTensor();

                // Convert to CPU immediately to free GPU memory
                token_embeddings_tensor = token_embeddings_tensor.to(torch::kCPU);
            }

            // Clear GPU tensors early
            input_ids_tensor = torch::Tensor{};
            attention_mask_tensor = torch::Tensor{};

            // Force memory synchronization if using CUDA
            if (m_device.is_cuda()) {
                torch::cuda::synchronize();
            }

            // Extract token embeddings efficiently
            // Shape: [batch_size, seq_len, hidden_size]
            auto embeddings_accessor = token_embeddings_tensor.accessor<float, 3>();
            const auto hidden_size = token_embeddings_tensor.size(2);

            // Add token embeddings matrices to result
            for (size_t i = 0; i < current_batch_size; ++i) {
                emb_mat_t current_emb_mat;
                current_emb_mat.reserve(max_seq_len); // Reserve space for padded length

                for (size_t j = 0; j < max_seq_len; ++j) { // Iterate up to padded length
                    emb_vec_t token_embedding;
                    token_embedding.reserve(hidden_size);
                    const float *token_data = &embeddings_accessor[i][j][0];
                    token_embedding.assign(token_data, token_data + hidden_size);
                    current_emb_mat.emplace_back(std::move(token_embedding));
                }
                result.emplace_back(std::move(current_emb_mat));
            }
        }

        return result;
    }

#endif

} // namespace neural_network
