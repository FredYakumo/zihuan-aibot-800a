#include "neural_network/text_model.h"

namespace neural_network {
    const auto tokenizer = neural_network::load_tokenizers("tokenizer/tokenizer.json");
    const auto tokenizer_wrapper = neural_network::TokenizerWrapper(tokenizer, neural_network::TokenizerConfig());

    inline const neural_network::TokenizerWrapper &get_tokenizer() {
        return tokenizer_wrapper;
    }


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
        auto token_ids = get_tokenizer().encode(text);
        attention_mask_list_t attention_mask(token_ids.size(), 1);
        return embed(token_ids, attention_mask);
    }

    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts) {
        std::vector<token_id_list_t> token_ids = get_tokenizer().encode_batch(texts);
        std::vector<attention_mask_list_t> attention_mask_list;
        attention_mask_list.reserve(token_ids.size());
        for (const auto& ids : token_ids) {
            attention_mask_list_t mask(ids.size(), 1);
            attention_mask_list.push_back(std::move(mask));
        }
        return embed(token_ids, attention_mask_list);
    }

    emb_mat_t TextEmbeddingModel::embed(const token_id_list_t &token_ids, const attention_mask_list_t &attention_mask) {
        assert(token_ids.size() == attention_mask.size());

        const size_t seq_len = token_ids.size();
        const std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};

        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_ids(token_ids.cbegin(), token_ids.cend());
        std::vector<int64_t> masks(attention_mask.cbegin(), attention_mask.cend());

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
                                                     const std::vector<attention_mask_list_t> &attention_mask) {
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
            for (auto id : current_ids) {
                input_ids_flat.push_back(id);
            }
            input_ids_flat.insert(input_ids_flat.end(), max_seq_len - current_ids.size(), 0LL);

            const auto &current_mask = attention_mask[i];
            for (auto mask : current_mask) {
                masks_flat.push_back(mask);
            }
            masks_flat.insert(masks_flat.end(), max_seq_len - current_mask.size(), 0LL);
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

    std::vector<emb_mat_t> TextEmbeddingModel::embed(const std::vector<std::string> &texts) {
        std::vector<emb_mat_t> embeddings;
        embeddings.reserve(texts.size());
        for (const auto &text : texts) {
            embeddings.push_back(embed(text));
        }
        return embeddings;
    }
} // namespace neural_network