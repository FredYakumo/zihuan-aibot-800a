from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
from utils.logging_config import logger

def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Silicon GPU")
        device = torch.device('mps')
    return device

class CosineSimilarityModel(nn.Module):
    def forward(self, a: torch.Tensor, B: torch.Tensor):
        assert a.dim() == 2 and a.shape[0] == 1, "a must be a [1, D] vector"
        assert B.dim() == 2, "B must be a [N, D] matrix"
        
        # avoid div 0
        eps = 1e-8
        
        a_norm = a.norm(dim=1, keepdim=True)  # [1,1]
        
        B_norm = B.norm(dim=1, keepdim=True)  # [N,1]
        B_norm = torch.max(B_norm, torch.tensor(eps))
        
        # a_norm = torch.sqrt(torch.sum(a ** 2, dim=1, keepdim=True))   # shape: [1, 1]
        # B_norm = torch.sqrt(torch.sum(B ** 2, dim=1, keepdim=True))   # shape: [N, 1]
        # 避免除以 0
        B_norm = B_norm.clamp(min=eps)
        
        dot_product = torch.matmul(a, B.t())  # [1,N]
        
        similarity = dot_product / (a_norm * B_norm.t())
        
        # [N]
        return similarity.squeeze(0)

class ReplyIntentClassifierModel(nn.Module):
    """Neural network model for classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze()
    
class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)  # This should be your Hugging Face model that accepts tokenized inputs.

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     # Assume you want to mean-pool the output embeddings.
    #     embeddings = outputs.last_hidden_state.mean(dim=1)
    #     return embeddings
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        # Assume you want to mean-pool the output embeddings.
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class TextEmbedder:
    """Generate text embeddings using a pre-trained model."""
    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', device=torch.device('cpu')):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        print(f"Max position embedding: {config.max_position_embeddings}")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
    def embed(self, texts):
        """Generate text embedding vectors."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    
def load_reply_intent_classifier_model(model_path):
    """Load the trained model"""
    embedder = TextEmbedder()
    model = ReplyIntentClassifierModel(embedder.model.config.hidden_size)
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, embedder

import torch.nn.functional as F

class ChatSessionToQueryStrModel(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 rep_dim=2048,
                 num_functions=6,
                 vocab_size=10000,
                 embedding_dim=256,
                 max_gen_len=20):
        """
        参数说明：
        - input_dim: 每条对话消息的嵌入维度（1024）
        - rep_dim: 对话融合表示的投影维度
        - num_functions: 功能类别数，包括：
            "聊天", "查询用户头像", "查询用户聊天记录", "查询用户资料", "查询知识库", "非法"
        - vocab_size: token 嵌入时的词汇表大小
        - embedding_dim: token 的嵌入维度（用于解码器以及额外文本输入的处理）
        - max_gen_len: 查询字符串生成时的最大 token 数
        """
        super(ChatSessionToQueryStrModel, self).__init__()
        
        # ---------------- 对话信息处理 ----------------
        # 处理对话中最后一条消息和整个对话的均值（整体上下文信息）
        self.linear_last = nn.Linear(input_dim, input_dim)
        self.linear_hist = nn.Linear(input_dim, input_dim)
        self.combined_proj = nn.Linear(2 * input_dim, rep_dim)
        
        # ---------------- 附加信息处理 ----------------
        # 利用共享的 token 嵌入层对额外文本信息（用户名与 QQ 号）进行处理，
        # 输入为 token 序列，经过平均池化后形状为 (1, embedding_dim)
        # 拼接四个信息后为 (1, 4*embedding_dim)，再通过线性层映射到 rep_dim 空间
        self.additional_proj = nn.Linear(4 * embedding_dim, rep_dim)
        
        # 将对话表示和附加信息表示拼接后，再投影回 rep_dim（作为最终融合表示）
        self.final_proj = nn.Linear(2 * rep_dim, rep_dim)
        
        # ---------------- 多任务输出 ----------------
        # 功能分类 Head（MLP 输出独热向量）
        self.function_classifier = nn.Sequential(
            nn.Linear(rep_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_functions)
        )
        # 时效性判断 Head（回归输出 0～1 之间的数值）
        self.time_regressor = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # ---------------- 查询文本生成 ----------------
        # 解码器：基于 GRU，利用最终融合的表征初始化隐藏状态
        self.init_hidden = nn.Linear(rep_dim, rep_dim)
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru_decoder = nn.GRU(input_size=embedding_dim, hidden_size=rep_dim, 
                                  num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(rep_dim, vocab_size)
        self.max_gen_len = max_gen_len
        self.num_functions = num_functions

    def forward(self, 
                x, 
                user_name_tokens, 
                user_qq_tokens, 
                target_name_tokens, 
                target_qq_tokens,
                decoder_input_tokens=None):
        """
        参数：
        - x: 对话嵌入，形状可以为
             (N, 1024) --> 表示一个对话中N条消息，
             或 (B, N, 1024) --> 表示多个对话；最后融合成一组信息（输出只有一个结果）。
        - user_name_tokens: 用户名字的 token 序列，形状 (1, L1)
        - user_qq_tokens: 用户 QQ 号的 token 序列，形状 (1, L2)
        - target_name_tokens: bot“紫幻”名字的 token 序列，形状 (1, L3)
        - target_qq_tokens: bot“紫幻” QQ 号的 token 序列，形状 (1, L4)
        - decoder_input_tokens: teacher forcing 时的目标 token 序列,
            形状 (1, seq_len)；若为 None，则进行推理模式（贪婪解码）
            
        返回一个三元组：
          - function_onehot: 形状 (1, num_functions) 的独热向量
          - time_score: 形状 (1, 1) 的时效性分数
          - query_text: 若 teacher forcing 下返回 logits（形状 (1, seq_len, vocab_size)），
                        否则返回生成的 token 序列（形状 (1, max_gen_len)）
        """
        # --------------- 处理对话记录 -----------------
        # 无论输入维度如何，最终统一当成一组对话消息来处理
        if x.dim() == 3:
            B, N, D = x.size()
            x = x.view(B * N, D)
        # 此时 x 的形状为 (N, 1024)
        # 取最后一条消息（最下方）作为核心
        last_msg = x[-1].unsqueeze(0)          # (1, 1024)
        # 对整个对话做平均获得整体上下文表示
        hist_avg = torch.mean(x, dim=0, keepdim=True)  # (1, 1024)
        last_feat = F.relu(self.linear_last(last_msg))
        hist_feat = F.relu(self.linear_hist(hist_avg))
        # 拼接最后一条与全局平均信息
        combined = torch.cat([last_feat, hist_feat], dim=1)  # (1, 2*input_dim)
        rep_conv = F.relu(self.combined_proj(combined))       # (1, rep_dim)
        
        # --------------- 处理附加信息 -----------------
        # 对于用户的名字和 QQ 号以及“紫幻”的名字和 QQ 号，
        # 采用共享的 token 嵌入层，并对 token 序列做平均池化，得到每个输入的固定表示
        user_name_embed = torch.mean(self.token_embeddings(user_name_tokens), dim=1)   # (1, embedding_dim)
        user_qq_embed   = torch.mean(self.token_embeddings(user_qq_tokens), dim=1)     # (1, embedding_dim)
        target_name_embed = torch.mean(self.token_embeddings(target_name_tokens), dim=1)  # (1, embedding_dim)
        target_qq_embed   = torch.mean(self.token_embeddings(target_qq_tokens), dim=1)    # (1, embedding_dim)
        # 拼接四个表示，形状 (1, 4 * embedding_dim)
        additional_features = torch.cat([
            user_name_embed, 
            user_qq_embed, 
            target_name_embed,
            target_qq_embed
        ], dim=1)
        additional_feat = F.relu(self.additional_proj(additional_features))  # (1, rep_dim)
        
        # --------------- 最终融合表示 -----------------
        # 将对话表示和附加表示拼接后，再投影回 rep_dim
        final_rep = torch.cat([rep_conv, additional_feat], dim=1)  # (1, 2*rep_dim)
        final_rep = F.relu(self.final_proj(final_rep))             # (1, rep_dim)
        
        # --------------- 功能分类与时效性回归 -----------------
        func_logits = self.function_classifier(final_rep)           # (1, num_functions)
        func_pred = torch.argmax(func_logits, dim=1)                  # (1,)
        function_onehot = F.one_hot(func_pred, num_classes=self.num_functions).float()
        time_score = self.time_regressor(final_rep)                   # (1, 1)
        
        # --------------- 查询文本生成 -----------------
        # 利用最终融合表示作为解码器初始状态（GRU 使用 first hidden state）
        hidden = self.init_hidden(final_rep).unsqueeze(0)  # (1, 1, rep_dim)
        
        if decoder_input_tokens is not None:
            # Teacher Forcing 模式下，使用目标 token 序列生成所有时间步的 logits
            token_embed = self.token_embeddings(decoder_input_tokens)  # (1, seq_len, embedding_dim)
            dec_out, _ = self.gru_decoder(token_embed, hidden)         # (1, seq_len, rep_dim)
            query_text_logits = self.output_layer(dec_out)             # (1, seq_len, vocab_size)
        else:
            # 推理模式（采用贪婪解码）
            # 假定 <start> token 的 id 是 1
            start_token = torch.ones(1, 1, dtype=torch.long, device=x.device)  # (1, 1)
            dec_input = self.token_embeddings(start_token)                     # (1, 1, embedding_dim)
            outputs = []
            for _ in range(self.max_gen_len):
                dec_out, hidden = self.gru_decoder(dec_input, hidden)  # dec_out: (1, 1, rep_dim)
                logits = self.output_layer(dec_out.squeeze(1))         # (1, vocab_size)
                predicted = torch.argmax(logits, dim=1, keepdim=True)    # (1, 1)
                outputs.append(predicted)
                dec_input = self.token_embeddings(predicted)           # (1, 1, embedding_dim)
            query_text_tokens = torch.cat(outputs, dim=1)               # (1, max_gen_len)
            query_text_logits = query_text_tokens
        
        return function_onehot, time_score, query_text_logits