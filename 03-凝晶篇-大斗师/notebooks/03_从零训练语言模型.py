"""
╔══════════════════════════════════════════════════════════════════╗
║              《焚诀》第三卷 · 凝晶篇 — 配套丹方                   ║
║                                                                  ║
║         从零训练语言模型（Mini GPT）— 首次凝晶实录                  ║
║                                                                  ║
║  "万卷丹书不如一炉实炼。"                                         ║
║  "斗气由气态凝为液态，再化为晶体，方为大斗师之境。"                  ║
╚══════════════════════════════════════════════════════════════════╝

丹方说明：
    本丹方实现一个完整的小型 GPT 语言模型训练流程：
    1. 训练自定义 BPE Tokenizer（文字炼化）
    2. 定义 Mini GPT 模型（斗气凝聚阵）
    3. 训练循环（炼丹过程）
    4. 文本生成（斗气释放）

药材需求：
    pip install torch tokenizers datasets tqdm

丹炉要求：
    - 最低：CPU（训练缓慢但可行）
    - 推荐：单张 GPU（RTX 3060 及以上）

运行方式：
    python 03_从零训练语言模型.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# 第一阶段：文字炼化 — 训练 BPE Tokenizer
# ============================================================================
# "药材入炉之前，必先研磨成标准大小的粉末，方能均匀受热，充分反应。"

def prepare_training_corpus():
    """
    准备训练语料 — 采集药材
    
    此处使用内置示例语料进行演示。
    实际修炼时，应替换为大规模文本数据（中文维基、新闻语料等）。
    """
    # 示例语料：混合中英文文本（实际应用时替换为真实大规模语料）
    corpus = [
        # --- 技术文本 ---
        "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的层次化表示。",
        "卷积神经网络在图像识别领域取得了巨大成功，能够自动学习图像的特征。",
        "循环神经网络通过隐藏状态来处理序列数据，但存在梯度消失的问题。",
        "Transformer架构使用自注意力机制，能够并行处理序列中的所有位置。",
        "BERT通过掩码语言模型和下一句预测两个任务进行预训练。",
        "GPT使用自回归方式进行预训练，每次预测序列中的下一个词元。",
        "注意力机制让模型能够关注输入序列中最相关的部分。",
        "反向传播算法通过链式法则计算损失函数对每个参数的梯度。",
        "随机梯度下降是最基本的优化算法，每次用一个小批量数据更新参数。",
        "学习率是训练过程中最重要的超参数之一，过大会导致发散，过小会收敛缓慢。",
        "批归一化通过标准化每层的输入来加速训练并提供正则化效果。",
        "残差连接让梯度能够直接流过网络，解决了深层网络训练困难的问题。",
        "Dropout是一种正则化技术，在训练时随机丢弃部分神经元以防止过拟合。",
        "词嵌入将离散的词语映射到连续的向量空间中，使得语义相似的词距离更近。",
        "位置编码为Transformer提供序列中词语的位置信息。",
        "预训练模型通过在大规模无标注数据上学习通用的语言表示。",
        "微调是在预训练模型的基础上，使用少量标注数据适应特定任务。",
        "分词器将原始文本切分成模型能够处理的词元序列。",
        "BPE算法从字符开始，不断合并最频繁的相邻对来构建词汇表。",
        "交叉熵损失函数衡量模型预测分布与真实分布之间的差异。",
        # --- 修炼叙事（增加语料多样性） ---
        "修炼之道在于循序渐进，根基不牢必遭反噬。",
        "吞噬异火需要强大的精神力来压制火焰的暴躁之气。",
        "炼丹师需要精准控制火候，温度过高则药材焚毁，温度过低则药效不显。",
        "斗气从气态凝为液态是突破大斗师的关键一步。",
        "每一次成功的炼丹都会让修炼者对火候的把控更加精准。",
        "三十年河东三十年河西莫欺少年穷，这是修炼者的座右铭。",
        "功法的等级决定了修炼者能够达到的上限。",
        "丹方记录了炼制每种丹药所需的药材配比和火候控制。",
        "药材的品质直接影响炼出丹药的效果和成功率。",
        "灵气充沛的地方更适合修炼，就如同算力充足的服务器更适合训练模型。",
    ]

    # 扩展语料量（实际训练需要更多数据）
    expanded_corpus = corpus * 100  # 重复扩展以增加数据量

    corpus_path = "/tmp/flamesutra_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in expanded_corpus:
            f.write(line + "\n")

    print(f"[药材采集完毕] 语料已写入 {corpus_path}")
    print(f"  总行数：{len(expanded_corpus)}")
    print(f"  总字符数：{sum(len(line) for line in expanded_corpus)}")

    return corpus_path


def train_tokenizer(corpus_path, vocab_size=2048, save_path="/tmp/flamesutra_tokenizer.json"):
    """
    训练 BPE Tokenizer — 文字炼化
    
    "从散碎的灵气微粒（单个字符）出发，
     将最常共同出现的微粒凝聚在一起（合并高频对），
     反复凝聚，直到形成一颗颗标准大小的灵气结晶（子词 token）。"
    
    参数：
        corpus_path: 语料文件路径
        vocab_size: 词汇表大小（灵气结晶的种类数）
        save_path: Tokenizer 保存路径
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    print("\n" + "=" * 60)
    print("  文字炼化开始 — 训练 BPE Tokenizer")
    print("=" * 60)

    # 初始化 BPE 模型 — 准备丹炉
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # 预分词器：按空格初步切分 — 药材初步分类
    tokenizer.pre_tokenizer = Whitespace()

    # 训练器配置 — 丹方参数
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,        # 至少出现2次的对才合并
        special_tokens=[
            "<pad>",    # ID 0: 填充标记 — 阵法空位
            "<unk>",    # ID 1: 未知标记 — 未知药材
            "<bos>",    # ID 2: 序列开始 — 阵法起点
            "<eos>",    # ID 3: 序列结束 — 阵法终点
        ],
        show_progress=True,
    )

    # 开始训练 — 开炉炼化
    print(f"  词汇表目标大小：{vocab_size}")
    tokenizer.train(files=[corpus_path], trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  实际词汇表大小：{actual_vocab_size}")

    # 测试分词效果 — 验证炼化成果
    test_sentences = [
        "深度学习改变世界",
        "Transformer架构非常强大",
        "修炼之道在于循序渐进",
    ]

    print("\n  分词效果测试：")
    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        print(f"    原文：{sent}")
        print(f"    Token：{encoded.tokens}")
        print(f"    ID：{encoded.ids}")
        print()

    # 保存 Tokenizer — 保存丹方
    tokenizer.save(save_path)
    print(f"  Tokenizer 已保存至：{save_path}")

    return tokenizer, actual_vocab_size


# ============================================================================
# 第二阶段：斗气凝聚阵 — Mini GPT 模型定义
# ============================================================================
# "以因果法则为核心，构建一座小型斗气凝聚阵。
#  虽然规模不大，但五脏俱全，是修炼者的第一颗斗气结晶。"


class MiniGPTConfig:
    """
    模型配置 — 丹方参数表
    
    记录了凝聚阵的每一个关键参数。
    修炼者可根据自身丹炉（硬件）条件调整。
    """
    def __init__(
        self,
        vocab_size=2048,
        max_seq_len=128,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    ):
        self.vocab_size = vocab_size        # 词汇表大小（药材种类数）
        self.max_seq_len = max_seq_len      # 最大序列长度（丹炉容量）
        self.n_layers = n_layers            # Transformer 层数（凝聚阵层数）
        self.n_heads = n_heads              # 注意力头数（感知通道数）
        self.d_model = d_model              # 隐藏维度（斗气密度）
        self.d_ff = d_ff                    # FFN 中间维度（斗气扩展空间）
        self.dropout = dropout              # Dropout 比率（灵气泄漏率）
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 校验：d_model 必须能被 n_heads 整除
        assert d_model % n_heads == 0, \
            f"斗气密度 d_model={d_model} 必须能被感知通道数 n_heads={n_heads} 整除！"
        self.head_dim = d_model // n_heads


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层 — 命运之眼
    
    "只知过去，不窥未来。以已知推未知，以过往定将来。"
    
    通过下三角掩码确保每个位置只能看到它之前（含自身）的 token。
    这是 GPT 自回归生成的核心机制。
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Q, K, V 投影 — 将斗气分解为三种感知维度
        # 合并为一个线性层，提高计算效率
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # 输出投影 — 将多头感知结果融合
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Attention Dropout — 灵气波动
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码 — 命运之链只能向前延伸
        # 注册为 buffer（不参与梯度计算，但会随模型保存/加载）
        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len)
        ).view(1, 1, config.max_seq_len, config.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        """
        前向传播 — 命运推演
        
        x: (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape

        # 计算 Q, K, V — 将斗气分解为查询、钥匙、价值三个维度
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv.unbind(0)  # 各为 (B, n_heads, T, head_dim)

        # 计算注意力分数 — 感知万物之间的关联
        scale = math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # 应用因果掩码 — 遮蔽未来的信息
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float('-inf')
        )

        # Softmax 归一化 — 将注意力分布归一
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权聚合 — 按关联强度融合信息
        out = attn_weights @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, d_model)

        # 输出投影 — 融合多头感知
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    前馈网络（MLP）— 斗气精炼室
    
    "注意力层负责感知万物关联，前馈层负责对感知到的信息进行深度加工。"
    
    结构：Linear → GELU → Linear → Dropout
    中间维度通常为 d_model 的 4 倍（斗气先扩张再压缩，提纯其精华）
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """先扩张、激活、再压缩 — 斗气提纯"""
        x = self.fc1(x)         # 扩张：d_model → d_ff
        x = self.act(x)         # 激活：GELU 非线性变换
        x = self.fc2(x)         # 压缩：d_ff → d_model
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 解码块 — 凝聚阵的基本单元
    
    每个 Block 包含：
    1. 层归一化 → 因果自注意力 → 残差连接
    2. 层归一化 → 前馈网络 → 残差连接
    
    采用 Pre-Norm 结构（GPT-2 风格），训练更稳定。
    "先净化斗气（LayerNorm），再进行感知/加工，最后与原始斗气融合（残差连接）。"
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)   # 第一次净化
        self.attn = CausalSelfAttention(config)    # 因果感知
        self.ln2 = nn.LayerNorm(config.d_model)    # 第二次净化
        self.ffn = FeedForward(config)              # 斗气精炼

    def forward(self, x):
        # Pre-Norm + Residual
        x = x + self.attn(self.ln1(x))  # 感知 + 残差
        x = x + self.ffn(self.ln2(x))   # 精炼 + 残差
        return x


class MiniGPT(nn.Module):
    """
    Mini GPT 语言模型 — 斗气凝聚阵
    
    "此阵虽小，五脏俱全。
     Token Embedding 为药材入口，
     Position Embedding 为位置标记，
     Transformer Block 为核心炼化区，
     输出层为成品出口。
     
     修炼者的第一颗斗气结晶，由此而生。"
    
    参数规模约 ~10M（根据配置可调），
    可在单张消费级 GPU 上完成训练。
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # ---- Embedding 层 ---- #
        # Token Embedding：将 token ID 映射为稠密向量（药材编号 → 灵气形态）
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Position Embedding：编码位置信息（可学习参数）
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Embedding Dropout：初始灵气波动
        self.emb_dropout = nn.Dropout(config.dropout)

        # ---- Transformer 层 ---- #
        # 多层 Transformer Block 堆叠（凝聚阵核心）
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 最终层归一化 — 最后一次净化
        self.ln_final = nn.LayerNorm(config.d_model)

        # ---- 输出层 ---- #
        # 将隐藏状态映射回词汇表大小（预测下一个 token 的概率分布）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享（Weight Tying）：
        # Embedding 层和输出层共享权重矩阵
        # 这是一个经典技巧，能减少参数量并提升性能
        # "入口与出口共用同一套灵气编码体系"
        self.token_emb.weight = self.lm_head.weight

        # 初始化参数 — 丹炉校准
        self.apply(self._init_weights)

        # 统计并显示参数量
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_emb = n_params - self.pos_emb.weight.numel()
        print(f"\n[凝聚阵构建完成]")
        print(f"  总参数量：{n_params:,} ({n_params / 1e6:.2f}M)")
        print(f"  非Embedding参数：{n_params_no_emb:,} ({n_params_no_emb / 1e6:.2f}M)")
        print(f"  层数：{config.n_layers}，注意力头：{config.n_heads}")
        print(f"  隐藏维度：{config.d_model}，FFN维度：{config.d_ff}")

    def _init_weights(self, module):
        """
        参数初始化 — 丹炉校准
        
        使用较小的标准差初始化，防止初期斗气紊乱（梯度爆炸）。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        """
        前向传播 — 斗气在凝聚阵中的流动
        
        参数：
            input_ids: (batch_size, seq_len) — 输入 token ID 序列
            targets: (batch_size, seq_len) — 目标 token ID 序列（训练时使用）
        
        返回：
            logits: (batch_size, seq_len, vocab_size) — 每个位置的预测概率分布
            loss: 标量 — 交叉熵损失（仅在提供 targets 时计算）
        """
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.config.max_seq_len, \
            f"序列长度 {T} 超过丹炉容量 {self.config.max_seq_len}！"

        # Token Embedding + Position Embedding
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        positions = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.pos_emb(positions)     # (T, d_model)

        # 融合 Token 和 Position 信息
        x = self.emb_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # 通过 Transformer Block 堆叠（凝聚阵核心炼化）
        for block in self.blocks:
            x = block(x)

        # 最终层归一化
        x = self.ln_final(x)  # (B, T, d_model)

        # 输出 logits — 预测每个位置的下一个 token
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（仅在训练时）
        loss = None
        if targets is not None:
            # 将 logits 和 targets 展平后计算交叉熵
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
                ignore_index=self.config.pad_token_id,  # 忽略 padding 位置
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9):
        """
        自回归生成 — 命运推演术
        
        "以已知推未知，一字一句，如命运之链，前因注定后果。"
        
        参数：
            input_ids: (1, seq_len) — 起始 token 序列（prompt）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（控制生成的随机性/创造性）
            top_k: Top-K 采样（只从概率最高的 K 个候选中选择）
            top_p: Nucleus 采样（从概率之和达到 p 的最小集合中选择）
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到最大序列长度（丹炉容量限制）
            idx_crop = input_ids[:, -self.config.max_seq_len:]

            # 前向传播 — 感知当前状态
            logits, _ = self(idx_crop)

            # 取最后一个位置的 logits（预测下一个 token）
            logits = logits[:, -1, :]  # (B, vocab_size)

            # 温度缩放 — 控制火候
            if temperature != 1.0:
                logits = logits / temperature

            # Top-K 过滤 — 排除不合理的候选
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_values, _ = torch.topk(logits, top_k_val)
                min_topk = kth_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_topk,
                    torch.full_like(logits, float('-inf')),
                    logits
                )

            # Top-P (Nucleus) 过滤 — 动态调整候选范围
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的 token（确保至少有一个候选）
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 采样 — 从概率分布中选择下一个 token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 遇到 EOS 则停止
            if next_token.item() == self.config.eos_token_id:
                break

            # 拼接新 token
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================================
# 第三阶段：药材预处理 — 数据集构建
# ============================================================================
# "药材已采集，分词器已炼成，现在需要将药材按丹方要求切片包装。"


class TextDataset(Dataset):
    """
    文本数据集 — 标准化后的药材库
    
    将文本语料转化为模型训练所需的 (input, target) 对。
    对于语言模型：input 是 token 序列，target 是右移一位的 token 序列。
    
    例如：
        原始序列：  [BOS, 深, 度, 学, 习, EOS]
        输入 (x)：  [BOS, 深, 度, 学, 习]
        目标 (y)：  [深, 度, 学, 习, EOS]
    """

    def __init__(self, corpus_path, tokenizer, max_seq_len=128):
        """
        参数：
            corpus_path: 语料文件路径
            tokenizer: 训练好的 tokenizer
            max_seq_len: 最大序列长度
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        # 特殊 token ID
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        # 读取并编码语料 — 药材切片
        print("\n[药材预处理] 编码语料中...")
        self.samples = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 编码文本
                encoded = tokenizer.encode(line)
                token_ids = encoded.ids

                # 添加 BOS 和 EOS
                token_ids = [self.bos_id] + token_ids + [self.eos_id]

                # 截断到最大长度 + 1（因为 input 和 target 各需要 max_seq_len）
                if len(token_ids) > max_seq_len + 1:
                    token_ids = token_ids[:max_seq_len + 1]

                # 只保留长度 >= 3 的样本（至少 BOS + 1 token + EOS）
                if len(token_ids) >= 3:
                    self.samples.append(token_ids)

        print(f"  有效样本数：{len(self.samples)}")
        print(f"  平均序列长度：{sum(len(s) for s in self.samples) / len(self.samples):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx]

        # 分离 input 和 target
        # input: 所有 token 除了最后一个
        # target: 所有 token 除了第一个（右移一位）
        x = token_ids[:-1]
        y = token_ids[1:]

        # Padding 到固定长度
        pad_len = self.max_seq_len - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [self.pad_id] * pad_len

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "targets": torch.tensor(y, dtype=torch.long),
        }


# ============================================================================
# 第四阶段：开炉炼丹 — 训练循环
# ============================================================================
# "万事俱备，只欠东风。点燃异火，开始凝晶！"


def train_model(model, train_loader, val_loader, config, tokenizer, device,
                max_steps=3000, lr=3e-4, warmup_steps=100, log_interval=50,
                val_interval=300, generate_interval=500):
    """
    训练循环 — 炼丹主过程
    
    "控制火候（学习率），持续投入药材（数据），
     监控丹炉状态（loss），防范反噬（NaN/梯度爆炸）。"
    """
    print("\n" + "=" * 60)
    print("  开炉炼丹 — 训练开始")
    print("=" * 60)
    print(f"  设备：{device}")
    print(f"  最大步数：{max_steps}")
    print(f"  学习率：{lr}")
    print(f"  Warmup 步数：{warmup_steps}")

    # 优化器 — 火候控制器
    # AdamW 是训练 Transformer 的标配
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),       # Adam 动量参数
        weight_decay=0.1,         # 权重衰减（正则化）
        eps=1e-8,
    )

    # 学习率调度函数 — 火候曲线
    def get_lr(step):
        """
        Warmup + Cosine Decay 学习率调度
        
        "起初文火慢热（warmup），随后达到最高火力，
         之后渐渐减小火力精细调控（cosine decay）。"
        """
        if step < warmup_steps:
            # 线性 warmup
            return lr * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 训练状态追踪
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    model.train()
    train_iter = iter(train_loader)
    step = 0

    progress_bar = tqdm(total=max_steps, desc="炼丹进度")

    while step < max_steps:
        # 获取下一批数据 — 投入药材
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        # 更新学习率 — 调整火候
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 前向传播 — 丹炉运转
        logits, loss = model(input_ids, targets)

        # 检测反噬（NaN）
        if torch.isnan(loss):
            print("\n[反噬警告] Loss 变为 NaN！训练中止。")
            print("  可能原因：学习率过大、数据异常、梯度爆炸")
            print("  建议：降低学习率，检查数据，增加梯度裁剪强度")
            break

        # 反向传播 — 计算梯度
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 — 防反噬护盾
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 参数更新 — 斗气精进
        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 更新进度条
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}",
            "grad": f"{grad_norm:.2f}",
        })

        # ---- 定期日志 ---- #
        if step > 0 and step % log_interval == 0:
            avg_loss = sum(train_losses[-log_interval:]) / log_interval
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            tqdm.write(
                f"  [Step {step:5d}/{max_steps}] "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Grad Norm: {grad_norm:.2f} | "
                f"Speed: {steps_per_sec:.1f} steps/s"
            )

        # ---- 定期验证 ---- #
        if step > 0 and step % val_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append((step, val_loss))
            perplexity = math.exp(min(val_loss, 20))  # 防止溢出

            tqdm.write(
                f"  [验证] Step {step} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Perplexity: {perplexity:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最优模型 — 保存最纯净的结晶
                save_path = "/tmp/flamesutra_mini_gpt_best.pt"
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"  [保存] 最优模型已保存至 {save_path}")

            model.train()

        # ---- 定期生成样本 ---- #
        if step > 0 and step % generate_interval == 0:
            model.eval()
            sample_text = generate_sample(model, tokenizer, device)
            tqdm.write(f"  [生成样本] {sample_text[:150]}...")
            model.train()

        step += 1

    progress_bar.close()

    # 训练结束统计
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  炼丹完成！")
    print("=" * 60)
    print(f"  总训练时间：{total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
    print(f"  最终训练 Loss：{train_losses[-1]:.4f}")
    if val_losses:
        print(f"  最优验证 Loss：{best_val_loss:.4f}")
        print(f"  最优 Perplexity：{math.exp(min(best_val_loss, 20)):.2f}")

    return train_losses, val_losses


@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    验证评估 — 检验结晶纯度
    
    计算验证集上的平均损失。
    """
    model.eval()
    total_loss = 0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        _, loss = model(input_ids, targets)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def generate_sample(model, tokenizer, device, prompt_text=None, max_tokens=80):
    """
    生成文本样本 — 释放斗气结晶之力
    
    参数：
        prompt_text: 起始文本（为 None 则从 BOS 开始）
        max_tokens: 最大生成 token 数
    """
    model.eval()

    if prompt_text:
        # 编码 prompt
        encoded = tokenizer.encode(prompt_text)
        input_ids = [tokenizer.token_to_id("<bos>")] + encoded.ids
    else:
        input_ids = [tokenizer.token_to_id("<bos>")]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 生成
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
    )

    # 解码
    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids)

    return text


# ============================================================================
# 第五阶段：主修炼流程 — 串联一切
# ============================================================================


def main():
    """
    主函数 — 完整的凝晶修炼流程
    
    "从药材采集到斗气结晶，一气呵成。"
    """
    print("╔══════════════════════════════════════════════════╗")
    print("║   《焚诀》第三卷 · 凝晶篇 — 首次凝晶实录         ║")
    print("║                                                  ║")
    print("║   目标：从零训练一个 Mini GPT 语言模型             ║")
    print("╚══════════════════════════════════════════════════╝")

    # ---- 设备检测 ---- #
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n[异火检测] 发现 GPU：{gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n[异火检测] 发现 Apple MPS 加速")
    else:
        device = torch.device("cpu")
        print("\n[异火检测] 未发现 GPU，使用 CPU（训练将较慢）")

    # ====================
    # 阶段一：采集药材
    # ====================
    print("\n" + "━" * 50)
    print("  阶段一：采集药材（准备语料）")
    print("━" * 50)
    corpus_path = prepare_training_corpus()

    # ====================
    # 阶段二：文字炼化
    # ====================
    print("\n" + "━" * 50)
    print("  阶段二：文字炼化（训练 Tokenizer）")
    print("━" * 50)
    vocab_size = 2048
    tokenizer, actual_vocab_size = train_tokenizer(
        corpus_path,
        vocab_size=vocab_size,
    )

    # ====================
    # 阶段三：构建凝聚阵
    # ====================
    print("\n" + "━" * 50)
    print("  阶段三：构建凝聚阵（定义模型）")
    print("━" * 50)

    config = MiniGPTConfig(
        vocab_size=actual_vocab_size,
        max_seq_len=128,
        n_layers=4,             # 4 层 Transformer（入门级凝聚阵）
        n_heads=4,              # 4 个注意力头
        d_model=256,            # 隐藏维度 256
        d_ff=1024,              # FFN 维度 1024 (4x d_model)
        dropout=0.1,
    )

    model = MiniGPT(config).to(device)

    # ====================
    # 阶段四：准备药材
    # ====================
    print("\n" + "━" * 50)
    print("  阶段四：药材预处理（构建数据集）")
    print("━" * 50)

    # 创建数据集
    full_dataset = TextDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
    )

    # 划分训练集和验证集（90% / 10%）
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"  训练集大小：{train_size}")
    print(f"  验证集大小：{val_size}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # ====================
    # 阶段五：开炉炼丹
    # ====================
    print("\n" + "━" * 50)
    print("  阶段五：开炉炼丹（训练模型）")
    print("━" * 50)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer=tokenizer,
        device=device,
        max_steps=2000,         # 2000 步（示例用，实际可增加）
        lr=3e-4,                # 学习率
        warmup_steps=100,       # Warmup 步数
        log_interval=50,        # 每 50 步打印日志
        val_interval=200,       # 每 200 步验证
        generate_interval=400,  # 每 400 步生成样本
    )

    # ====================
    # 阶段六：结晶释放
    # ====================
    print("\n" + "━" * 50)
    print("  阶段六：结晶释放（文本生成测试）")
    print("━" * 50)

    # 加载最优模型
    best_model_path = "/tmp/flamesutra_mini_gpt_best.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("  已加载最优模型")

    model.eval()

    # 测试生成
    test_prompts = [
        None,              # 从头生成
        "深度学习",         # 技术 prompt
        "修炼",            # 修炼 prompt
        "Transformer",     # 英文 prompt
    ]

    for prompt in test_prompts:
        display_prompt = prompt if prompt else "<BOS>"
        print(f"\n  Prompt: 「{display_prompt}」")
        for i in range(3):  # 每个 prompt 生成 3 个样本
            text = generate_sample(
                model, tokenizer, device,
                prompt_text=prompt,
                max_tokens=60,
            )
            print(f"    [{i+1}] {text[:120]}")

    # ====================
    # 最终评估
    # ====================
    print("\n" + "━" * 50)
    print("  最终评估 — 检验结晶纯度")
    print("━" * 50)

    final_val_loss = evaluate(model, val_loader, device)
    final_perplexity = math.exp(min(final_val_loss, 20))

    print(f"  最终验证 Loss：{final_val_loss:.4f}")
    print(f"  最终 Perplexity：{final_perplexity:.2f}")
    print(f"  （Perplexity 表示模型平均在 {final_perplexity:.0f} 个候选词中犹豫）")

    # ---- 修炼总结 ---- #
    print("\n" + "=" * 60)
    print("  凝晶修炼完成！")
    print("=" * 60)
    print("""
    你已经完成了大斗师境界的核心修炼：
    
    [已掌握] 文字炼化 — BPE Tokenizer 训练
    [已掌握] 斗气凝聚阵 — Mini GPT 模型构建
    [已掌握] 炼丹全流程 — 从数据到生成的完整管线
    [已掌握] 火候控制 — 学习率调度、梯度裁剪
    [已掌握] 结晶检验 — Perplexity 评估

    注意：本丹方使用的是极小规模的语料和模型，
    生成质量有限。要获得更好的效果，需要：
    1. 更多的药材（更大的语料库）
    2. 更大的凝聚阵（更多层、更大维度）
    3. 更强的异火（更好的 GPU）
    4. 更长的炼制时间（更多训练步数）

    "斗气固化，结晶已成。虽是初生之晶，亦蕴含无限可能。"
    """)


# ============================================================================
# 入口 — 修炼开始
# ============================================================================

if __name__ == "__main__":
    main()
