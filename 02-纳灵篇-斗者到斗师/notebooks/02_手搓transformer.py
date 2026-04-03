"""
=============================================================
  《焚诀》第二卷 · 天道法则 · 手搓 Transformer 完整实录

  功法名称：天道法则（Transformer）
  修炼目标：从零实现完整 Transformer，在序列复制任务上验证
  所需异火：CPU 即可（小规模验证），GPU 可加速

  术语对照：
    异火 = GPU/算力        药材 = 数据集
    丹方 = 训练脚本        炼丹 = 模型训练
    反噬 = Loss NaN/OOM    斗气 = 模型参数
    斗技 = 算法/架构

  运行方式：
    python 02_手搓transformer.py

  修炼者须知：
    本丹方包含完整的 Transformer 实现，每一行都有注释。
    建议逐段阅读、逐段理解，切忌囫囵吞枣。
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ╔══════════════════════════════════════════════════════════════╗
# ║          第一阵法：位置编码 —— 时空感知法阵                    ║
# ║  Transformer 没有循环结构，不知道 token 的先后顺序。            ║
# ║  位置编码就是为每个位置注入独特的"灵力指纹"，               ║
# ║  让模型能够区分"第一个字"和"第十个字"。                    ║
# ╚══════════════════════════════════════════════════════════════╝

class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Sinusoidal Positional Encoding）

    使用不同频率的正弦和余弦函数为每个位置生成唯一的编码向量。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    灵力学原理：
        低维度（小 i）→ 高频波动 → 感知局部位置差异（相邻字的区别）
        高维度（大 i）→ 低频波动 → 感知全局位置结构（段落级别的位置）

    就像不同频率的灵力共振叠加，在每个位置形成独一无二的灵力场。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        参数：
            d_model:  斗气维度（模型隐藏层维度）
            max_len:  最大序列长度（预计算的位置编码数量）
            dropout:  灵力散逸率（Dropout 比例）
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ---- 预计算位置编码矩阵 ----
        # 此矩阵在训练过程中不更新（register_buffer），属于固定法阵
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        # position: 每个位置的索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # → (max_len, 1)

        # div_term: 频率因子 10000^(2i/d_model)
        # 用 exp(log) 形式计算，避免大数幂运算的数值不稳定（防止反噬）
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        # → (d_model/2,)

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos * freq)
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos * freq)

        # 添加 batch 维度: (max_len, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer: 不参与梯度计算，但会随模型保存/加载/移动设备
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码注入输入张量。

        输入: x, shape = (batch, seq_len, d_model)
        输出: shape = (batch, seq_len, d_model)  —— 维度不变，但带上了位置信息
        """
        # self.pe[:, :x.size(1), :] → 取前 seq_len 个位置的编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ╔══════════════════════════════════════════════════════════════╗
# ║        第二阵法：缩放点积注意力 —— 天道法则核心心法            ║
# ║  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V          ║
# ║  这个公式是整个现代 AI 的基石，务必彻底理解。                ║
# ╚══════════════════════════════════════════════════════════════╝

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple:
    """
    缩放点积注意力 —— Transformer 的灵魂所在。

    修炼心法解读：
        每个 token（修炼者）通过 Q 提出问题，K 展示自己的信息标签，
        V 提供实际内容。注意力机制让每个 token 都能直接关注到序列中
        任何其他 token —— 无论距离多远。

    参数：
        Q: (batch, heads, seq_len_q, d_k)   —— 查询矩阵（我需要什么？）
        K: (batch, heads, seq_len_k, d_k)   —— 键矩阵（我有什么？）
        V: (batch, heads, seq_len_k, d_v)   —— 值矩阵（我能提供什么？）
        mask: 可选遮罩，用于因果推演阵（Decoder 的因果遮罩）
        dropout: 可选 Dropout 层

    返回：
        output:  (batch, heads, seq_len_q, d_v)    —— 注意力输出
        weights: (batch, heads, seq_len_q, seq_len_k)  —— 注意力权重分布

    维度追踪（以 Q, K, V 形状为 (B, h, S, d_k) 为例）：
        Q @ K^T     → (B, h, S, d_k) @ (B, h, d_k, S) = (B, h, S, S)
        softmax     → (B, h, S, S)
        @ V         → (B, h, S, S) @ (B, h, S, d_k)   = (B, h, S, d_k)
    """
    d_k = Q.size(-1)  # 每个头的维度

    # ---- 步骤 1：计算原始注意力分数 ----
    # Q @ K^T: 每个 query 与所有 key 的相似度
    # 结果矩阵的 [i, j] 位置表示 token i 对 token j 的关注程度
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))
    # → (B, h, S_q, S_k)

    # ---- 步骤 2：缩放 ----
    # 除以 sqrt(d_k) 防止点积过大导致 softmax 饱和
    # 若不缩放，d_k 很大时点积结果方差约为 d_k，
    # softmax 输入过大会趋近 one-hot，梯度消失 → 修炼停滞（反噬）
    attention_scores = attention_scores / math.sqrt(d_k)

    # ---- 步骤 3：应用遮罩（如有需要） ----
    # 在 Decoder 中，未来的位置必须被屏蔽（因果律：不能偷看未来）
    # mask 中为 0 的位置会被填充为 -inf，softmax 后变为 0
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # ---- 步骤 4：Softmax 归一化 ----
    # 将分数转化为概率分布（每行之和为 1）
    # 这决定了每个 token 对其他各 token 的"关注度"分配
    attention_weights = F.softmax(attention_scores, dim=-1)
    # → (B, h, S_q, S_k)，每行是一个概率分布

    # 可选的注意力 Dropout（防止对某些位置过度依赖）
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # ---- 步骤 5：加权求和 ----
    # 每个 token 的输出 = 所有 token 的 Value 按注意力权重的加权和
    # 关注度高的 token 贡献更大
    output = torch.matmul(attention_weights, V)
    # → (B, h, S_q, d_v)

    return output, attention_weights


# ╔══════════════════════════════════════════════════════════════╗
# ║        第三阵法：多头注意力 —— 多重神识并行感知                ║
# ║  将注意力分成 h 个头，每个头关注信息的不同方面。              ║
# ║  就像修炼者同时开启多种灵识，各有所长，互相补充。            ║
# ╚══════════════════════════════════════════════════════════════╝

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    核心思想：
        单个注意力头只能学习一种"关注模式"（比如只关注语法结构）。
        多个头并行运作，可以同时关注不同维度的信息：
            - Head 1: 关注语法结构（主语在哪？）
            - Head 2: 关注语义关联（谁做了什么？）
            - Head 3: 关注局部关系（相邻词的联系）
            - Head 4: 关注指代关系（"它"指的是谁？）

    实现方式：
        并非真的创建 h 个独立的注意力模块。
        而是用一个大的线性层投影后，reshape 成 h 个头。
        这样更高效，且数学上等价。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数：
            d_model:    斗气维度（总维度）
            num_heads:  神识数量（注意力头数）
            dropout:    灵力散逸率
        """
        super().__init__()

        # 基础校验：d_model 必须能被 num_heads 整除
        # （确保斗气可以均匀分配到每个神识）
        assert d_model % num_heads == 0, \
            f"斗气维度 d_model={d_model} 无法均匀分配到 {num_heads} 个神识头"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头分配到的维度

        # Q / K / V 的线性投影矩阵
        # 每个矩阵都是 (d_model, d_model)，投影后再拆分成多头
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)

        # 输出投影矩阵：将多头拼接后的结果投影回 d_model 维
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # 保存最近一次的注意力权重，用于可视化分析
        self._attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数：
            query: (B, S_q, d_model)  —— 查询来源
            key:   (B, S_k, d_model)  —— 键来源
            value: (B, S_k, d_model)  —— 值来源
            mask:  可选遮罩

        输出: (B, S_q, d_model)  —— 维度与 query 相同

        自注意力时：query = key = value = x（自己关注自己）
        交叉注意力时：query 来自 Decoder，key/value 来自 Encoder
        """
        batch_size = query.size(0)

        # ---- 步骤 1：线性投影 ----
        # 将输入通过线性层投影为 Q, K, V
        Q = self.W_Q(query)  # (B, S_q, d_model)
        K = self.W_K(key)    # (B, S_k, d_model)
        V = self.W_V(value)  # (B, S_k, d_model)

        # ---- 步骤 2：拆分多头 ----
        # (B, S, d_model) → (B, S, h, d_k) → (B, h, S, d_k)
        # 先 view 拆分最后一个维度，再 transpose 把 head 维度提前
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在 Q: (B, h, S_q, d_k), K: (B, h, S_k, d_k), V: (B, h, S_k, d_k)

        # ---- 步骤 3：计算注意力 ----
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        # attn_output: (B, h, S_q, d_k)
        # attn_weights: (B, h, S_q, S_k)

        # 保存注意力权重（用于后续分析和可视化）
        self._attention_weights = attn_weights.detach()

        # ---- 步骤 4：合并多头 ----
        # (B, h, S_q, d_k) → (B, S_q, h, d_k) → (B, S_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # ---- 步骤 5：输出投影 ----
        # 将拼接后的多头输出映射回统一的表示空间
        output = self.W_O(attn_output)  # (B, S_q, d_model)
        return self.output_dropout(output)


# ╔══════════════════════════════════════════════════════════════╗
# ║      第四阵法：逐位置前馈网络 —— 灵力放大与变换炉             ║
# ║  对每个位置独立施加两层非线性变换。                          ║
# ║  注意力负责"交流"，FFN 负责"消化"。                       ║
# ╚══════════════════════════════════════════════════════════════╝

class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络（Position-wise Feed-Forward Network）

    修炼原理：
        注意力机制完成了 token 之间的信息交流，
        但每个 token 还需要独自"消化吸收"所获信息。
        FFN 就是这个消化过程：先放大（升维），再凝练（降维）。

    结构：
        Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    维度变化：
        (B, S, d_model) → (B, S, d_ff) → (B, S, d_model)
        通常 d_ff = 4 * d_model（灵力放大 4 倍后再凝聚回原始维度）
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        参数：
            d_model: 输入/输出维度（斗气维度）
            d_ff:    中间层维度（灵力放大倍率 × d_model）
            dropout: 灵力散逸率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 灵力放大
        self.linear2 = nn.Linear(d_ff, d_model)    # 灵力凝聚
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model)  →  output: (B, S, d_model)
        """
        # 放大 → 激发 → 散逸控制 → 凝聚
        x = self.linear1(x)      # (B, S, d_model) → (B, S, d_ff)   灵力放大
        x = F.gelu(x)            # GELU 激活：平滑的灵力激发函数
        x = self.dropout(x)      # 防止灵力过于集中（正则化）
        x = self.linear2(x)      # (B, S, d_ff) → (B, S, d_model)   灵力凝聚
        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║     第五阵法：Transformer 编码器层 —— 天道法则·感知阵         ║
# ║  一个完整的编码器层 = 多头自注意力 + 前馈网络                  ║
# ║  两个子层各自配有残差连接和层归一化。                         ║
# ╚══════════════════════════════════════════════════════════════╝

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层（Pre-LN 版本）

    架构：
        x → LayerNorm → Multi-Head Self-Attention → + (残差)
          → LayerNorm → Feed-Forward Network        → + (残差) → output

    Pre-LN vs Post-LN:
        原始论文使用 Post-LN（先子层运算再归一化），但训练不稳定。
        现代实践使用 Pre-LN（先归一化再子层运算），训练更平稳。
        就像运功之前先稳定斗气，比运功之后再调整更安全。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 多头自注意力（多重神识感知）
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 逐位置前馈网络（灵力变换炉）
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 层归一化（稳定灵力流的法阵）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, S, d_model)  →  output: (B, S, d_model)   维度不变！

        这意味着多个编码器层可以像积木一样堆叠。
        """
        # ---- 子层 1：自注意力 + 残差 ----
        # Pre-LN: 先归一化，再做注意力
        residual = x                                # 保存原始输入（斗气捷径备份）
        x_norm = self.norm1(x)                      # 稳定灵力流
        attn_output = self.self_attention(           # 多重神识感知
            query=x_norm, key=x_norm, value=x_norm,  # 自注意力：Q=K=V
            mask=mask
        )
        x = residual + attn_output                  # 残差连接（斗气捷径）

        # ---- 子层 2：前馈网络 + 残差 ----
        residual = x                                # 保存（斗气捷径备份）
        x_norm = self.norm2(x)                      # 稳定灵力流
        ff_output = self.feed_forward(x_norm)       # 灵力变换
        x = residual + self.dropout(ff_output)      # 残差连接（斗气捷径）

        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║     第六阵法：Transformer 解码器层 —— 天道法则·推演阵         ║
# ║  解码器比编码器多一个交叉注意力层（从 Encoder 获取信息）。     ║
# ║  且自注意力使用因果遮罩（不能偷看未来）。                     ║
# ╚══════════════════════════════════════════════════════════════╝

class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层（Pre-LN 版本）

    架构：
        x → LN → Masked Self-Attention   → + (残差)    ← 因果自注意力
          → LN → Cross-Attention          → + (残差)    ← 从 Encoder 获取信息
          → LN → Feed-Forward Network     → + (残差)    ← 灵力变换
          → output

    与编码器的关键区别：
    1. 自注意力使用因果遮罩（因果律：只能感知过去，不能偷看未来）
    2. 多了一个交叉注意力层：Q 来自 Decoder，K/V 来自 Encoder
       （从感知阵的结果中获取源序列的信息）
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 因果自注意力（遵循因果律的神识感知）
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 交叉注意力（从编码器获取信息的桥梁）
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 三个子层各自的层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数：
            x:               (B, S_tgt, d_model)  —— Decoder 当前输入
            encoder_output:  (B, S_src, d_model)  —— Encoder 的输出
            src_mask:        源序列遮罩（如 padding mask）
            tgt_mask:        目标序列遮罩（因果遮罩）

        输出: (B, S_tgt, d_model)
        """
        # ---- 子层 1：因果自注意力 ----
        # 每个位置只能看到它之前的位置（因果律）
        residual = x
        x_norm = self.norm1(x)
        self_attn_output = self.masked_self_attention(
            query=x_norm, key=x_norm, value=x_norm,
            mask=tgt_mask  # 因果遮罩：上三角为 -inf
        )
        x = residual + self_attn_output

        # ---- 子层 2：交叉注意力 ----
        # Q 来自 Decoder（"我需要源序列的什么信息？"）
        # K, V 来自 Encoder（"源序列能提供什么信息"）
        residual = x
        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attention(
            query=x_norm,            # Q: 来自 Decoder
            key=encoder_output,      # K: 来自 Encoder
            value=encoder_output,    # V: 来自 Encoder
            mask=src_mask
        )
        x = residual + cross_attn_output

        # ---- 子层 3：前馈网络 ----
        residual = x
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + self.dropout(ff_output)

        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║            完整 Transformer 模型 —— 天道法则全阵             ║
# ║  编码器堆栈 + 解码器堆栈 + 嵌入层 + 输出投影                  ║
# ╚══════════════════════════════════════════════════════════════╝

class Transformer(nn.Module):
    """
    完整的 Transformer 模型 —— 天道法则·全阵

    组件清单：
    1. 源序列嵌入层（灵文→斗气向量）
    2. 目标序列嵌入层
    3. 位置编码（时空感知法阵）
    4. N 层 Encoder blocks（感知阵堆叠）
    5. N 层 Decoder blocks（推演阵堆叠）
    6. 最终层归一化
    7. 输出线性投影（斗气→灵文预测）

    数据流：
        src tokens → embedding → pos encoding → encoder stack → encoder output
        tgt tokens → embedding → pos encoding → decoder stack (+ encoder output) → logits
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = False,
    ):
        """
        参数：
            src_vocab_size:     源灵文库大小
            tgt_vocab_size:     目标灵文库大小
            d_model:            斗气维度（各层统一维度）
            num_heads:          神识头数
            d_ff:               灵力放大维度（通常 4 * d_model）
            num_encoder_layers: 感知阵层数
            num_decoder_layers: 推演阵层数
            max_seq_len:        最大序列长度
            dropout:            灵力散逸率
            tie_embeddings:     是否共享源/目标嵌入权重
        """
        super().__init__()

        self.d_model = d_model

        # ---- 嵌入层：将离散的灵文 ID 映射为连续的斗气向量 ----
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 可选：共享嵌入权重（当源/目标使用相同灵文库时）
        if tie_embeddings:
            assert src_vocab_size == tgt_vocab_size, "共享嵌入要求词表大小相同"
            self.tgt_embedding.weight = self.src_embedding.weight

        # ---- 位置编码 ----
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # ---- 嵌入缩放因子 ----
        # 原始论文建议将嵌入乘以 sqrt(d_model)
        # 理由：嵌入向量的方差约为 1/d_model，乘以 sqrt(d_model) 后方差约为 1
        # 使其与位置编码在同一量级上
        self.embed_scale = math.sqrt(d_model)

        # ---- Encoder 堆叠 ----
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)  # 最终归一化

        # ---- Decoder 堆叠 ----
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)  # 最终归一化

        # ---- 输出投影层 ----
        # 将 d_model 维的斗气向量映射到目标灵文库大小
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # ---- 参数初始化 ----
        self._init_parameters()

    def _init_parameters(self):
        """
        使用 Xavier 均匀初始化——保证训练初期灵力流动平稳。
        良好的初始化就像修炼前先调息运气，避免一开始就发生反噬。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(size: int, device: torch.device = None) -> torch.Tensor:
        """
        生成因果遮罩（下三角矩阵）

        因果律：在推演阵（Decoder）中，每个位置只能"看到"之前的位置。
        第 i 个位置只能关注位置 0, 1, ..., i。

        返回: (1, 1, size, size) 的下三角遮罩矩阵
              1 = 可见，0 = 不可见

        示例（size=4）：
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 head 维度

    @staticmethod
    def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        生成填充遮罩

        将 padding 位置标记为 0（不可见），其他位置标记为 1（可见）。
        这样注意力计算时不会关注到无意义的 padding 位置。

        seq: (B, S) 输入序列
        返回: (B, 1, 1, S) 遮罩矩阵
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        编码器前向传播

        src: (B, S_src) —— 源序列 token IDs
        返回: (B, S_src, d_model) —— 编码后的表示
        """
        # 嵌入 + 缩放 + 位置编码
        x = self.src_embedding(src) * self.embed_scale
        x = self.pos_encoding(x)

        # 通过所有编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)

        # 最终归一化
        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        解码器前向传播

        tgt: (B, S_tgt) —— 目标序列 token IDs
        encoder_output: (B, S_src, d_model) —— 编码器输出
        返回: (B, S_tgt, d_model) —— 解码后的表示
        """
        # 嵌入 + 缩放 + 位置编码
        x = self.tgt_embedding(tgt) * self.embed_scale
        x = self.pos_encoding(x)

        # 通过所有解码器层
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # 最终归一化
        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        完整的前向传播：编码 → 解码 → 投影

        src: (B, S_src)
        tgt: (B, S_tgt)
        输出: (B, S_tgt, tgt_vocab_size)  —— 每个位置对目标灵文库的预测分布
        """
        # 编码
        encoder_output = self.encode(src, src_mask)

        # 解码
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # 投影到灵文库
        logits = self.output_projection(decoder_output)
        # → (B, S_tgt, tgt_vocab_size)

        return logits


# ╔══════════════════════════════════════════════════════════════╗
# ║            修炼验证：序列复制任务                               ║
# ║  模型目标：输入什么就输出什么（学会完美复制）。                ║
# ║  这是检验 Transformer 实现是否正确的经典方法。                ║
# ║  如果连复制都学不会，说明阵法有缺陷。                        ║
# ╚══════════════════════════════════════════════════════════════╝

def create_copy_task_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple:
    """
    生成序列复制任务的一批训练药材。

    任务：输入 [3, 7, 1, 9, 5] → 输出 [3, 7, 1, 9, 5]

    为什么用这个任务？
    因为它简单且可验证：如果模型学会了完美复制，说明：
    1. 编码器能正确编码输入信息
    2. 解码器能正确利用编码器的输出
    3. 注意力机制能正确传递信息
    4. 因果遮罩工作正常

    返回：
        src:        (B, seq_len)       源序列
        tgt_input:  (B, seq_len - 1)   解码器输入（去掉最后一个 token）
        tgt_output: (B, seq_len - 1)   预测目标（去掉第一个 token）
    """
    # 生成随机序列，值在 [2, vocab_size-1]
    # 保留 0 = padding, 1 = BOS (begin of sequence)
    src = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)

    # 解码器使用教师强制（teacher forcing）：
    # 输入 = 目标序列右移一位（去掉最后一个 token）
    # 输出 = 目标序列左移一位（去掉第一个 token）
    tgt_input = src[:, :-1]
    tgt_output = src[:, 1:]

    return src, tgt_input, tgt_output


def train_copy_task():
    """
    序列复制修炼：验证天道法则的完整性。

    若修炼成功（loss 降至接近 0，准确率接近 100%），
    说明手搓的 Transformer 阵法完整无缺。
    """
    # ============ 修炼配置 ============
    VOCAB_SIZE = 30          # 灵文种类
    SEQ_LEN = 12             # 序列长度
    D_MODEL = 64             # 斗气维度
    NUM_HEADS = 4            # 神识头数（d_model / num_heads = 16 per head）
    D_FF = 256               # 灵力放大维度（4 × d_model）
    NUM_LAYERS = 2           # 阵法层数（编码器和解码器各 2 层）
    BATCH_SIZE = 64          # 每炉药材数量
    NUM_EPOCHS = 200         # 修炼轮数
    LEARNING_RATE = 3e-4     # 斗气调节步幅
    DROPOUT = 0.1            # 灵力散逸率
    GRAD_CLIP = 1.0          # 梯度裁剪阈值（防止梯度爆炸 = 反噬）

    # ============ 选择异火 ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'=' * 60}")
    print(f"  《焚诀》第二卷 · 天道法则修炼验证")
    print(f"  任务：序列复制")
    print(f"  异火：{device}")
    print(f"{'=' * 60}")

    # ============ 构建天道法则阵法 ============
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN + 10,
        dropout=DROPOUT,
        tie_embeddings=True,  # 源/目标共享灵文库
    ).to(device)

    # 统计模型参数量（斗气总量）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n阵法参数总量（斗气总量）: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print()

    # ============ 配置优化器 ============
    # Adam 优化器：自适应斗气调节法
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),  # 原始论文推荐的 beta 值
        eps=1e-9,
    )

    # 损失函数：交叉熵（天罚函数）
    criterion = nn.CrossEntropyLoss()

    # ============ 开始修炼 ============
    print("开始修炼...\n")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()

        # 生成训练药材
        src, tgt_input, tgt_output = create_copy_task_batch(
            BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device
        )

        # 生成因果遮罩（解码器必须遵守因果律）
        tgt_mask = Transformer.generate_causal_mask(
            tgt_input.size(1), device=device
        )

        # ---- 前向传播 ----
        logits = model(src, tgt_input, tgt_mask=tgt_mask)
        # logits: (B, S_tgt, vocab_size)

        # ---- 计算天罚（损失） ----
        # reshape: (B * S_tgt, vocab_size) vs (B * S_tgt,)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )

        # ---- 反向传播 + 参数更新 ----
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪：防止梯度爆炸（反噬）
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        # ---- 每 20 轮输出修炼日志 ----
        if (epoch + 1) % 20 == 0:
            # 计算准确率
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)  # (B, S_tgt)
                correct = (predictions == tgt_output).float().mean().item()

            elapsed = time.time() - start_time
            print(
                f"第 {epoch+1:>3d}/{NUM_EPOCHS} 轮 | "
                f"天罚值: {loss.item():.4f} | "
                f"准确率: {correct*100:.1f}% | "
                f"耗时: {elapsed:.1f}s"
            )

    # ============ 最终验证 ============
    print(f"\n{'=' * 60}")
    print("修炼完成！开始最终试炼...\n")

    model.eval()
    num_perfect = 0
    num_tests = 10

    with torch.no_grad():
        for i in range(num_tests):
            # 生成测试序列
            test_src = torch.randint(
                2, VOCAB_SIZE, (1, SEQ_LEN), device=device
            )
            test_tgt_input = test_src[:, :-1]
            test_tgt_output = test_src[:, 1:]

            tgt_mask = Transformer.generate_causal_mask(
                test_tgt_input.size(1), device=device
            )

            logits = model(test_src, test_tgt_input, tgt_mask=tgt_mask)
            predicted = logits.argmax(dim=-1)

            # 构造完整预测序列（加上第一个 token）
            full_predicted = torch.cat(
                [test_src[:, :1], predicted], dim=1
            )

            is_perfect = torch.equal(full_predicted, test_src)
            if is_perfect:
                num_perfect += 1

            status = "PASS" if is_perfect else "FAIL"
            print(f"  试炼 {i+1:>2d} [{status}]:")
            print(f"    源序列: {test_src[0].tolist()}")
            print(f"    预测:   {full_predicted[0].tolist()}")

    accuracy = num_perfect / num_tests * 100
    print(f"\n最终试炼结果: {num_perfect}/{num_tests} 完美复制 ({accuracy:.0f}%)")

    if accuracy >= 80:
        print("\n天道法则修炼验证通过！")
        print("你的手搓 Transformer 阵法完整无缺。")
        print("恭喜突破到斗师之境！")
    else:
        print("\n修炼尚未圆满，建议增加修炼轮数或调整配置后重试。")

    print(f"\n{'=' * 60}")

    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║            附加修炼：仅编码器模型（类 BERT 感知阵）           ║
# ║  许多任务只需要编码器：分类、命名实体识别等。                ║
# ║  这里展示如何用我们的组件搭建一个纯编码器模型。              ║
# ╚══════════════════════════════════════════════════════════════╝

class TransformerClassifier(nn.Module):
    """
    基于 Transformer Encoder 的序列分类模型 —— 感知阵·判别型

    用法：将输入序列编码后，取 [CLS] token 或平均池化的结果做分类。
    类似于 BERT 的架构思路（简化版）。
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 3,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, S)  →  logits: (B, num_classes)
        """
        # 嵌入 + 位置编码
        x = self.pos_encoding(self.embedding(x) * self.scale)

        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.encoder_norm(x)

        # 平均池化：取所有位置的平均作为序列表示
        # (B, S, d_model) → (B, d_model)
        x = x.mean(dim=1)

        # 分类
        return self.classifier(x)


# ╔══════════════════════════════════════════════════════════════╗
# ║                      主修炼入口                               ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║     《焚诀》第二卷 · 天道法则 · 手搓 Transformer        ║")
    print("║                                                          ║")
    print("║  '纸上得来终觉浅，绝知此事要躬行。'                     ║")
    print("║  '唯有亲手搓过 Transformer 的人，才算真正入了道。'       ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 运行序列复制修炼
    trained_model = train_copy_task()

    print()
    print("提示：你也可以导入本文件中的组件，搭建自己的模型：")
    print("  from 02_手搓transformer import Transformer, TransformerClassifier")
    print()
    print("下一步修炼建议：")
    print("  1. 修改超参数（层数、头数、维度），观察对训练的影响")
    print("  2. 增加序列长度和词表大小，挑战更难的复制任务")
    print("  3. 尝试字符级语言模型任务（预测下一个字符）")
    print("  4. 可视化注意力权重，理解模型在'看'什么")
    print("  5. 进入第三卷，学习预训练与微调的大道！")
    print()
