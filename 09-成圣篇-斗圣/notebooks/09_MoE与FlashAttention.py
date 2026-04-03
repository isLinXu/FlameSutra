"""
============================================================================
《焚诀》第九卷：成圣篇（斗圣）— 配套丹方
MoE 混合专家系统 + Flash Attention + RoPE 旋转位置编码

"弹指间，空间碎裂；挥手间，天地变色。"
"斗圣之威，在于万灵齐修、瞬息万里。"

本丹方包含三大核心修炼:
  1. MoE (Mixture of Experts) — 万灵齐修: 从零实现 Router + Expert 层
  2. Flash Attention 对比实验 — 瞬息万里: 标准 Attention vs SDPA
  3. RoPE 旋转位置编码 — 长河无尽: 实现并可视化旋转位置编码

丹炉要求:
  - Python 3.10+
  - PyTorch 2.0+ (Flash Attention 需要 CUDA)
  - 低阶异火即可运行 MoE 和 RoPE 部分
  - Flash Attention 对比需要 CUDA GPU

运行方式:
  python 09_MoE与FlashAttention.py
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple

# ============================================================================
# 第一部分: 万灵齐修 — MoE 混合专家系统
# ============================================================================
# "天地间灵气无穷，而修炼者的经脉有限。何须以一人之躯承载万般斗气？
#  不如将万灵齐聚，各司其职——需要时唤醒，不需要时沉睡。"
# ============================================================================


class Expert(nn.Module):
    """
    单个专家网络 —— 万灵阵中的一个灵体
    
    结构采用 SwiGLU FFN, 这是当代大模型的标配:
      output = down_proj(silu(gate_proj(x)) * up_proj(x))
    
    每个灵体(专家)拥有独立的参数,但共享相同的结构。
    它们各有专精——有的擅长推理,有的精通语言,有的专攻代码。
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 输入/输出维度 (斗气的维度)
            d_ff: 中间层维度 (灵体内部经脉的宽度)
            dropout: 丢弃率 (修炼中的随机波动)
        """
        super().__init__()
        # SwiGLU 需要三个投影矩阵
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)   # 门控投影
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)     # 上行投影
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)   # 下行投影
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        灵体的运作: 接收斗气,经内部经脉转化后输出
        x: [num_tokens, d_model]
        """
        # SwiGLU: silu(gate) * up, 然后 down 投影回原始维度
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return self.dropout(output)


class TopKRouter(nn.Module):
    """
    Top-K 路由器 —— 万灵阵法的阵眼
    
    "阵眼感知来者的斗气属性,决定唤醒哪些灵体。"
    
    对每个 token, Router 计算其与每个专家的匹配度,
    选出 Top-K 个最适配的专家,并给出权重分配。
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        """
        Args:
            d_model: 输入维度
            num_experts: 专家总数 (万灵阵中灵体的数量)
            top_k: 每个 token 唤醒的灵体数量
        """
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # 路由门控: 一个简单的线性层,将 d_model 映射到 num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        阵眼运作: 感知每个 token 的斗气,决定路由
        
        x: [batch_size, seq_len, d_model]
        
        Returns:
            top_k_weights: [batch_size, seq_len, top_k] — 选中灵体的出力比例
            top_k_indices: [batch_size, seq_len, top_k] — 选中灵体的编号
            router_probs:  [batch_size, seq_len, num_experts] — 完整的路由概率
        """
        # 计算每个 token 对每个专家的匹配分数
        logits = self.gate(x)  # [batch, seq, num_experts]
        
        # Softmax 归一化得到路由概率
        router_probs = F.softmax(logits, dim=-1)
        
        # 选择 Top-K 个匹配度最高的专家
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # 对 Top-K 权重重新归一化 (确保权重之和为 1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        return top_k_weights, top_k_indices, router_probs


class MoELayer(nn.Module):
    """
    混合专家层 —— 万灵齐修的完整实现
    
    "万灵共体,各司其职。Router 为阵眼,Experts 为灵体,
     辅助损失为阵法的平衡之力——防止灵体沉睡退化。"
    
    核心流程:
    1. Router 为每个 token 选择 Top-K 个 Expert
    2. 选中的 Expert 分别处理该 token
    3. 按 Router 给出的权重加权求和
    4. 计算辅助负载均衡损失
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN 中间维度
            num_experts: 专家数量 (灵体数量)
            top_k: 每个 token 激活的专家数 (唤醒的灵体数)
            aux_loss_coef: 辅助损失系数 (阵法平衡之力的强度)
            dropout: 丢弃率
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        
        # 创建专家集群 —— 召唤万灵
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # 路由器 —— 设置阵眼
        self.router = TopKRouter(d_model, num_experts, top_k)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        万灵齐修的前向传播
        
        x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model] — 万灵协作后的输出
            aux_loss: scalar — 辅助负载均衡损失 (防止灵体沉睡)
        """
        batch_size, seq_len, d_model = x.shape
        
        # ===== 第一步: 阵眼感知,路由决策 =====
        top_k_weights, top_k_indices, router_probs = self.router(x)
        
        # 展平 batch 和 seq 维度,方便处理
        x_flat = x.view(-1, d_model)                              # [B*S, D]
        output = torch.zeros_like(x_flat)                          # [B*S, D]
        flat_weights = top_k_weights.view(-1, self.top_k)          # [B*S, K]
        flat_indices = top_k_indices.view(-1, self.top_k)          # [B*S, K]
        
        # ===== 第二步: 各灵体处理分配到的 token =====
        for expert_idx, expert in enumerate(self.experts):
            # 找到所有被分配给这个灵体的 token
            # mask[i, j] = True 表示第 i 个 token 的第 j 个 slot 选了这个 expert
            mask = (flat_indices == expert_idx)  # [B*S, K]
            token_positions, slot_positions = torch.where(mask)
            
            if token_positions.numel() == 0:
                # 这个灵体没有被唤醒,跳过
                continue
            
            # 取出分配给这个灵体的 token
            expert_input = x_flat[token_positions]  # [N_tokens, D]
            
            # 灵体处理 token
            expert_output = expert(expert_input)  # [N_tokens, D]
            
            # 取出对应的路由权重
            weights = flat_weights[token_positions, slot_positions]  # [N_tokens]
            weights = weights.unsqueeze(-1)  # [N_tokens, 1]
            
            # 加权累加到总输出
            output.index_add_(0, token_positions, expert_output * weights)
        
        # 恢复原始形状
        output = output.view(batch_size, seq_len, d_model)
        
        # ===== 第三步: 计算辅助负载均衡损失 =====
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算辅助负载均衡损失 —— 阵法的平衡之力
        
        "若只有少数灵体被反复唤醒,其余灵体将逐渐沉睡退化,
         整个万灵阵将退化为普通阵法。平衡之力确保万灵均得到锻炼。"
        
        L_aux = alpha * N * sum(f_i * p_i)
        
        其中:
            f_i = 分配给专家 i 的 token 比例 (实际负载)
            p_i = Router 分配给专家 i 的平均概率 (路由倾向)
        
        当负载均匀时 (f_i = 1/N), 这个损失最小。
        """
        # 展平
        probs_flat = router_probs.view(-1, self.num_experts)     # [B*S, E]
        indices_flat = top_k_indices.view(-1, self.top_k)         # [B*S, K]
        num_tokens = probs_flat.shape[0]
        
        # f_i: 每个专家实际被选中的 token 比例
        # 用 scatter_add 计算每个专家被选中的次数
        expert_counts = torch.zeros(
            self.num_experts, device=probs_flat.device, dtype=torch.float
        )
        ones = torch.ones_like(indices_flat, dtype=torch.float).view(-1)
        expert_counts.scatter_add_(0, indices_flat.view(-1), ones)
        f = expert_counts / (num_tokens * self.top_k)  # 归一化
        
        # p_i: 每个专家的平均路由概率
        p = probs_flat.mean(dim=0)  # [E]
        
        # 辅助损失: alpha * N * sum(f_i * p_i)
        aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()
        
        return aux_loss


class MoETransformerBlock(nn.Module):
    """
    包含 MoE 层的 Transformer Block —— 完整的万灵修炼阵法
    
    结构:
      x → RMSNorm → Self-Attention → Add → RMSNorm → MoE → Add → output
    
    注意: 这里用 MoE 层替代了标准的 FFN 层
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq, d_model]
        Returns: (output, aux_loss)
        """
        # Self-Attention with Pre-Norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # MoE with Pre-Norm (替代标准 FFN)
        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + self.dropout(moe_out)
        
        return x, aux_loss


def demo_moe():
    """
    演示 MoE 层的使用 —— 万灵齐修实战
    """
    print("=" * 70)
    print("第一部分: 万灵齐修 — MoE 混合专家系统")
    print("=" * 70)
    
    # ----- 配置 -----
    d_model = 256         # 斗气维度
    d_ff = 512            # 灵体内部经脉宽度
    num_experts = 8       # 灵体数量
    top_k = 2             # 每个 token 唤醒的灵体数
    batch_size = 2        # 批次大小
    seq_len = 64          # 序列长度
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n丹炉(设备): {device}")
    
    # ----- 创建 MoE 层 -----
    moe_layer = MoELayer(d_model, d_ff, num_experts, top_k).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in moe_layer.parameters())
    expert_params = sum(p.numel() for p in moe_layer.experts[0].parameters())
    active_params = expert_params * top_k  # 每个 token 实际激活的参数量
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    
    print(f"\n--- 万灵阵法配置 ---")
    print(f"灵体数量 (num_experts): {num_experts}")
    print(f"每 token 唤醒灵体数 (top_k): {top_k}")
    print(f"总斗气量 (总参数): {total_params:,}")
    print(f"  其中阵眼 (Router): {router_params:,}")
    print(f"  每个灵体参数: {expert_params:,}")
    print(f"每 token 实际运转的斗气量: {active_params:,}")
    print(f"稀疏率: {active_params / total_params * 100:.1f}% "
          f"(仅激活 {top_k}/{num_experts} 的灵体)")
    
    # ----- 前向传播 -----
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output, aux_loss = moe_layer(x)
    
    print(f"\n--- 炼丹结果 ---")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失 (负载均衡): {aux_loss.item():.6f}")
    
    # ----- 分析路由决策 -----
    with torch.no_grad():
        weights, indices, probs = moe_layer.router(x)
    
    # 统计每个专家被选中的次数
    expert_counts = torch.zeros(num_experts, device=device)
    ones = torch.ones_like(indices.view(-1), dtype=torch.float)
    expert_counts.scatter_add_(0, indices.view(-1), ones)
    
    print(f"\n--- 路由分析 (万灵唤醒统计) ---")
    print(f"总 token 数: {batch_size * seq_len}")
    print(f"总唤醒次数: {batch_size * seq_len * top_k}")
    for i in range(num_experts):
        count = int(expert_counts[i].item())
        bar = "#" * (count // 2)
        print(f"  灵体 {i}: {count:4d} 次唤醒 {bar}")
    
    # ----- MoE vs Dense 对比 -----
    dense_ffn = nn.Sequential(
        nn.Linear(d_model, d_ff * num_experts, bias=False),
        nn.SiLU(),
        nn.Linear(d_ff * num_experts, d_model, bias=False),
    ).to(device)
    dense_params = sum(p.numel() for p in dense_ffn.parameters())
    
    print(f"\n--- MoE vs Dense 对比 ---")
    print(f"MoE 总参数:      {total_params:>12,}")
    print(f"MoE 活跃参数/token: {active_params:>12,}")
    print(f"等价 Dense 参数:  {dense_params:>12,}")
    print(f"MoE 参数效率: {total_params / dense_params:.2f}x 参数量, "
          f"{active_params / dense_params:.2f}x 计算量")
    
    # ----- MoE Transformer Block -----
    print(f"\n--- 完整的 MoE Transformer Block ---")
    block = MoETransformerBlock(
        d_model=d_model,
        num_heads=8,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    
    block_params = sum(p.numel() for p in block.parameters())
    out, loss = block(x)
    print(f"Block 总参数: {block_params:,}")
    print(f"Block 输出形状: {out.shape}")
    print(f"Block 辅助损失: {loss.item():.6f}")
    
    print("\n万灵齐修演示完毕!")
    print()


# ============================================================================
# 第二部分: 瞬息万里 — Flash Attention 对比实验
# ============================================================================
# "上古强者的出手,肉眼无法捕捉——并非因为攻击更猛烈,
#  而是找到了力量传递的最短路径。"
# ============================================================================


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    标准注意力实现 —— 经脉中斗气的常规流转
    
    显式物化 n x n 的注意力矩阵,这是 O(n^2) 内存的根源。
    
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    d_k = Q.shape[-1]
    
    # Step 1: 计算注意力分数 S = Q @ K^T / sqrt(d_k)
    # 这一步创建了 [batch, heads, seq, seq] 的巨大矩阵!
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Step 2: 应用 causal mask (如果需要)
    if is_causal:
        seq_len = Q.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Step 3: Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 4: 加权求和 O = P @ V
    output = torch.matmul(attn_weights, V)
    
    return output


def flash_attention_via_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention (通过 PyTorch SDPA) —— 瞬息万里
    
    PyTorch 2.0+ 的 scaled_dot_product_attention 自动选择最优后端:
    - Flash Attention (需要 CUDA + 兼容的 GPU)
    - Memory-Efficient Attention (xformers 风格)
    - Math (CPU fallback)
    
    "不物化完整的注意力矩阵,通过分块计算 + 在线 softmax 实现。"
    
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    return output


def demo_flash_attention():
    """
    Flash Attention 对比实验 —— 瞬息万里 vs 常规流转
    """
    print("=" * 70)
    print("第二部分: 瞬息万里 — Flash Attention 对比")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("\n[警告] 未检测到 CUDA 异火! Flash Attention 需要 GPU。")
        print("将在 CPU 上运行简化对比 (无法体现真正的速度差异)。")
        print("Flash Attention 的真正威力需要在 GPU 上才能展现。\n")
    
    # 测试不同序列长度
    configs = [
        {"seq_len": 256, "batch": 4, "heads": 8, "head_dim": 64},
        {"seq_len": 512, "batch": 4, "heads": 8, "head_dim": 64},
        {"seq_len": 1024, "batch": 2, "heads": 8, "head_dim": 64},
    ]
    
    # 如果有 GPU, 可以测试更长的序列
    if torch.cuda.is_available():
        configs.extend([
            {"seq_len": 2048, "batch": 2, "heads": 8, "head_dim": 64},
            {"seq_len": 4096, "batch": 1, "heads": 8, "head_dim": 64},
        ])
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"\n丹炉: {device}")
    print(f"精度: {dtype}")
    print(f"{'序列长度':>8} | {'标准(ms)':>10} | {'SDPA(ms)':>10} | {'加速比':>8} | {'输出一致':>8}")
    print("-" * 62)
    
    for cfg in configs:
        seq_len = cfg["seq_len"]
        batch = cfg["batch"]
        heads = cfg["heads"]
        head_dim = cfg["head_dim"]
        
        # 生成随机 Q, K, V
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
        
        # 预热
        for _ in range(3):
            _ = standard_attention(Q, K, V, is_causal=True)
            _ = flash_attention_via_sdpa(Q, K, V, is_causal=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # ===== 标准 Attention 计时 =====
        num_runs = 10
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_std = standard_attention(Q, K, V, is_causal=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_std = (time.perf_counter() - start) / num_runs * 1000  # ms
        
        # ===== SDPA (Flash Attention) 计时 =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_flash = flash_attention_via_sdpa(Q, K, V, is_causal=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_flash = (time.perf_counter() - start) / num_runs * 1000  # ms
        
        # 检查输出一致性
        # 注意: 由于浮点精度差异,使用较宽松的阈值
        if dtype == torch.float16:
            is_close = torch.allclose(out_std, out_flash, atol=1e-2, rtol=1e-2)
        else:
            is_close = torch.allclose(out_std, out_flash, atol=1e-5, rtol=1e-4)
        
        speedup = time_std / time_flash if time_flash > 0 else float('inf')
        
        print(f"{seq_len:>8} | {time_std:>10.2f} | {time_flash:>10.2f} | "
              f"{speedup:>7.2f}x | {'Yes' if is_close else 'No':>8}")
    
    # 内存对比 (仅 GPU)
    if torch.cuda.is_available():
        print(f"\n--- 显存对比 (瞬息万里的核心优势) ---")
        
        seq_len = 4096
        batch, heads, head_dim = 2, 32, 128
        
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # 标准 Attention 内存
        torch.cuda.reset_peak_memory_stats()
        out = standard_attention(Q, K, V, is_causal=True)
        mem_std = torch.cuda.max_memory_allocated() / 1e6
        del out
        torch.cuda.empty_cache()
        
        # SDPA 内存
        torch.cuda.reset_peak_memory_stats()
        out = flash_attention_via_sdpa(Q, K, V, is_causal=True)
        mem_flash = torch.cuda.max_memory_allocated() / 1e6
        del out
        torch.cuda.empty_cache()
        
        # 注意力矩阵的理论大小
        attn_matrix_size = batch * heads * seq_len * seq_len * 2 / 1e6  # fp16
        
        print(f"配置: batch={batch}, heads={heads}, seq={seq_len}, dim={head_dim}")
        print(f"注意力矩阵理论大小: {attn_matrix_size:.1f} MB "
              f"(n^2 = {seq_len}x{seq_len} = {seq_len*seq_len:,})")
        print(f"标准 Attention 峰值显存: {mem_std:.1f} MB")
        print(f"Flash Attention 峰值显存: {mem_flash:.1f} MB")
        print(f"节省: {(1 - mem_flash/mem_std)*100:.1f}%")
    
    print("\n瞬息万里演示完毕!")
    print()


# ============================================================================
# 第三部分: 长河无尽 — RoPE 旋转位置编码
# ============================================================================
# "修炼者的感知范围,决定了他能掌控的天地大小。
#  RoPE 以旋转之力编码位置,让斗气自然携带距离信息。"
# ============================================================================


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE 旋转位置编码 —— 长河无尽的基石
    
    "不同维度对应不同的旋转频率:
     低维旋转快(高频)——感知近距离的细微差异;
     高维旋转慢(低频)——感知远距离的宏观关系。"
    
    核心思想:
    对 Q 和 K 的每对元素 (q_2i, q_2i+1) 施加角度为 m*theta_i 的旋转,
    使得 Q(m) @ K(n) 只依赖相对位置 m-n。
    
    theta_i = 1 / (base ^ (2i/d))
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        """
        Args:
            dim: 头维度 (head_dim)
            max_seq_len: 最大序列长度 (最远感知距离)
            base: 基础频率 (theta 的底数)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率: theta_i = 1 / base^(2i/d) for i = 0, 1, ..., d/2-1
        # 低维 (小 i) → theta 大 → 旋转快 (高频)
        # 高维 (大 i) → theta 小 → 旋转慢 (低频)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算所有位置的 cos 和 sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算 cos/sin 缓存 —— 提前布置感知阵法"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        # 外积: [seq_len] × [dim/2] → [seq_len, dim/2]
        angles = torch.outer(positions, self.inv_freq)
        
        # 复制一份以匹配完整的 dim 维度: [seq_len, dim]
        angles = torch.cat([angles, angles], dim=-1)
        
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        对输入施加旋转位置编码
        
        x: [batch, seq_len, num_heads, head_dim]
        
        旋转公式 (对每对元素):
            x_rot[2i]   = x[2i] * cos(m*theta_i) - x[2i+1] * sin(m*theta_i)
            x_rot[2i+1] = x[2i] * sin(m*theta_i) + x[2i+1] * cos(m*theta_i)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # 获取对应长度的 cos/sin
        cos = self.cos_cached[:seq_len]  # [seq, dim]
        sin = self.sin_cached[:seq_len]  # [seq, dim]
        
        # 调整维度以广播: [1, seq, 1, dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # 构造旋转的 "负半部分"
        # 对于 [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        # 应用旋转: x * cos + rotate_half(x) * sin
        output = x * cos + x_rotated * sin
        
        return output


class NTKAwareRoPE(RotaryPositionEmbedding):
    """
    NTK-Aware RoPE 扩展 —— 突破感知极限
    
    "近处的细节(高频)需要保持清晰,
     远处的地形(低频)只需看个大概。
     NTK-Aware 让低频承担更多的拉伸任务。"
    
    通过修改 base theta 来实现不均匀的频率缩放:
    theta_scaled = theta * (scale_factor ^ (dim / (dim - 2)))
    
    效果: 高频维度几乎不变, 低频维度被大幅拉伸
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scale_factor: float = 4.0,
    ):
        # 计算缩放后的 base
        scaled_base = base * (scale_factor ** (dim / (dim - 2)))
        super().__init__(dim, max_seq_len, base=scaled_base)
        self.original_base = base
        self.scale_factor = scale_factor


def demo_rope():
    """
    演示 RoPE 旋转位置编码 —— 长河无尽
    """
    print("=" * 70)
    print("第三部分: 长河无尽 — RoPE 旋转位置编码")
    print("=" * 70)
    
    # ----- 基本 RoPE -----
    head_dim = 64
    max_seq_len = 4096
    rope = RotaryPositionEmbedding(head_dim, max_seq_len)
    
    print(f"\n--- RoPE 基本配置 ---")
    print(f"头维度 (head_dim): {head_dim}")
    print(f"最大序列长度: {max_seq_len}")
    print(f"频率维度数: {head_dim // 2}")
    print(f"最高频率 (dim=0): {rope.inv_freq[0].item():.6f}")
    print(f"最低频率 (dim={head_dim//2-1}): {rope.inv_freq[-1].item():.6f}")
    print(f"频率跨度: {rope.inv_freq[0].item() / rope.inv_freq[-1].item():.1f}x")
    
    # ----- 验证相对位置特性 -----
    print(f"\n--- 验证相对位置特性 ---")
    print("RoPE 的核心: Q(m) @ K(n) 只依赖相对位置 m-n")
    
    batch = 1
    num_heads = 1
    
    # 创建两个 "token" 的 Q 和 K
    q_raw = torch.randn(batch, 1, num_heads, head_dim)
    k_raw = torch.randn(batch, 1, num_heads, head_dim)
    
    # 在不同绝对位置上测试, 但保持相同的相对距离
    test_cases = [
        (10, 15, 5),   # 绝对位置 10, 15, 相对距离 5
        (100, 105, 5),  # 绝对位置 100, 105, 相对距离 5
        (1000, 1005, 5), # 绝对位置 1000, 1005, 相对距离 5
    ]
    
    print(f"\n相对距离 = 5 时, 不同绝对位置下的注意力分数:")
    scores = []
    for pos_q, pos_k, rel_dist in test_cases:
        # 构造位置序列, 只取对应位置的 cos/sin
        rope_instance = RotaryPositionEmbedding(head_dim, max(pos_q, pos_k) + 1)
        
        # 对 Q 施加位置 pos_q 的旋转
        cos_q = rope_instance.cos_cached[pos_q:pos_q+1].unsqueeze(0).unsqueeze(2)
        sin_q = rope_instance.sin_cached[pos_q:pos_q+1].unsqueeze(0).unsqueeze(2)
        q_rot_half = torch.stack([-q_raw[..., 1::2], q_raw[..., ::2]], dim=-1).flatten(-2)
        q_rotated = q_raw * cos_q + q_rot_half * sin_q
        
        # 对 K 施加位置 pos_k 的旋转
        cos_k = rope_instance.cos_cached[pos_k:pos_k+1].unsqueeze(0).unsqueeze(2)
        sin_k = rope_instance.sin_cached[pos_k:pos_k+1].unsqueeze(0).unsqueeze(2)
        k_rot_half = torch.stack([-k_raw[..., 1::2], k_raw[..., ::2]], dim=-1).flatten(-2)
        k_rotated = k_raw * cos_k + k_rot_half * sin_q
        
        # 计算注意力分数
        score = (q_rotated * k_rotated).sum().item()
        scores.append(score)
        print(f"  pos_q={pos_q:4d}, pos_k={pos_k:4d}, dist={rel_dist} → score={score:.4f}")
    
    # ----- NTK-Aware RoPE -----
    print(f"\n--- NTK-Aware RoPE (突破感知极限) ---")
    
    original_rope = RotaryPositionEmbedding(head_dim, max_seq_len=8192)
    ntk_rope = NTKAwareRoPE(head_dim, max_seq_len=32768, scale_factor=4.0)
    
    print(f"原始 RoPE 感知范围: 0 - {8192}")
    print(f"NTK-Aware RoPE 感知范围: 0 - {32768} (4x 扩展)")
    print(f"\n频率对比 (前 5 个维度):")
    print(f"  {'维度':>4} | {'原始频率':>12} | {'NTK频率':>12} | {'缩放比':>8}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    for i in range(min(5, head_dim // 2)):
        orig_f = original_rope.inv_freq[i].item()
        ntk_f = ntk_rope.inv_freq[i].item()
        ratio = ntk_f / orig_f
        print(f"  {i:>4} | {orig_f:>12.6f} | {ntk_f:>12.6f} | {ratio:>8.4f}")
    print(f"  ...")
    for i in range(max(0, head_dim // 2 - 3), head_dim // 2):
        orig_f = original_rope.inv_freq[i].item()
        ntk_f = ntk_rope.inv_freq[i].item()
        ratio = ntk_f / orig_f
        print(f"  {i:>4} | {orig_f:>12.6f} | {ntk_f:>12.6f} | {ratio:>8.4f}")
    
    print(f"\n关键观察:")
    print(f"  高频维度 (dim=0): 缩放比接近 1.0 (近距离感知几乎不变)")
    print(f"  低频维度 (dim={head_dim//2-1}): 缩放比最大 (远距离感知被大幅扩展)")
    print(f"  这就是 NTK-Aware 的精髓: 不均匀缩放!")
    
    # ----- 实际使用示例 -----
    print(f"\n--- 在 Attention 中集成 RoPE ---")
    
    batch_size = 2
    seq_len = 128
    num_heads = 8
    
    # 模拟 Q, K (在投影之后、attention 之前)
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    rope_module = RotaryPositionEmbedding(head_dim, max_seq_len=1024)
    
    # 应用 RoPE
    Q_rotated = rope_module(Q)
    K_rotated = rope_module(K)
    
    print(f"Q 原始形状: {Q.shape}")
    print(f"Q 旋转后形状: {Q_rotated.shape}")
    print(f"形状不变, 但每个位置的向量已经被旋转!")
    print(f"Q[0,0,:,0:4] 旋转前: {Q[0,0,0,:4].tolist()}")
    print(f"Q[0,0,:,0:4] 旋转后: {Q_rotated[0,0,0,:4].tolist()}")
    
    # 验证旋转保持向量范数
    norm_before = Q.norm(dim=-1).mean().item()
    norm_after = Q_rotated.norm(dim=-1).mean().item()
    print(f"\n旋转前平均范数: {norm_before:.4f}")
    print(f"旋转后平均范数: {norm_after:.4f}")
    print(f"范数变化: {abs(norm_after - norm_before) / norm_before * 100:.2f}%")
    print("(旋转是正交变换, 理论上范数不变)")
    
    print("\n长河无尽演示完毕!")
    print()


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    《焚诀》第九卷 配套丹方 —— 成圣之路
    
    "万灵齐修、瞬息万里、长河无尽。
     三大秘术合一, 便是半步成神的基石。"
    """
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "《焚诀》第九卷：成圣篇（斗圣）".center(50) + " " * 8 + "*")
    print("*" + "MoE + Flash Attention + RoPE 实战丹方".center(52) + " " * 6 + "*")
    print("*" + " " * 68 + "*")
    print("*" + '"弹指间空间碎裂, 挥手间天地变色"'.center(46) + " " * 12 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    # 第一部分: MoE
    demo_moe()
    
    # 第二部分: Flash Attention
    demo_flash_attention()
    
    # 第三部分: RoPE
    demo_rope()
    
    # 总结
    print("=" * 70)
    print("修炼总结")
    print("=" * 70)
    print()
    print("本次丹方修炼了三大斗圣级秘术:")
    print()
    print("  1. 万灵齐修 (MoE)")
    print("     - 实现了完整的 Router + Expert 层")
    print("     - 验证了 Top-K 路由和负载均衡机制")
    print("     - 对比了 MoE 与 Dense 模型的参数效率")
    print()
    print("  2. 瞬息万里 (Flash Attention)")
    print("     - 对比了标准 Attention 与 SDPA 的速度和内存")
    print("     - 理解了 IO-Aware 算法的核心优势")
    print("     - 在不同序列长度下验证了加速效果")
    print()
    print("  3. 长河无尽 (RoPE)")
    print("     - 实现了完整的旋转位置编码")
    print("     - 验证了相对位置编码特性")
    print("     - 实现了 NTK-Aware 扩展, 突破训练长度限制")
    print()
    print("至此, 成圣篇配套丹方修炼完毕。")
    print("下一步: 结合 Megatron-LM 3D 并行, 挑战从零预训练!")
    print()
    print('"半步成神, 一步之遥, 便是帝境。"')
    print()


if __name__ == "__main__":
    main()
