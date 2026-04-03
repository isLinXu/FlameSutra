"""
╔══════════════════════════════════════════════════════════════════╗
║          焚 诀 · 第五卷 · 化翼篇（斗王）                         ║
║          分布式训练实战 —— 异火共鸣，斗气化翼                      ║
╚══════════════════════════════════════════════════════════════════╝

修炼心法：
    当单株异火（单卡 GPU）的力量不足以驾驭万钧斗气（大模型）时，
    修炼者需学会异火共鸣之术 —— 令多株异火协同运作，共同炼丹。

    此丹方融合了三大功法：
    1. 混元真气（混合精度训练）—— 以 FP16/BF16 压缩斗气，降低显存消耗
    2. 聚火成阵（分布式数据并行 DDP）—— 多株异火各自持有斗气副本，协同修炼
    3. 梯度聚灵（梯度累加）—— 多步聚少成多，等效大批次训练

    此外，附有深渊之力（DeepSpeed）的配置示例。

运行方式：
    单卡模式:
        python 05_分布式训练实战.py

    多卡 DDP 模式（以 4 卡为例）:
        torchrun --nproc_per_node=4 05_分布式训练实战.py --ddp

    DeepSpeed 模式:
        deepspeed 05_分布式训练实战.py --deepspeed --deepspeed_config ds_config.json

所需灵材：
    pip install torch torchvision transformers accelerate deepspeed
"""

import os
import sys
import json
import math
import time
import argparse
import tempfile
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════════════════
# 第一节：丹方参数解析 —— 从命令行接收炼丹指令
# ═══════════════════════════════════════════════════════════════

def parse_args():
    """解析修炼者传入的丹方参数"""
    parser = argparse.ArgumentParser(
        description="焚诀 · 分布式训练实战 —— 异火共鸣，斗气化翼"
    )

    # ------ 核心模式选择 ------
    parser.add_argument(
        "--ddp", action="store_true",
        help="启用聚火成阵（DDP 分布式训练）模式"
    )
    parser.add_argument(
        "--deepspeed", action="store_true",
        help="启用深渊之力（DeepSpeed）模式"
    )
    parser.add_argument(
        "--deepspeed_config", type=str, default=None,
        help="DeepSpeed 配置文件路径"
    )

    # ------ 炼丹参数 ------
    parser.add_argument("--epochs", type=int, default=5, help="炼丹轮次")
    parser.add_argument("--batch_size", type=int, default=32, help="每炉药材数量")
    parser.add_argument("--lr", type=float, default=1e-3, help="火候（学习率）")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="梯度聚灵步数")
    parser.add_argument("--amp", action="store_true", default=True, help="启用混元真气（混合精度）")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # ------ 模型参数 ------
    parser.add_argument("--hidden_dim", type=int, default=512, help="斗气经脉宽度")
    parser.add_argument("--num_layers", type=int, default=8, help="经脉层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")

    args = parser.parse_args()

    # 禁用 AMP 的开关
    if args.no_amp:
        args.amp = False

    return args


# ═══════════════════════════════════════════════════════════════
# 第二节：分布式初始化 —— 聚火成阵的准备仪式
# ═══════════════════════════════════════════════════════════════

def setup_distributed(args) -> Tuple[int, int, torch.device]:
    """
    聚火成阵 —— 初始化分布式训练环境

    DDP 运作原理：
        1. 每株异火（GPU）运行一个独立进程（rank）
        2. 每个进程持有模型的完整副本
        3. 每个进程处理不同的数据子集（DistributedSampler）
        4. 反向传播后，所有进程同步梯度（all-reduce）
        5. 因此每个进程的模型参数始终保持一致

    返回:
        rank:       当前进程编号（0 为主进程）
        world_size: 总进程数（= 总 GPU 数）
        device:     当前进程对应的异火
    """
    if args.ddp:
        # DDP 模式 —— 由 torchrun 启动，自动设置环境变量
        # RANK: 全局进程编号
        # LOCAL_RANK: 本机 GPU 编号
        # WORLD_SIZE: 总进程数
        if "RANK" not in os.environ:
            raise RuntimeError(
                "[阵法失败] 未检测到 torchrun 环境变量！\n"
                "请使用以下命令启动:\n"
                "  torchrun --nproc_per_node=<GPU数> 05_分布式训练实战.py --ddp"
            )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # 初始化进程组 —— 建立异火之间的通信通道
        torch.distributed.init_process_group(
            backend="nccl",     # NCCL: NVIDIA 优化的 GPU 通信库（最快）
                                # 备选 "gloo"（CPU/跨平台）或 "mpi"
        )

        # 每个进程绑定对应的异火
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"[聚火成阵] 成功组阵！共 {world_size} 株异火协同")
            print(f"[通信后端] NCCL —— NVIDIA 集群通信库")

        return rank, world_size, device

    else:
        # 单卡模式 —— 独行侠
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"[单火模式] 使用异火: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[凡火模式] 未检测到异火，使用 CPU 运行（极慢）")

        return 0, 1, device


def cleanup_distributed(args):
    """散阵收功 —— 清理分布式训练资源"""
    if args.ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        print("[散阵] 异火阵法已安全解散")


# ═══════════════════════════════════════════════════════════════
# 第三节：修炼药材 —— 模拟训练数据集
# ═══════════════════════════════════════════════════════════════

class SyntheticDataset(Dataset):
    """
    合成药材库 —— 用于演示的模拟数据集

    在实战中，应替换为真实药材：
    - 文本数据: 使用 HuggingFace datasets 库加载
    - 图像数据: 使用 torchvision.datasets 加载
    - 自定义数据: 从文件系统或数据库加载
    """
    def __init__(self, num_samples: int = 10000, seq_len: int = 128, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        print(f"[药材准备] 合成药材 {num_samples} 份，序列长度 {seq_len}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机 token 序列作为模拟输入
        # 实战中这里返回真实的 tokenized 文本
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()  # 语言模型：labels = input_ids (shifted inside model)
        return {"input_ids": input_ids, "labels": labels}


# ═══════════════════════════════════════════════════════════════
# 第四节：斗气本体 —— 简化版 Transformer 模型
# ═══════════════════════════════════════════════════════════════

class MiniTransformer(nn.Module):
    """
    简化版 Transformer —— 用于演示分布式训练的斗气本体

    结构：
        Embedding -> N × TransformerBlock -> LayerNorm -> LM Head

    注意：这是教学用的简化模型。实战中请使用
    HuggingFace Transformers 提供的预训练模型。
    """
    def __init__(self, vocab_size=32000, hidden_dim=512, num_layers=8,
                 num_heads=8, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 词嵌入 —— 将 token 转化为斗气向量
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer 层 —— 斗气经脉
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN: 更稳定的训练
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 输出层 —— 将斗气映射回词汇空间
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # 权重共享：lm_head 与 token_embedding 共享权重（减少参数量）
        self.lm_head.weight = self.token_embedding.weight

        # 初始化权重
        self._init_weights()

        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[斗气凝形] MiniTransformer 就绪")
        print(f"  参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
        print(f"  经脉层数: {num_layers}, 注意力头数: {num_heads}")
        print(f"  经脉宽度: {hidden_dim}, FFN 宽度: {hidden_dim * 4}")

    def _init_weights(self):
        """Xavier 初始化 —— 斗气均衡分布"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # 嵌入 —— token 化为斗气
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # 因果掩码 —— 禁止偷看未来（自回归约束）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=x.dtype
        )

        # Transformer 前向传播 —— 斗气在经脉中流转
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)

        # 输出 logits
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 计算交叉熵损失 —— 度量斗气与真意的偏差
            # shift: 用前 n-1 个 token 预测后 n-1 个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


# ═══════════════════════════════════════════════════════════════
# 第五节：显存探查 —— 知己知彼，百战不殆
# ═══════════════════════════════════════════════════════════════

def print_gpu_memory(prefix: str = ""):
    """探查异火消耗 —— 显存使用情况"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[显存探查] {prefix}")
    print(f"  已分配: {allocated:.2f} GB | 已预留: {reserved:.2f} GB | 峰值: {max_allocated:.2f} GB")


def estimate_memory(model: nn.Module, batch_size: int, seq_len: int, precision: str = "fp16"):
    """
    预估炼丹所需显存 —— 知道需要多大的丹炉

    显存组成：
    1. 模型参数 (Parameters)
    2. 梯度 (Gradients)  —— 与参数等大
    3. 优化器状态 (Optimizer States) —— Adam: 2倍参数量（FP32）
    4. 激活值 (Activations) —— 取决于 batch_size, seq_len, hidden_dim
    """
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if precision in ("fp16", "bf16") else 4

    param_mem = param_count * bytes_per_param / 1024**3
    grad_mem = param_mem  # 梯度与参数等大
    # Adam 状态: momentum (FP32) + variance (FP32) = 8 bytes/param
    optim_mem = param_count * 8 / 1024**3

    print(f"\n[显存预估] 精度: {precision.upper()}")
    print(f"  模型参数:   {param_mem:.2f} GB")
    print(f"  梯度:       {grad_mem:.2f} GB")
    print(f"  优化器状态: {optim_mem:.2f} GB (Adam FP32)")
    print(f"  小计:       {param_mem + grad_mem + optim_mem:.2f} GB (不含激活值)")
    print(f"  [提示] 激活值显存取决于 batch_size={batch_size}, seq_len={seq_len}")
    print(f"  [提示] 实际显存 ≈ 小计 × 1.2~1.5（含激活值和碎片）")


# ═══════════════════════════════════════════════════════════════
# 第六节：深渊之力 —— DeepSpeed 配置
# ═══════════════════════════════════════════════════════════════

def get_deepspeed_config(args) -> dict:
    """
    构建深渊之力配置 —— DeepSpeed ZeRO Stage 2

    ZeRO 三阶段详解：
    - Stage 1: 分割优化器状态（每卡只存 1/N 的 Adam 状态）
    - Stage 2: 分割优化器状态 + 梯度（进一步减少显存）
    - Stage 3: 分割一切：参数 + 梯度 + 优化器状态（极致节省，但通信开销大）

    此处使用 Stage 2 作为起点，兼顾显存节省与训练速度。
    """
    ds_config = {
        # ---- 训练批次配置 ----
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,

        # ---- ZeRO 优化 Stage 2 ----
        # 聚火阵法的核心：将优化器状态和梯度分散到各株异火
        "zero_optimization": {
            "stage": 2,                          # ZeRO Stage 2
            "offload_optimizer": {               # 优化器状态卸载至 CPU（可选，进一步省显存）
                "device": "none",                # "cpu" 可启用 CPU offload
                "pin_memory": True,
            },
            "allgather_partitions": True,        # 高效梯度聚合
            "allgather_bucket_size": 2e8,        # 通信桶大小（字节）
            "overlap_comm": True,                # 通信与计算重叠
            "reduce_scatter": True,              # 使用 reduce_scatter 替代 all_reduce
            "reduce_bucket_size": 2e8,           # 梯度规约桶大小
            "contiguous_gradients": True,        # 连续梯度存储
        },

        # ---- 混合精度配置 ----
        # 混元真气 —— BF16 模式
        "bf16": {
            "enabled": True,
        },

        # ---- 梯度裁剪 ----
        "gradient_clipping": 1.0,

        # ---- 优化器 ----
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },

        # ---- 学习率调度器 ----
        "scheduler": {
            "type": "WarmupCosineWithMinLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "total_num_steps": 1000,
                "min_lr": 1e-6,
            },
        },

        # ---- 日志 ----
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    return ds_config


def write_deepspeed_config(args) -> str:
    """将 DeepSpeed 配置写入临时文件"""
    config = get_deepspeed_config(args)
    config_path = os.path.join(tempfile.gettempdir(), "ds_config_flamesutra.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[深渊配置] DeepSpeed 配置已写入: {config_path}")
    return config_path


# ═══════════════════════════════════════════════════════════════
# 第七节：核心炼丹循环 —— 融合所有功法
# ═══════════════════════════════════════════════════════════════

def train(args):
    """
    核心炼丹循环 —— 融合混合精度、DDP、梯度累加

    此函数是整个丹方的核心，展示了如何将
    混元真气、聚火成阵、梯度聚灵三大功法融为一体。
    """
    # ---- 1. 聚火准备 ----
    rank, world_size, device = setup_distributed(args)
    is_main_process = (rank == 0)

    # 设定随机种子 —— 保证修炼可复现
    torch.manual_seed(args.seed + rank)  # 每个进程使用不同种子以获取不同数据
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- 2. 铸造斗气本体（模型） ----
    if is_main_process:
        print("\n" + "=" * 60)
        print("  [铸造斗气] 构建 MiniTransformer...")
        print("=" * 60)

    model = MiniTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    # 显存预估
    if is_main_process:
        estimate_memory(model, args.batch_size, 128, "bf16" if args.amp else "fp32")
        print_gpu_memory("模型加载后")

    # ---- 3. 聚火成阵（DDP 包装） ----
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=False,  # 若所有参数都参与计算，设为 False 提升性能
        )
        if is_main_process:
            print("[聚火成阵] 模型已包装为 DistributedDataParallel")

    # ---- 4. 准备药材（数据集和数据加载器） ----
    dataset = SyntheticDataset(num_samples=2000, seq_len=128)

    # DDP 模式下使用 DistributedSampler —— 每株异火处理不同的药材
    sampler = None
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if is_main_process:
            print(f"[药材分配] DistributedSampler 就绪，每株异火处理 {len(dataset)//world_size} 份药材")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),       # 有 sampler 时不需要 shuffle
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 5. 配置优化器 —— 调节火候 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # 余弦退火调度器
    total_steps = len(dataloader) * args.epochs // args.grad_accum_steps
    warmup_steps = min(100, total_steps // 10)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )

    # ---- 6. 混元真气准备（混合精度） ----
    # AMP: Automatic Mixed Precision
    # autocast: 自动将运算转为低精度（FP16/BF16）
    # GradScaler: 梯度缩放，防止 FP16 下梯度下溢（BF16 不需要）
    use_amp = args.amp and torch.cuda.is_available()
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # GradScaler: 仅 FP16 需要，BF16 动态范围足够大，不需要
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    if is_main_process:
        if use_amp:
            print(f"[混元真气] 已启用混合精度训练: {amp_dtype}")
            if not use_bf16:
                print("[梯度缩放] GradScaler 已启用（FP16 模式）")
            else:
                print("[梯度缩放] BF16 模式无需 GradScaler")
        else:
            print("[全精度] 使用 FP32 全精度训练")

    # ---- 7. 计算等效批次 ----
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    if is_main_process:
        print(f"\n[炼丹参数总览]")
        print(f"  每卡批次:     {args.batch_size}")
        print(f"  梯度聚灵步数: {args.grad_accum_steps}")
        print(f"  异火数量:     {world_size}")
        print(f"  等效批次:     {args.batch_size} x {args.grad_accum_steps} x {world_size} = {effective_batch}")
        print(f"  训练轮次:     {args.epochs}")
        print(f"  总步数:       ~{total_steps}")

    # ═══════════════════════════════════════════════════════════
    # 核心炼丹循环 —— 斗气在经脉中反复流转
    # ═══════════════════════════════════════════════════════════

    if is_main_process:
        print("\n" + "=" * 60)
        print("  [丹炉点火] 炼丹正式开始！")
        print("=" * 60 + "\n")

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        # DDP: 每个 epoch 需要重新设置 sampler 的随机种子
        # 确保每个 epoch 数据顺序不同，但所有进程一致
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            # 将药材投入丹炉（移至 GPU）
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # ---- 混元真气 + 前向传播 ----
            # autocast 上下文管理器自动将合适的运算转为低精度
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

                # 梯度聚灵：损失需除以累加步数以保持梯度量级一致
                # 若不除，累加后的梯度会比单步大 grad_accum_steps 倍
                loss = loss / args.grad_accum_steps

            # ---- 反向传播 ----
            # GradScaler: 放大 loss 以防止 FP16 梯度下溢
            scaler.scale(loss).backward()

            # ---- 梯度聚灵：累积多步后再更新 ----
            if (step + 1) % args.grad_accum_steps == 0:
                # 梯度裁剪 —— 防止斗气暴走（梯度爆炸）
                # 需要先 unscale，才能正确裁剪
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                # 更新参数 —— 斗气精进
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # set_to_none=True 更高效

                # 学习率调度
                if global_step >= warmup_steps:
                    scheduler.step()

                global_step += 1

            # 记录损失
            epoch_loss += loss.item() * args.grad_accum_steps
            num_batches += 1

            # 定期输出炼丹状况
            if is_main_process and (step + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / num_batches
                print(
                    f"  [Epoch {epoch+1}/{args.epochs}] "
                    f"Step {step+1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad Norm: {grad_norm:.2f}"
                )

        # ---- Epoch 结束统计 ----
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_batches, 1)

        # DDP 模式下，同步各进程的 loss（求平均）
        if args.ddp:
            loss_tensor = torch.tensor([avg_epoch_loss], device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_epoch_loss = loss_tensor.item()

        if is_main_process:
            throughput = len(dataset) / epoch_time
            print(f"\n  === Epoch {epoch+1} 完成 ===")
            print(f"  平均 Loss: {avg_epoch_loss:.4f}")
            print(f"  耗时: {epoch_time:.1f}s | 吞吐: {throughput:.0f} samples/s")
            print_gpu_memory(f"Epoch {epoch+1} 结束")
            print()

    # ═══════════════════════════════════════════════════════════
    # 炼丹完成 —— 保存丹药
    # ═══════════════════════════════════════════════════════════

    if is_main_process:
        print("=" * 60)
        print("  [丹成！] 训练完成")
        print("=" * 60)

        # 保存模型（DDP 模式下需要取出原始模型）
        save_model = model.module if args.ddp else model
        save_path = "./distributed_model_output"
        os.makedirs(save_path, exist_ok=True)
        torch.save(save_model.state_dict(), os.path.join(save_path, "model.pt"))
        print(f"[收丹] 模型已保存至: {save_path}/model.pt")

        print_gpu_memory("最终")

    # 清理
    cleanup_distributed(args)


# ═══════════════════════════════════════════════════════════════
# 第八节：DeepSpeed 炼丹流程
# ═══════════════════════════════════════════════════════════════

def train_with_deepspeed(args):
    """
    深渊之力 —— 使用 DeepSpeed 引擎炼丹

    DeepSpeed 是微软开发的深度学习优化库，提供：
    - ZeRO: 极致的显存优化
    - 混合精度训练
    - 梯度累加
    - CPU/NVMe Offload
    - 通信优化

    相比手写 DDP + AMP，DeepSpeed 将这些功法打包，
    修炼者只需一份配置文件即可驾驭。
    """
    try:
        import deepspeed
    except ImportError:
        print("[深渊之力] DeepSpeed 未安装！请执行: pip install deepspeed")
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (rank == 0)

    if is_main_process:
        print("\n" + "=" * 60)
        print("  [深渊之力] DeepSpeed 训练模式")
        print("=" * 60)

    # 构建模型
    model = MiniTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    # 准备 DeepSpeed 配置
    if args.deepspeed_config:
        config_path = args.deepspeed_config
    else:
        config_path = write_deepspeed_config(args)

    # 准备数据
    dataset = SyntheticDataset(num_samples=2000, seq_len=128)

    # DeepSpeed 初始化 —— 一键布阵
    # deepspeed.initialize 会自动处理：
    # - 模型分布式包装
    # - 优化器创建（若配置中指定）
    # - 混合精度设置
    # - ZeRO 分片策略
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config_path,
    )

    device = model_engine.local_rank

    dataloader = DataLoader(
        dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if is_main_process:
        print(f"[深渊布阵] DeepSpeed 引擎就绪")
        print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")
        print(f"  微批次大小: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"  梯度累加步: {model_engine.gradient_accumulation_steps()}")

    # 炼丹循环
    model_engine.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # DeepSpeed 自动处理混合精度和梯度累加
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # DeepSpeed 的 backward —— 自动处理梯度缩放
            model_engine.backward(loss)

            # DeepSpeed 的 step —— 自动处理梯度累加和优化器更新
            model_engine.step()

            epoch_loss += loss.item()

            if is_main_process and (step + 1) % 10 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"  [DeepSpeed] Epoch {epoch+1} Step {step+1} | Loss: {avg_loss:.4f}")

        if is_main_process:
            print(f"  === DeepSpeed Epoch {epoch+1} 完成 | Loss: {epoch_loss / len(dataloader):.4f} ===\n")

    # 保存
    if is_main_process:
        save_path = "./deepspeed_model_output"
        os.makedirs(save_path, exist_ok=True)
        model_engine.save_checkpoint(save_path)
        print(f"[深渊收功] 模型检查点已保存至: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 第九节：主控丹方 —— 入口函数
# ═══════════════════════════════════════════════════════════════

def main():
    """
    主控丹方 —— 根据命令行参数选择修炼路径

    三种修炼路径：
    1. 单火修炼（单卡）: python 05_分布式训练实战.py
    2. 聚火成阵（DDP）:  torchrun --nproc_per_node=4 05_分布式训练实战.py --ddp
    3. 深渊之力（DeepSpeed）: deepspeed 05_分布式训练实战.py --deepspeed
    """
    args = parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   焚 诀 · 分布式训练实战                                  ║
    ║   Flame Sutra · Distributed Training Practice            ║
    ║                                                          ║
    ║   异火共鸣，斗气化翼                                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    if args.deepspeed:
        print("[修炼路径] 深渊之力 —— DeepSpeed 模式")
        train_with_deepspeed(args)
    elif args.ddp:
        print("[修炼路径] 聚火成阵 —— DDP 分布式模式")
        train(args)
    else:
        print("[修炼路径] 单火修炼 —— 单卡模式")
        train(args)

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   修炼完毕！                                              ║
    ║                                                          ║
    ║   你已掌握：                                              ║
    ║   - 混元真气（混合精度训练）                                ║
    ║   - 聚火成阵（DDP 分布式数据并行）                          ║
    ║   - 梯度聚灵（梯度累加）                                   ║
    ║   - 深渊之力（DeepSpeed ZeRO）                            ║
    ║                                                          ║
    ║   下一步：第六卷·凌空篇 —— 更高层次的修炼                   ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
