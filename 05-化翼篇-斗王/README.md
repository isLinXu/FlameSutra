# 第五卷：化翼篇（斗王）— 斗气化翼

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   焚 诀 · 第五卷                                                     ║
║                                                                      ║
║   化 翼 篇 （ 斗 王 ）                                                ║
║                                                                      ║
║   "斗气化翼，翱翔九天"                                                 ║
║                                                                      ║
║   当单株异火之力已不足以驾驭你的修为，                                   ║
║   你需要学会——异火共鸣。                                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 开篇引言

修炼至此，你已历经筑基、纳灵、凝晶、化形四重境界。你掌握了 Python 与 PyTorch 的基础功法（筑基），学会了加载与处理药材（纳灵），理解了斗气经脉的运作原理——Transformer 架构（凝晶），更在上一卷中习得了 LoRA 这等以小搏大的精妙功法（化形）。

然而，直到此刻，你始终被困于一个根本性的桎梏之中——

**你只有一株异火。**

一张 GPU 显卡，一个计算节点，一份有限的显存。当你试图驾驭更庞大的斗气（更大的模型），当你需要在更短的时间内完成炼丹（训练），当你的药材库（数据集）膨胀到单卡无法承载的规模——你会撞上一堵无形的墙：

```
CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 24.00 GiB, of which 512.00 MiB is free.
```

**反噬。**

这是每个修炼者的必经之痛。在玄幻世界中，这如同一个飞不起来的斗王——你的斗气足以称王，但缺少化翼飞升的关键功法。

本卷将教你突破这一桎梏。你将学会：

- **洞悉异火本质**：显存到底被什么吞噬了？如何精确预估所需显存？
- **混元真气（混合精度训练）**：用 FP16/BF16 压缩斗气，以一半的显存驾驭同样的力量
- **聚火成阵（分布式数据并行 DDP）**：令多株异火协同运作，力量倍增
- **梯度聚灵（梯度累加）**：以时间换空间，模拟大批次训练
- **深渊之力（DeepSpeed）**：来自深渊的禁忌力量，将显存优化推至极致

学成此卷，你的斗气将凝聚成翼，突破单机单卡的束缚，翱翔于多卡并行的广阔天际。

---

## 目录

- [第一章：异火共鸣 — 显存与计算的本质](#第一章异火共鸣--显存与计算的本质)
- [第二章：混元真气 — 混合精度训练](#第二章混元真气--混合精度训练-mixed-precision)
- [第三章：聚火成阵 — 分布式数据并行 DDP](#第三章聚火成阵--分布式数据并行-ddp)
- [第四章：梯度聚灵 — 梯度累加](#第四章梯度聚灵--梯度累加)
- [第五章：深渊之力 — DeepSpeed 入门](#第五章深渊之力--deepspeed-入门)
- [第六章：化翼飞升 — 多卡训练实战](#第六章化翼飞升--多卡训练实战)
- [修炼总结](#修炼总结)

---

## 第一章：异火共鸣 — 显存与计算的本质

> *"知己知彼，百战不殆。欲驾驭异火，先要理解异火。"*

### 1.1 异火的本质：GPU 显存解剖

每株异火（GPU）的力量由两个维度决定：

1. **算力**：每秒能执行多少次浮点运算（FLOPS）—— 这决定了炼丹的速度
2. **显存**：能同时容纳多少数据（GB）—— 这决定了能炼多大的丹

绝大多数修炼者遭遇的"反噬"（OOM），都是因为第二个维度——显存不足。

那么，炼丹时显存到底被什么占据了？答案是四大组成部分：

```
┌─────────────────────────────────────────────────┐
│                GPU 显存（VRAM）                   │
│                                                   │
│  ┌───────────────┐  ┌───────────────────────┐    │
│  │ 模型参数       │  │ 梯度（Gradients）      │    │
│  │ (Parameters)  │  │ 与参数等大             │    │
│  │               │  │                       │    │
│  └───────────────┘  └───────────────────────┘    │
│                                                   │
│  ┌───────────────────────────────────────────┐    │
│  │ 优化器状态（Optimizer States）              │    │
│  │ Adam: 每个参数需额外 8 字节（FP32 下）      │    │
│  │ = 一阶矩估计(4B) + 二阶矩估计(4B)          │    │
│  └───────────────────────────────────────────┘    │
│                                                   │
│  ┌───────────────────────────────────────────┐    │
│  │ 激活值（Activations）                       │    │
│  │ 随 batch_size, seq_len, hidden_dim 增长     │    │
│  │ 注意力层: O(batch × heads × seq_len²)       │    │
│  └───────────────────────────────────────────┘    │
│                                                   │
│  ┌───────────────┐                                │
│  │ 临时缓冲区    │ ← CUDA 内存碎片、通信缓冲等     │
│  └───────────────┘                                │
└─────────────────────────────────────────────────┘
```

### 1.2 显存精确估算公式

让我们以具体的数字来理解。假设我们要训练一个 **7B 参数**的模型：

#### 模型参数（Parameters）

每个参数在不同精度下占用的字节数：

| 精度 | 每参数字节数 | 7B 模型参数显存 |
|------|------------|----------------|
| FP32 | 4 bytes | 7B × 4 = **28 GB** |
| FP16 / BF16 | 2 bytes | 7B × 2 = **14 GB** |
| INT8 | 1 byte | 7B × 1 = **7 GB** |
| INT4 | 0.5 bytes | 7B × 0.5 = **3.5 GB** |

#### 梯度（Gradients）

梯度与参数一一对应，精度相同则显存等大：

- FP32 训练：7B × 4 bytes = **28 GB**
- FP16 混合精度：梯度通常以 FP16 存储 = 7B × 2 = **14 GB**

#### 优化器状态（Optimizer States）

这是显存消耗的"隐藏大户"。不同优化器的额外开销：

| 优化器 | 每参数额外字节数 | 7B 模型额外显存 |
|--------|----------------|----------------|
| SGD | 0 bytes（无状态） | 0 GB |
| SGD + Momentum | 4 bytes（动量，FP32） | 28 GB |
| Adam / AdamW (FP32) | 8 bytes（一阶矩 + 二阶矩） | **56 GB** |
| Adam (混合精度) | 8 bytes（状态仍为 FP32）+ 4B 主权重 | **84 GB**¹ |

> ¹ 混合精度 Adam 的完整开销：FP32 主权重副本(28GB) + 一阶矩(28GB) + 二阶矩(28GB) = 84GB。
> 这就是为什么混合精度训练虽然降低了参数和梯度的显存，但优化器状态仍然巨大。

#### 激活值（Activations）

激活值是前向传播中间结果，需要保留以供反向传播使用。对于 Transformer：

```
激活值显存 ≈ batch_size × seq_len × hidden_dim × num_layers × 精度字节数 × 常数因子
```

具体来说，对于一个标准 Transformer 层，主要激活值包括：
- 注意力分数矩阵: `batch × heads × seq_len × seq_len` —— 这是 O(seq_len²) 的来源
- 各线性层输入/输出: `batch × seq_len × hidden_dim`
- LayerNorm 的中间值
- FFN 的中间激活（通常是 `4 × hidden_dim`）

#### 总显存估算（7B 模型，FP16 混合精度，Adam）

```
模型参数 (FP16):            14 GB
梯度 (FP16):                14 GB
优化器状态 (FP32):
  - FP32 主权重副本:         28 GB
  - 一阶矩估计:             28 GB
  - 二阶矩估计:             28 GB
激活值 (估):                10~30 GB（取决于 batch_size）
临时缓冲:                   ~2 GB
─────────────────────────────────
总计:                       ~124~144 GB
```

**这就是为什么 7B 模型在单张 24GB 显卡上根本无法全参数训练！**

就算你只有 1.5B 的小模型，全参数训练也需要约 30~40 GB 显存。理解这个现实，是突破的第一步。

### 1.3 深入理解反噬（OOM）

OOM 不是简单的"显存不够"。理解其深层原因才能对症下药：

**原因一：显存碎片化**

PyTorch 使用缓存式内存分配器。频繁的分配和释放会导致碎片化——总显存虽然够，但找不到足够大的连续块。

```python
# 查看显存碎片信息
print(torch.cuda.memory_summary())
# 关注 "Allocated memory" vs "Reserved memory" 的差异
# 差异越大，碎片化越严重
```

**原因二：峰值显存尖峰**

某些操作会产生短暂但巨大的显存需求。例如：
- 大矩阵乘法的中间结果
- 注意力分数的完整矩阵（seq_len × seq_len）
- 损失函数计算时的 logits 展平（vocab_size 通常很大）

**原因三：隐式的显存泄漏**

```python
# 错误！loss_history 持有计算图引用，显存不断累积
loss_history = []
for batch in dataloader:
    loss = model(batch)
    loss_history.append(loss)        # 保留了整个计算图！

# 正确做法：只保留数值
loss_history.append(loss.item())     # .item() 取出标量，释放计算图
```

### 1.4 梯度检查点：以时间换空间

梯度检查点（Gradient Checkpointing）是一种经典的显存优化技术。其核心思想：

**正常训练**：前向传播时保存所有中间激活值，反向传播时直接使用。
**梯度检查点**：只保存少数"检查点"处的激活值，反向传播时重新计算被丢弃的中间值。

```
正常模式（保存所有激活值）：
Layer1 → [save] → Layer2 → [save] → Layer3 → [save] → Layer4 → [save] → Loss
                                                                          ↓
Layer1 ← [use]  ← Layer2 ← [use]  ← Layer3 ← [use]  ← Layer4 ← [use]  ← Backward

检查点模式（只保存部分）：
Layer1 → [save] → Layer2 → [drop] → Layer3 → [save] → Layer4 → [drop] → Loss
                                                                          ↓
Layer1 ← [use]  ← Layer2 ← [recompute] ← Layer3 ← [use] ← Layer4 ← [recompute] ← Backward
```

显存节省：激活值显存从 O(N) 降至 O(√N)（N 为层数）
代价：约 30% 的额外计算时间（需要重新前向传播部分层）

```python
# PyTorch 原生梯度检查点
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # 不使用检查点（正常模式）
        x = self.attention(x)
        x = self.ffn(x)
        return x

    def forward_with_checkpoint(self, x):
        # 使用检查点（以时间换显存）
        x = checkpoint(self.attention, x, use_reentrant=False)
        x = checkpoint(self.ffn, x, use_reentrant=False)
        return x

# HuggingFace 模型一键启用
model.gradient_checkpointing_enable()
```

### 1.5 代码：显存分析实战

```python
import torch
import torch.nn as nn

def memory_profiling_demo():
    """
    异火探查术 —— 显存分析实战

    学会这门探查术，你就能精确知道每一步操作消耗了多少显存，
    从而有针对性地优化。
    """
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    def print_mem(tag):
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{tag}] 当前: {alloc:.2f} GB | 峰值: {peak:.2f} GB")

    print("=== 显存分析开始 ===\n")
    print_mem("初始状态")

    # 1. 加载模型参数
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
    ).to(device)
    print_mem("模型加载")

    # 2. 创建优化器（Adam 状态会额外占用显存）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print_mem("优化器创建")

    # 3. 前向传播（产生激活值）
    x = torch.randn(64, 4096, device=device)
    y = model(x)
    print_mem("前向传播后")

    # 4. 计算损失 + 反向传播（产生梯度）
    loss = y.sum()
    loss.backward()
    print_mem("反向传播后")

    # 5. 优化器更新（首次更新会创建 Adam 状态）
    optimizer.step()
    print_mem("优化器更新后")

    # 6. 清理梯度
    optimizer.zero_grad(set_to_none=True)
    print_mem("清理梯度后")

    print("\n=== 显存分析完成 ===")
    print(f"\n完整显存报告:\n{torch.cuda.memory_summary()}")

# 取消注释以运行:
# memory_profiling_demo()
```

---

## 第二章：混元真气 — 混合精度训练 (Mixed Precision)

> *"以混元真气之法，将斗气在 FP32 与 FP16 之间自如转换，*
> *既保持修炼的稳定，又大幅降低异火的消耗。"*

### 2.1 数值精度：FP32 vs FP16 vs BF16

在深度学习的世界里，数字的表示方式直接影响显存消耗和计算速度。

#### FP32（单精度浮点数）—— 全精度斗气

```
符号位(1) | 指数位(8) | 尾数位(23)
   S        EEEEEEEE    MMMMMMMMMMMMMMMMMMMMMMM

- 范围: ±3.4 × 10³⁸
- 精度: ~7 位有效数字
- 大小: 4 字节/参数
```

FP32 是传统训练的标准精度。精度高、范围大、但显存消耗也大。

#### FP16（半精度浮点数）—— 压缩斗气

```
符号位(1) | 指数位(5) | 尾数位(10)
   S        EEEEE       MMMMMMMMMM

- 范围: ±65504（很小！）
- 精度: ~3.3 位有效数字
- 大小: 2 字节/参数 → 显存减半！
```

FP16 的优势明显：显存减半、计算速度提升（GPU 的 Tensor Core 对 FP16 有硬件加速）。但其致命弱点是**动态范围太小**——很容易发生：
- **上溢（Overflow）**：梯度过大，超出 65504 → 变成 `inf`
- **下溢（Underflow）**：梯度过小，低于 2⁻²⁴ ≈ 5.96 × 10⁻⁸ → 变成 0

当梯度变成 0，参数就不再更新——训练静默失败。这就是为什么纯 FP16 训练不可靠。

#### BF16（Brain Floating Point）—— 最优解

```
符号位(1) | 指数位(8) | 尾数位(7)
   S        EEEEEEEE    MMMMMMM

- 范围: ±3.4 × 10³⁸（与 FP32 相同！）
- 精度: ~2.4 位有效数字（比 FP16 低）
- 大小: 2 字节/参数
```

BF16 由 Google Brain 团队提出，它保留了 FP32 的指数位宽度（8 位），因此拥有与 FP32 相同的动态范围。虽然精度略低于 FP16，但在深度学习中，动态范围远比精度重要。

**BF16 的优势总结**：
- 与 FP16 同样的显存节省（2 字节/参数）
- 与 FP32 相同的动态范围（不会上溢/下溢）
- 无需 Loss Scaling（因为不会下溢）
- NVIDIA A100/H100、AMD MI250+ 等现代 GPU 原生支持

| 特性 | FP32 | FP16 | BF16 |
|------|------|------|------|
| 字节数 | 4 | 2 | 2 |
| 动态范围 | 极大 | 小 | 极大 |
| 精度 | 高 | 中 | 低 |
| 需要 Loss Scaling | 否 | **是** | 否 |
| Tensor Core 加速 | 有限 | 是 | 是 |
| 推荐度 | 基线 | 可用 | **首选** |

### 2.2 AMP：自动混合精度的运作原理

PyTorch 的 AMP（Automatic Mixed Precision）通过两个核心组件实现混合精度训练：

#### torch.amp.autocast —— 自动精度转换

`autocast` 是一个上下文管理器，它会自动决定每个操作应使用什么精度：

```
可安全使用 FP16/BF16 的操作（不敏感）：          保持 FP32 的操作（精度敏感）：
├── 矩阵乘法 (matmul)                         ├── 损失函数 (loss functions)
├── 线性层 (linear)                            ├── Softmax
├── 卷积 (conv)                                ├── LayerNorm / BatchNorm
├── BMM (batch matrix multiply)                ├── 权重更新
└── 注意力计算                                  └── 小规模归约操作
```

`autocast` 会在幕后自动插入类型转换，你的代码几乎不需要修改。

#### torch.amp.GradScaler —— 梯度缩放器（仅 FP16 需要）

Loss Scaling 的工作原理：

```
正常 FP16 训练：
  loss = 0.0001              → 梯度可能 < 5.96e-8 → 下溢为 0！

使用 Loss Scaling：
  scaled_loss = loss × 1024  → 梯度被放大 → 安全范围内
  反向传播后：
  real_grad = scaled_grad / 1024  → 恢复真实梯度
```

GradScaler 的动态缩放策略：
1. 从一个较大的 scale factor 开始（如 2¹⁶ = 65536）
2. 若干步没有出现 `inf/nan`，则加倍 scale factor（更激进地利用精度）
3. 一旦出现 `inf/nan`，将 scale factor 减半，并跳过这步更新

### 2.3 完整代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def mixed_precision_training_demo():
    """
    混元真气修炼法 —— 混合精度训练完整示例

    展示 AMP 的正确使用姿势，包括：
    - autocast 的使用范围
    - GradScaler 的完整流程
    - BF16 和 FP16 两种模式的区别
    """
    device = torch.device("cuda:0")

    # 构建一个简单模型
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, 1024),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 检测是否支持 BF16
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[混元真气] 使用精度: {amp_dtype}")

    # GradScaler：仅 FP16 需要，BF16 动态范围足够大
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))
    if not use_bf16:
        print("[梯度缩放] GradScaler 已启用（FP16 模式）")
    else:
        print("[梯度缩放] BF16 模式，无需 GradScaler")

    # 模拟数据
    x = torch.randn(1000, 1024, device=device)
    y = torch.randn(1000, 1024, device=device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_y in dataloader:

            # ==============================
            # 1. 前向传播（在 autocast 中）
            # ==============================
            # autocast 会自动将矩阵乘法等操作转为低精度
            # 但 loss 计算等精度敏感操作保持 FP32
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = model(batch_x)
                loss = nn.functional.mse_loss(pred, batch_y)

            # ==============================
            # 2. 反向传播（通过 scaler）
            # ==============================
            optimizer.zero_grad()

            # scaler.scale(loss)：将 loss 乘以 scale factor
            # .backward()：在放大后的 loss 上计算梯度
            scaler.scale(loss).backward()

            # ==============================
            # 3. 梯度裁剪（可选但推荐）
            # ==============================
            # 必须先 unscale 才能正确裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ==============================
            # 4. 参数更新
            # ==============================
            # scaler.step()：如果梯度中没有 inf/nan，执行 optimizer.step()
            #                如果有 inf/nan，跳过这步更新
            scaler.step(optimizer)

            # 更新 scale factor
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} | Loss: {avg_loss:.6f}")

    print("[混元真气] 训练完成！")

# 取消注释以运行:
# mixed_precision_training_demo()
```

### 2.4 混合精度训练的注意事项

**常见陷阱一：autocast 范围过大**

```python
# 错误：将优化器更新也放在 autocast 中
with torch.amp.autocast("cuda", dtype=torch.float16):
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()     # 权重更新不应在 autocast 中！

# 正确：只包裹前向传播和损失计算
with torch.amp.autocast("cuda", dtype=torch.float16):
    pred = model(x)
    loss = criterion(pred, y)
loss.backward()          # 反向传播可以在 autocast 外
optimizer.step()         # 优化器更新一定要在 autocast 外
```

**常见陷阱二：忘记处理 inf/nan**

FP16 训练中偶尔出现 `inf/nan` 是正常的（scaler 会自动处理），但如果频繁出现，说明：
- 学习率过大
- 模型初始化有问题
- 数据中存在异常值

**常见陷阱三：在 autocast 内手动指定精度**

```python
# 不推荐：手动 .half() 可能导致精度问题
with torch.amp.autocast("cuda"):
    x = x.half()              # 不要手动转换！让 autocast 自动决定
    pred = model(x)

# 推荐：让 autocast 自动管理精度
with torch.amp.autocast("cuda"):
    pred = model(x)            # x 保持原始精度，autocast 自动处理
```

---

## 第三章：聚火成阵 — 分布式数据并行 (DDP)

> *"一株异火虽猛，终有极限。*
> *聚百火成阵，方能焚天煮海。"*

### 3.1 DataParallel vs DistributedDataParallel

PyTorch 提供两种多卡训练方式：

#### DataParallel (DP) —— 简易聚火阵（不推荐）

```python
# 一行代码即可，但性能低下
model = nn.DataParallel(model)
```

DP 的工作方式：

```
               GPU 0 (主卡)
              ┌─────────┐
  数据 ──────→│ 分发数据  │──── 数据片段 ──→ GPU 1
              │ 收集梯度  │──── 数据片段 ──→ GPU 2
              │ 更新参数  │──── 数据片段 ──→ GPU 3
              │ 广播参数  │
              └─────────┘
```

**DP 的致命缺陷**：
1. **GIL 瓶颈**：使用多线程，受 Python GIL 限制
2. **显存不均**：GPU 0 承担额外的数据分发和梯度收集工作，显存占用远超其他卡
3. **通信开销大**：所有梯度必须先汇总到 GPU 0，再广播回去
4. **无法跨节点**：只能在单台机器的多卡之间使用

#### DistributedDataParallel (DDP) —— 真·聚火成阵（推荐）

```python
# 需要更多设置代码，但性能优异
model = DistributedDataParallel(model, device_ids=[local_rank])
```

DDP 的工作方式：

```
  进程 0 (GPU 0)         进程 1 (GPU 1)         进程 2 (GPU 2)
  ┌─────────┐           ┌─────────┐           ┌─────────┐
  │ 模型副本  │           │ 模型副本  │           │ 模型副本  │
  │ 数据子集  │           │ 数据子集  │           │ 数据子集  │
  │ 前向传播  │           │ 前向传播  │           │ 前向传播  │
  │ 反向传播  │           │ 反向传播  │           │ 反向传播  │
  └────┬─────┘           └────┬─────┘           └────┬─────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                    ┌─────────────────┐
                    │  All-Reduce 梯度  │ ← 所有进程同时参与，无主从之分
                    │  （NCCL 高速通信）  │
                    └─────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
  ┌────┴─────┐           ┌────┴─────┐           ┌────┴─────┐
  │ 更新参数  │           │ 更新参数  │           │ 更新参数  │  ← 每个进程独立更新
  └──────────┘           └──────────┘           └──────────┘     （结果完全一致）
```

**DDP 的核心优势**：
1. **多进程**：每个 GPU 一个独立进程，绕过 GIL
2. **显存均衡**：每个进程的工作完全对称
3. **All-Reduce**：梯度同步通过高效的 All-Reduce 操作完成，无瓶颈
4. **跨节点**：支持多机多卡训练
5. **通信重叠**：梯度计算与通信可以重叠进行

### 3.2 通信后端

DDP 的核心操作是 **All-Reduce**——将所有进程的梯度求平均。这需要高效的通信后端：

| 后端 | 适用场景 | 性能 | 说明 |
|------|---------|------|------|
| **NCCL** | GPU 间通信 | 最快 | NVIDIA 优化，支持 NVLink/NVSwitch |
| **Gloo** | CPU 间通信 / 备用 | 中等 | 跨平台，也支持 GPU（但慢于 NCCL） |
| **MPI** | HPC 集群 | 取决于实现 | 需要单独安装 MPI 实现 |

在绝大多数场景下，**GPU 训练用 NCCL**，无需考虑其他选项。

### 3.3 DDP 核心概念

在编写 DDP 代码之前，你需要理解几个关键概念：

**World Size**：参与训练的总进程数（通常等于总 GPU 数）
**Rank**：每个进程的全局唯一编号（0 到 world_size - 1）
**Local Rank**：每个进程在本机上的 GPU 编号

```
假设 2 台机器，每台 4 张 GPU：

机器 1:                      机器 2:
  GPU 0: rank=0, local_rank=0   GPU 0: rank=4, local_rank=0
  GPU 1: rank=1, local_rank=1   GPU 1: rank=5, local_rank=1
  GPU 2: rank=2, local_rank=2   GPU 2: rank=6, local_rank=2
  GPU 3: rank=3, local_rank=3   GPU 3: rank=7, local_rank=3

world_size = 8
```

**DistributedSampler**：确保每个进程获取不同的数据子集

```python
# 假设数据集有 1000 个样本，4 个 GPU：
# GPU 0 获得: 样本 0, 4, 8, 12, ...
# GPU 1 获得: 样本 1, 5, 9, 13, ...
# GPU 2 获得: 样本 2, 6, 10, 14, ...
# GPU 3 获得: 样本 3, 7, 11, 15, ...
# 每个 GPU 处理 250 个样本，合起来覆盖全部 1000 个
```

### 3.4 完整代码：从单卡到 DDP 的改造

下面展示如何将一个单卡训练脚本改造为支持 DDP 的版本。

#### 原始单卡代码

```python
# === 单卡版本 ===

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 模型
model = nn.Linear(1024, 1024).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 数据
x = torch.randn(1000, 1024)
y = torch.randn(1000, 1024)
dataloader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

# 训练
for epoch in range(10):
    for bx, by in dataloader:
        bx, by = bx.cuda(), by.cuda()
        loss = nn.functional.mse_loss(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
```

#### 改造后的 DDP 版本

```python
# === DDP 版本 ===
# 运行方式: torchrun --nproc_per_node=4 train_ddp.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def main():
    # ========================
    # [新增] 1. 初始化分布式环境
    # ========================
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[聚火成阵] {world_size} 株异火就位")

    # ========================
    # 2. 模型 → DDP 包装
    # ========================
    model = nn.Linear(1024, 1024).to(device)
    model = DDP(model, device_ids=[local_rank])  # [新增] DDP 包装

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ========================
    # [新增] 3. 数据 → DistributedSampler
    # ========================
    x = torch.randn(1000, 1024)
    y = torch.randn(1000, 1024)
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,       # [修改] shuffle=True → sampler=sampler
        pin_memory=True,
    )

    # ========================
    # 4. 训练循环
    # ========================
    for epoch in range(10):
        sampler.set_epoch(epoch)  # [新增] 每个 epoch 重新打乱数据

        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            loss = nn.functional.mse_loss(model(bx), by)
            optimizer.zero_grad()
            loss.backward()       # DDP 自动在此处 all-reduce 梯度
            optimizer.step()

        # [新增] 只在主进程打印
        if rank == 0:
            print(f"Epoch {epoch+1} done")

    # ========================
    # [新增] 5. 保存模型（只在主进程保存）
    # ========================
    if rank == 0:
        # .module 取出 DDP 包装内的原始模型
        torch.save(model.module.state_dict(), "model.pt")

    # ========================
    # [新增] 6. 清理
    # ========================
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

#### 改造清单总结

| 步骤 | 修改内容 | 说明 |
|------|---------|------|
| 1 | `dist.init_process_group("nccl")` | 初始化通信 |
| 2 | `model = DDP(model, device_ids=[...])` | 包装模型 |
| 3 | `DistributedSampler` 替代 `shuffle=True` | 数据分片 |
| 4 | `sampler.set_epoch(epoch)` | 每 epoch 重新打乱 |
| 5 | `if rank == 0:` 保护日志和保存 | 避免重复输出 |
| 6 | `dist.destroy_process_group()` | 清理资源 |
| 7 | 使用 `torchrun` 启动 | 自动设置环境变量 |

### 3.5 torchrun 启动命令

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train_ddp.py

# 多机（2 台机器，每台 4 卡）
# 机器 1（主节点）:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_ddp.py

# 机器 2:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_ddp.py
```

---

## 第四章：梯度聚灵 — 梯度累加

> *"一次吞不下的灵药，分多次服用，效果等同。"*

### 4.1 为什么需要梯度累加？

大模型训练中，我们通常希望使用较大的 batch size 以获得更稳定的梯度估计和更好的训练效果。但显存限制了单次能处理的样本数量。

**梯度累加**的思想极为朴素：既然一次放不下大批次，就多做几次前向/反向传播，累积梯度后再更新参数。

```
等效批次公式：

  effective_batch_size = micro_batch_size × gradient_accumulation_steps × num_gpus

例如：
  micro_batch = 4（每卡每步处理 4 个样本）
  accum_steps = 8（累积 8 步的梯度）
  num_gpus = 4（4 张卡 DDP）

  effective_batch = 4 × 8 × 4 = 128
```

这意味着你可以用 4 个样本的显存，实现 128 个样本的训练效果。

### 4.2 实现细节

梯度累加看似简单，但有几个容易出错的细节：

#### 细节一：Loss 需要除以累加步数

```python
# 错误！累积后的梯度会比正常大 accum_steps 倍
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()           # 梯度累积
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 正确：将 loss 除以累加步数
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accum_steps  # 归一化！
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**为什么要除？** 因为 `loss.backward()` 默认是将梯度**累加**（而非替换）到 `.grad` 上。连续调用 `accum_steps` 次，梯度就是正常值的 `accum_steps` 倍。除以 `accum_steps` 相当于手动求平均。

#### 细节二：DDP 中的梯度同步时机

DDP 默认在每次 `backward()` 后都进行 All-Reduce 梯度同步。但在梯度累加中，我们只需要在最后一步同步：

```python
# 优化：在累积步中禁用梯度同步，只在最后一步同步
for i, batch in enumerate(dataloader):
    # 是否是累积的最后一步？
    is_accumulation_boundary = (i + 1) % accum_steps == 0

    # no_sync(): 跳过 All-Reduce，节省通信开销
    context = model.no_sync() if not is_accumulation_boundary else nullcontext()

    with context:
        loss = model(batch) / accum_steps
        loss.backward()

    if is_accumulation_boundary:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.3 完整代码示例

```python
import torch
import torch.nn as nn
from contextlib import nullcontext

def gradient_accumulation_demo():
    """
    梯度聚灵术 —— 梯度累加完整示例

    以小批次模拟大批次训练，突破显存限制。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 参数
    micro_batch_size = 4        # 显存只能承受这么多
    accum_steps = 8             # 累积 8 步
    effective_batch = micro_batch_size * accum_steps  # = 32

    print(f"[梯度聚灵] 微批次: {micro_batch_size}, 累积步: {accum_steps}")
    print(f"[梯度聚灵] 等效批次: {effective_batch}")

    # 模拟 dataloader
    num_samples = 320
    data = torch.randn(num_samples, 512, device=device)
    targets = torch.randn(num_samples, 512, device=device)

    num_epochs = 3
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_updates = 0

        for i in range(0, num_samples, micro_batch_size):
            batch_x = data[i:i+micro_batch_size]
            batch_y = targets[i:i+micro_batch_size]

            # 前向 + 反向
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            loss = loss / accum_steps    # 关键：归一化
            loss.backward()              # 梯度累积到 .grad

            step_in_accum = (i // micro_batch_size + 1) % accum_steps

            # 累积完毕，执行更新
            if step_in_accum == 0:
                # 可选：梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                num_updates += 1

            epoch_loss += loss.item() * accum_steps

        avg_loss = epoch_loss / (num_samples / micro_batch_size)
        print(f"  Epoch {epoch+1} | Loss: {avg_loss:.6f} | Updates: {num_updates}")

    print("[梯度聚灵] 训练完成！")

# 取消注释以运行:
# gradient_accumulation_demo()
```

---

## 第五章：深渊之力 — DeepSpeed 入门

> *"传说深渊之中藏有一股远古之力，能将百株异火的力量压缩至一株之中。*
> *此力名为 ZeRO —— Zero Redundancy Optimizer。"*

### 5.1 为什么需要 DeepSpeed？

回顾第一章的显存分析，训练一个 7B 模型需要约 124~144 GB 显存。即使使用 DDP 和 4 张 A100 80GB，每张卡仍需独立承担全部的模型参数、梯度和优化器状态——DDP 不减少单卡显存。

```
DDP 的局限：

  GPU 0                GPU 1                GPU 2                GPU 3
  ┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
  │ 完整参数   │         │ 完整参数   │         │ 完整参数   │         │ 完整参数   │
  │ 完整梯度   │         │ 完整梯度   │         │ 完整梯度   │         │ 完整梯度   │
  │ 完整优化器 │         │ 完整优化器 │         │ 完整优化器 │         │ 完整优化器 │
  └──────────┘         └──────────┘         └──────────┘         └──────────┘

  每卡显存: ~140 GB     → 单张 A100 80GB 装不下！
```

DeepSpeed 的 ZeRO 系列算法正是为解决这个问题而生。

### 5.2 ZeRO 三阶段详解

ZeRO（Zero Redundancy Optimizer）的核心思想：**消除冗余**。DDP 中每张卡都存储完整的参数、梯度和优化器状态，这是巨大的浪费。ZeRO 将这些状态分片（partition）到各张卡上。

#### ZeRO Stage 1：分割优化器状态

```
传统 DDP（4 卡）：每卡存储完整 Adam 状态（= 56 GB for 7B model）

ZeRO Stage 1（4 卡）：每卡只存 1/4 的 Adam 状态

  GPU 0              GPU 1              GPU 2              GPU 3
  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
  │ 完整参数    │     │ 完整参数    │     │ 完整参数    │     │ 完整参数    │
  │ 完整梯度    │     │ 完整梯度    │     │ 完整梯度    │     │ 完整梯度    │
  │ Adam 0~24% │     │ Adam 25~49%│     │ Adam 50~74%│     │ Adam 75~99%│
  └────────────┘     └────────────┘     └────────────┘     └────────────┘

  优化器显存: 56 GB / 4 = 14 GB/卡 （节省 75%！）
  参数和梯度: 不变
```

**显存节省**：优化器状态降至 1/N（N = GPU 数）
**通信开销**：与 DDP 基本相同

#### ZeRO Stage 2：分割优化器状态 + 梯度

```
ZeRO Stage 2（4 卡）：优化器状态和梯度都分片

  GPU 0              GPU 1              GPU 2              GPU 3
  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
  │ 完整参数    │     │ 完整参数    │     │ 完整参数    │     │ 完整参数    │
  │ 梯度 0~24% │     │ 梯度 25~49%│     │ 梯度 50~74%│     │ 梯度 75~99%│
  │ Adam 0~24% │     │ Adam 25~49%│     │ Adam 50~74%│     │ Adam 75~99%│
  └────────────┘     └────────────┘     └────────────┘     └────────────┘

  优化器显存: 14 GB/卡
  梯度显存: 14 GB / 4 = 3.5 GB/卡
  参数显存: 14 GB（不变）
```

**显存节省**：优化器 + 梯度降至 1/N
**通信开销**：略多于 DDP（需要额外的梯度 reduce-scatter）
**推荐场景**：大多数训练场景的最佳起点

#### ZeRO Stage 3：分割一切

```
ZeRO Stage 3（4 卡）：参数、梯度、优化器状态全部分片

  GPU 0              GPU 1              GPU 2              GPU 3
  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
  │ 参数 0~24% │     │ 参数 25~49%│     │ 参数 50~74%│     │ 参数 75~99%│
  │ 梯度 0~24% │     │ 梯度 25~49%│     │ 梯度 50~74%│     │ 梯度 75~99%│
  │ Adam 0~24% │     │ Adam 25~49%│     │ Adam 50~74%│     │ Adam 75~99%│
  └────────────┘     └────────────┘     └────────────┘     └────────────┘

  每卡总显存: (14 + 14 + 56) / 4 = 21 GB/卡
  对比 DDP: 14 + 14 + 56 = 84 GB/卡

  显存降至原来的 25%！
```

**显存节省**：所有状态降至 1/N
**通信开销**：显著增加（前向/反向传播时都需要 all-gather 参数）
**推荐场景**：模型大到 Stage 2 也装不下时

#### 三阶段对比总结

| | Stage 1 | Stage 2 | Stage 3 |
|---|---------|---------|---------|
| 分片内容 | 优化器状态 | + 梯度 | + 参数 |
| 单卡显存（7B, 4卡） | ~70 GB | ~31 GB | **~21 GB** |
| 通信开销 | ≈ DDP | 略高 | 显著增加 |
| 代码侵入性 | 极低 | 低 | 中等 |
| 速度影响 | 基本无 | 轻微 | 明显（10~30%） |
| 推荐度 | 入门 | **通用推荐** | 极大模型 |

### 5.3 ZeRO-Offload：借力 CPU

当 GPU 显存仍然不足时，DeepSpeed 还提供了"借力"方案——将部分数据卸载（offload）到 CPU 内存甚至 NVMe SSD：

```
ZeRO-Offload（以 Stage 2 + CPU Offload 为例）：

  GPU                           CPU 内存（通常 256~512 GB）
  ┌─────────────┐               ┌──────────────────┐
  │ 完整参数     │               │ 优化器状态         │
  │ （FP16）     │ ←── 通信 ───→ │ （FP32，分片）     │
  │             │               │                  │
  │ 激活值       │               │ FP32 主权重副本    │
  │ （计算时）    │               │                  │
  └─────────────┘               └──────────────────┘

  GPU 只需存放: 参数 (FP16) + 激活值
  优化器状态全部在 CPU 上

  代价: CPU ↔ GPU 数据传输成为瓶颈（PCIe 带宽有限）
```

**适用场景**：
- 单卡微调大模型（如单张 24GB 卡微调 7B）
- CPU 内存充裕但 GPU 显存不足的情况
- 可接受一定速度损失（通常慢 1.5~2x）

### 5.4 DeepSpeed 配置文件详解

DeepSpeed 通过一个 JSON 配置文件控制所有优化策略。下面逐段解析一个实用的 Stage 2 配置：

```json
{
    // ============================================
    // 训练批次配置
    // ============================================
    "train_micro_batch_size_per_gpu": 4,
    // 每个 GPU 单步处理的样本数
    // 这是实际占用显存的批次大小

    "gradient_accumulation_steps": 8,
    // 梯度累加步数
    // 等效 batch = 4 × 8 × num_gpus

    // ============================================
    // ZeRO 优化配置 —— 聚火阵法的核心
    // ============================================
    "zero_optimization": {
        "stage": 2,
        // Stage 2: 分片优化器状态 + 梯度
        // 改为 1: 只分片优化器（更快但省得少）
        // 改为 3: 全部分片（最省显存但更慢）

        "offload_optimizer": {
            "device": "none"
            // "none": 优化器留在 GPU（快）
            // "cpu":  卸载到 CPU（慢但省 GPU 显存）
        },

        "allgather_partitions": true,
        // 使用高效的 allgather 操作收集分片参数

        "allgather_bucket_size": 200000000,
        // 通信桶大小（200MB），控制通信粒度
        // 太小: 通信次数多，开销大
        // 太大: 内存峰值高

        "overlap_comm": true,
        // 通信与计算重叠 —— 边算边传
        // 计算第 N+1 层的同时传输第 N 层的梯度

        "reduce_scatter": true,
        // 使用 reduce_scatter 而非 all_reduce
        // 直接将梯度规约到负责的那个 GPU 上

        "reduce_bucket_size": 200000000,
        // 梯度规约桶大小

        "contiguous_gradients": true
        // 连续内存存储梯度，减少碎片化
    },

    // ============================================
    // 混合精度配置 —— 混元真气
    // ============================================

    // BF16 模式（推荐，A100/H100 等现代 GPU）：
    "bf16": {
        "enabled": true
    },

    // 或 FP16 模式（老 GPU 或不支持 BF16 时）：
    // "fp16": {
    //     "enabled": true,
    //     "loss_scale": 0,           // 0 = 动态 loss scaling
    //     "loss_scale_window": 1000, // 多少步没有 overflow 就加倍
    //     "initial_scale_power": 16, // 初始 scale = 2^16
    //     "hysteresis": 2,           // overflow 容忍次数
    //     "min_loss_scale": 1        // 最小 scale factor
    // },

    // ============================================
    // 梯度裁剪
    // ============================================
    "gradient_clipping": 1.0,
    // 梯度 L2 范数超过此值则裁剪
    // 防止梯度爆炸（斗气暴走）

    // ============================================
    // 优化器配置
    // ============================================
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    // ============================================
    // 学习率调度器
    // ============================================
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    },

    // ============================================
    // 日志与调试
    // ============================================
    "steps_per_print": 50,
    "wall_clock_breakdown": false
    // 设为 true 可看到每步的耗时分解
}
```

> **注意**：标准 JSON 不支持注释。上述注释仅用于教学说明。实际使用时请删除所有 `//` 注释，或使用 DeepSpeed 支持的 JSONC 格式。

### 5.5 与 HuggingFace Trainer 集成

DeepSpeed 与 HuggingFace Transformers 的 Trainer 无缝集成，只需几行配置即可启用：

```python
from transformers import TrainingArguments, Trainer

# 方法一：通过 TrainingArguments 参数
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,

    # DeepSpeed 配置 —— 只需这一行！
    deepspeed="ds_config.json",

    # 或者直接传入字典
    # deepspeed={
    #     "zero_optimization": {"stage": 2},
    #     "bf16": {"enabled": True},
    # },
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # ...
)

trainer.train()
```

启动命令：

```bash
# 单机多卡
deepspeed --num_gpus=4 train.py \
    --deepspeed ds_config.json

# 或使用 torchrun（HuggingFace 推荐）
torchrun --nproc_per_node=4 train.py \
    --deepspeed ds_config.json

# 多机（2 节点，每节点 4 卡）
deepspeed --num_nodes=2 --num_gpus=4 \
    --hostfile hostfile.txt \
    train.py --deepspeed ds_config.json
```

### 5.6 实战配置推荐

根据你的硬件和模型大小，选择合适的 DeepSpeed 配置：

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| 1.5B 模型 + 4×24GB | ZeRO Stage 2 + BF16 | 轻松驾驭 |
| 7B 模型 + 4×24GB | ZeRO Stage 3 + BF16 + CPU Offload | 刚好能跑 |
| 7B 模型 + 4×80GB | ZeRO Stage 2 + BF16 | 游刃有余 |
| 13B 模型 + 4×80GB | ZeRO Stage 2 + BF16 | 够用 |
| 70B 模型 + 8×80GB | ZeRO Stage 3 + BF16 | 需要精细调参 |
| 7B LoRA + 1×24GB | 无需 DeepSpeed（LoRA 已够高效） | 直接用上一卷的方法 |

---

## 第六章：化翼飞升 — 多卡训练实战

> *"混元真气凝于体内，聚火成阵环于周身，*
> *梯度聚灵蓄势待发，深渊之力涌动不息。*
> *四大功法合为一体，斗气化翼，翱翔天际！"*

### 6.1 综合实战：从零搭建多卡训练

前面五章分别介绍了各项功法，现在是将它们融合的时候了。请参考本卷附带的实战代码：

```
notebooks/05_分布式训练实战.py
```

该脚本支持三种运行模式：

```bash
# 模式 1: 单卡训练（入门验证）
python notebooks/05_分布式训练实战.py

# 模式 2: DDP 多卡训练（推荐）
torchrun --nproc_per_node=4 notebooks/05_分布式训练实战.py --ddp

# 模式 3: DeepSpeed 训练（大模型推荐）
deepspeed notebooks/05_分布式训练实战.py --deepspeed
```

### 6.2 调试技巧

多卡训练的调试比单卡复杂得多。以下是实战中积累的技巧：

#### 技巧一：先用单卡验证正确性

```bash
# 永远先确保单卡能跑通
python train.py --batch_size=2 --epochs=1

# 然后再开启多卡
torchrun --nproc_per_node=2 train.py --ddp --batch_size=2 --epochs=1
```

#### 技巧二：使用 NCCL 调试环境变量

```bash
# 打印 NCCL 调试信息
export NCCL_DEBUG=INFO

# 若多机通信失败，检查网络接口
export NCCL_SOCKET_IFNAME=eth0

# 超时设置（秒），默认 1800
export NCCL_TIMEOUT=3600

# 禁用 P2P 通信（某些网络拓扑下需要）
export NCCL_P2P_DISABLE=1
```

#### 技巧三：检查梯度同步是否正确

```python
# 在 backward 后检查所有 rank 的梯度是否一致
def check_gradient_sync(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_sum = param.grad.data.sum()
            # 所有 rank 应该得到相同的值
            torch.distributed.all_reduce(grad_sum)
            grad_mean = grad_sum / torch.distributed.get_world_size()
            print(f"[Rank {torch.distributed.get_rank()}] {name}: grad_mean={grad_mean.item():.6f}")
```

#### 技巧四：处理随机种子

```python
# 每个 rank 需要不同的数据种子（看到不同数据）
# 但模型初始化种子需要相同（确保参数一致）
torch.manual_seed(42)          # 模型初始化：所有 rank 相同
model = create_model()

# DDP 后，数据部分用不同种子
torch.manual_seed(42 + rank)   # 数据采样：每个 rank 不同
```

#### 技巧五：处理训练中断

```python
# 保存检查点（包含所有恢复训练所需的状态）
def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    if dist.get_rank() == 0:
        checkpoint = {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        torch.save(checkpoint, path)
        print(f"[检查点] 已保存至 {path}")

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]
```

### 6.3 性能分析与优化

#### 使用 PyTorch Profiler 分析瓶颈

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        with record_function("forward"):
            outputs = model(batch)
            loss = outputs["loss"]
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        prof.step()

# 查看报告
# tensorboard --logdir=./profile_logs
```

#### 关键性能指标

```
1. GPU 利用率（应 > 80%）
   nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1

2. 显存利用率（应尽量高但留余量）
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

3. 通信/计算比（通信时间应 < 总时间的 20%）
   使用 PyTorch Profiler 或 NCCL_DEBUG=INFO 查看

4. 吞吐量（samples/sec 或 tokens/sec）
   比较单卡 vs 多卡，理想情况下应线性扩展
```

#### 常见性能瓶颈及解决方案

| 瓶颈 | 症状 | 解决方案 |
|------|------|---------|
| 数据加载慢 | GPU 利用率低，间歇性下降 | 增加 `num_workers`，启用 `pin_memory` |
| 通信瓶颈 | 多卡加速比远低于线性 | 检查网络带宽，使用 `overlap_comm` |
| 显存碎片 | OOM 但显存显示有余 | 使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 梯度累加过多 | 训练不稳定 | 减少累加步数，配合多卡 |
| 不均匀负载 | 某些卡显存明显高于其他 | 检查是否误用 DataParallel |

### 6.4 常见陷阱

**陷阱一：忘记 `sampler.set_epoch(epoch)`**

```python
# 错误：每个 epoch 的数据顺序不变，影响训练效果
for epoch in range(num_epochs):
    for batch in dataloader:
        ...

# 正确：
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)    # 必须！每 epoch 重新打乱
    for batch in dataloader:
        ...
```

**陷阱二：所有 rank 都在保存和打印**

```python
# 错误：所有进程都保存模型和打印日志
torch.save(model.state_dict(), "model.pt")  # 多个进程同时写同一个文件！
print(f"Loss: {loss.item()}")                # 打印 N 遍！

# 正确：只在主进程操作
if rank == 0:
    torch.save(model.module.state_dict(), "model.pt")
    print(f"Loss: {loss.item()}")
```

**陷阱三：死锁**

DDP 要求所有进程执行相同的操作序列。如果某个进程跳过了某次 `forward/backward`，其他进程会在 all-reduce 处永久等待。

```python
# 错误：只有部分进程执行 forward
if some_condition_that_differs_across_ranks:
    loss = model(batch)
    loss.backward()  # 其他进程在等待 all-reduce!

# 正确：所有进程必须执行相同的 forward/backward 序列
loss = model(batch)
loss.backward()
if rank == 0 and some_condition:
    # 只在主进程做额外操作（如日志记录）
    log_metrics(loss)
```

**陷阱四：batch size 和学习率的缩放**

从单卡扩展到多卡时，等效 batch size 增大了 N 倍。根据线性缩放规则（linear scaling rule），学习率也应相应调整：

```python
# 单卡: batch=32, lr=1e-4
# 4 卡 DDP: 等效 batch=128, lr 应调整为 4e-4（线性缩放）
# 注意：这只是经验法则，实际需要根据训练稳定性调整
base_lr = 1e-4
lr = base_lr * world_size          # 线性缩放
# 或更保守的 sqrt 缩放:
lr = base_lr * math.sqrt(world_size)

# 同时增加 warmup 步数以保持稳定
warmup_steps = base_warmup * world_size
```

---

## 附录 A：FSDP — PyTorch 原生分布式利器

> *"DeepSpeed 深渊之力固然强大，但需要引入额外依赖。FSDP 是 PyTorch 内置的分片策略，开箱即用，与 PyTorch 生态无缝集成。"*

### A.1 FSDP vs DeepSpeed ZeRO

| 维度 | DeepSpeed ZeRO | FSDP |
|------|---------------|------|
| 来源 | 微软 DeepSpeed 库 | PyTorch 原生（v1.12+） |
| 依赖 | 需安装 deepspeed | PyTorch 自带 |
| 分片粒度 | 按参数组 | 按模块（nn.Module） |
| 通信策略 | 自定义 | NCCL（PyTorch 默认） |
| 灵活性 | 非常高 | 中等 |
| 易用性 | 中等（需配置 JSON） | 较好（纯 Python API） |
| 生态支持 | HuggingFace/ Megatron | PyTorch 原生 |
| 模型并行 | 支持张量并行 | 支持 CPU Offload |

### A.2 FSDP 实战代码

```python
"""
=== 丹方 5.A.1：使用 FSDP 训练大模型 ===
"""
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.distributed as dist

def setup_fsdp(model):
    """
    FSDP 配置 — 将模型分片到多个 GPU
    """
    # 自动包装策略：参数量 > 100M 的模块单独分片
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=100_000_000  # 100M
    )

    # 分片策略
    sharding_strategy = ShardingStrategy.FULL_SHARD  # 等同于 ZeRO-3

    # FSDP 配置
    fsdp_config = {
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "cpu_offload": torch.distributed.fsdp.CPUOffload(offload_params=False),
        "backward_prefetch": torch.distributed.fsdp.BackwardPrefetch.BACKWARD_PRE,
        "mixed_precision": torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        "device_id": torch.cuda.current_device(),
    }

    model = FSDP(model, **fsdp_config)
    return model


# 使用 HuggingFace 的 FSDP 集成
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fsdp_output",
    fsdp="full_shard auto_wrap",  # 一行启用 FSDP！
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch_policy": "backward_pre",
        "fsdp_backward_prefetch_num_layers": 4,
    },
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    bf16=True,
    # ... 其他参数
)
```

### A.3 分布式训练调试技巧

```python
"""
=== 分布式训练调试工具 ===
"""
import torch.distributed as dist

def check_distributed():
    """检查分布式环境是否正确配置"""
    if not dist.is_initialized():
        print("⚠️ 分布式未初始化！请使用 torchrun 或 torch.distributed.launch")
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Rank {rank}/{world_size} | Local Rank {local_rank} | "
          f"Device: cuda:{local_rank}")

    # 检查 GPU 间通信
    if rank == 0:
        data = torch.tensor([42.0], device=f"cuda:{local_rank}")
        for i in range(1, world_size):
            dist.send(data, dst=i)
    else:
        data = torch.tensor([0.0], device=f"cuda:{local_rank}")
        dist.recv(data, src=0)
        print(f"Rank {rank} received: {data.item()}")

    print("✅ 分布式通信正常！")


# 显存监控（分布式环境）
def log_memory_usage(rank):
    """记录每个 GPU 的显存使用情况"""
    allocated = torch.cuda.memory_allocated(rank) / 1e9
    reserved = torch.cuda.memory_reserved(rank) / 1e9
    print(f"[Rank {rank}] Allocated: {allocated:.2f}GB | "
          f"Reserved: {reserved:.2f}GB")


# 常见分布式训练问题排查
print("""
╔══════════════════════════════════════════════════════════════╗
║            分布式训练常见问题排查                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                            ║
║  问题 1: NCCL 超时                                          ║
║  现象：Hang / Timeout / RuntimeError                        ║
║  原因：GPU 间通信问题                                       ║
║  解决：export NCCL_DEBUG=INFO                               ║
║        export NCCL_TIMEOUT=1800                             ║
║        检查网络连接（ibstat / ifconfig）                     ║
║                                                            ║
║  问题 2: 各 GPU 显存不均衡                                  ║
║  现象：Rank 0 OOM 但其他 GPU 还有余量                       ║
║  原因：数据分布不均 / 某些 rank 处理更长序列                ║
║  解决：确保 Dataset shuffle + drop_last=True                ║
║        使用 PackDataset 均衡序列长度                        ║
║                                                            ║
║  问题 3: Loss 不一致                                        ║
║  现象：各 rank 打印的 loss 不同                             ║
║  原因：DDP 的 gradient all-reduce 在 backward 之后才同步     ║
║  解决：这是正常的（backward 期间梯度在各 GPU 上不同）        ║
║        optimizer.step() 之后各 rank 参数一致                 ║
║                                                            ║
║  问题 4: 训练速度不随 GPU 数量线性增长                      ║
║  原因：通信开销 / Batch 太小 / 数据加载瓶颈                 ║
║  解决：增大 per_device_batch_size                           ║
║        使用 Gradient Accumulation 保持大等效 batch           ║
║        增加 num_workers (DataLoader)                        ║
║                                                            ║
╚══════════════════════════════════════════════════════════════╝
""")
```

---

## 附录 B：分布式训练效率优化

### B.1 通信重叠与流水线

```python
"""
=== 通信-计算重叠优化 ===
"""
class OverlappedTraining:
    """
    通信-计算重叠 — 在等待 all-reduce 的同时进行下一个微批次的前向计算
    """
    def train_step(self, model, optimizer, dataloader, device):
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            # 正常模式：前向 → 反向 → 通信（同步）→ 更新
            # 重叠模式：微批次 1 反向时，同时开始微批次 2 的前向
            loss = model(batch.to(device))
            loss.backward()

            # 对于 FSDP/DDP，backward 中的 all-reduce 可以通过
            # async 操作与下一步的前向计算重叠
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True 比 zero 更快
```

### B.2 数据加载优化

```python
"""
=== 高效数据加载 — 消除 IO 瓶颈 ===
"""
from torch.utils.data import DataLoader

# 关键优化参数
optimized_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,        # 预取线程数（通常设为 CPU 核数的一半）
    pin_memory=True,      # 使用锁页内存加速 GPU 传输
    prefetch_factor=2,    # 每个 worker 预取 2 个 batch
    persistent_workers=True,  # 保持 worker 进程存活（避免重复初始化开销）
    drop_last=True,       # 丢弃最后不完整的 batch（防止分布式训练 hang）
)

print("""
数据加载优化清单：
  ✅ num_workers=4~8（取决于 CPU 核数和 IO 速度）
  ✅ pin_memory=True（GPU 训练必须开启）
  ✅ persistent_workers=True（避免 worker 重复创建开销）
  ✅ 预先将数据转为 tensor 格式存储（避免实时转换）
  ✅ 使用 WebDataset 或 StreamingDataset 处理超大文件
  ✅ drop_last=True（分布式训练必须）
""")
```

### B.3 分布式训练性能基准

```python
print("""
╔═══════════════════════════════════════════════════════════════════╗
║           典型分布式训练吞吐量参考                                ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                 ║
║  7B 模型训练（seq_len=2048, batch_per_gpu=4）                    ║
║  ┌───────────┬──────────┬────────────┬──────────────┐           ║
║  │ GPU 配置  │ tokens/s │ 显存/卡    │ 扩展效率     │           ║
║  ├───────────┼──────────┼────────────┼──────────────┤           ║
║  │ 1×A100    │ ~3,000   │ ~40GB      │ 1.00×        │           ║
║  │ 2×A100    │ ~5,500   │ ~25GB      │ 0.92×        │           ║
║  │ 4×A100    │ ~10,000  │ ~20GB      │ 0.83×        │           ║
║  │ 8×A100    │ ~18,000  │ ~18GB      │ 0.75×        │           ║
║  └───────────┴──────────┴────────────┴──────────────┘           ║
║                                                                 ║
║  注意：                                                         ║
║  - 扩展效率随 GPU 数增加而下降（通信开销占比增大）               ║
║  - ZeRO-3/FSDP 的通信开销高于 ZeRO-2                            ║
║  - 增大 batch size 可提高扩展效率（减少通信/计算比）              ║
║  - 使用 InfiniBand 网络 > 使用以太网（延迟更低）                 ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════════╝
""")
```

---


---

## 附录 C：ZeRO-3 深度实战 — 分片一切的极致功法

> *"深渊之力的第三重境界，将模型参数、梯度、优化器状态全部切碎分片，散布于万张显卡之间。每张卡只握有拼图的一角，却能在需要时瞬间汇聚全貌。"*

### C.1 ZeRO-3 的本质

前面学过 ZeRO Stage 1/2 只分片优化器状态和梯度。**ZeRO-3** 的极致之处在于——连**模型参数**也分片了。

```
ZeRO 三重境界对比：

ZeRO-1（分片优化器状态）：
  GPU 0: Optimizer_0   GPU 1: Optimizer_1   GPU 2: Optimizer_2
  All GPU: [完整参数] [完整梯度]
  显存节省：~4x 优化器状态

ZeRO-2（+分片梯度）：
  GPU 0: Optimizer_0 + Grad_0   GPU 1: Optimizer_1 + Grad_1
  All GPU: [完整参数]
  显存节省：~4x 优化器 + ~2x 梯度

ZeRO-3（+分片参数）：
  GPU 0: Param_0 + Optimizer_0 + Grad_0
  GPU 1: Param_1 + Optimizer_1 + Grad_1
  GPU 2: Param_2 + Optimizer_2 + Grad_2
  前向/反向时按需 all-gather 参数
  显存节省：与 GPU 数量成正比！8卡 = 参数显存 / 8
```

### C.2 DeepSpeed ZeRO-3 实战

```python
# === deepspeed_config_z3.json ===
# ZeRO-3 配置文件
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 10000,
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 300
        }
    },

    "fp16": {
        "enabled": "auto"
    },

    "zero_optimization": {
        "stage": 3,                    // 关键！Stage 3
        "overlap_comm": true,           // 通信-计算重叠
        "contiguous_gradients": true,   // 梯度连续存储
        "sub_group_size": 1e9,          // 子组大小（控制 all-gather 粒度）
        "reduce_bucket_size": 5e8,      // all-reduce 桶大小
        "stage3_prefetch_bucket_size": 5e8,  // 预取桶大小
        "stage3_param_persistence_threshold": 1e5,  // 小于此的参数不分片
        "stage3_max_live_parameters": 1e9,        // 最大活跃参数数
        "stage3_max_reuse_distance": 1e9,         // 最大复用距离
        "stage3_gather_16bit_weights_on_model_save": true  // 保存时汇聚权重
    },

    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
```

```python
# === ZeRO-3 训练代码（与 ZeRO-2 几乎相同！）===
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# DeepSpeed 初始化（配置文件指定 ZeRO-3）
ds_args = {
    "deepspeed_config": "deepspeed_config_z3.json",
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
}
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    args=type('Args', (), ds_args)(),
    optimizer=None,  # 让 DeepSpeed 自动创建
    model_parameters=model.parameters(),
)

# 训练循环（与普通训练完全一致）
for batch in dataloader:
    input_ids = tokenizer(batch["text"], return_tensors="pt", 
                          truncation=True, max_length=2048).input_ids.cuda()
    outputs = model_engine(input_ids, labels=input_ids)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()

# ZeRO-3 的魔法：model_engine 会在前向传播时自动 all-gather 所需参数，
# 反向传播后自动丢弃参数只保留梯度分片。
# 对使用者完全透明！
```

### C.3 ZeRO-3 的通信开销与取舍

```
┌─────────────────────────────────────────────────────────────────────┐
│               ZeRO-3 通信开销分析                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  前向传播：                                                         │
│    每一层 all-gather 参数 → GPU数量 次通信                           │
│    通信量 = 2 * 参数量 / GPU数 * 层数                               │
│                                                                     │
│  反向传播：                                                         │
│    每一层 reduce-scatter 梯度 → GPU数量 次通信                      │
│    通信量 = 2 * 梯度量 / GPU数 * 层数                               │
│                                                                     │
│  总通信 = 4 * 参数量 * 层数 / GPU数                                 │
│                                                                     │
│  与 ZeRO-2 的区别：                                                 │
│    ZeRO-2: 通信量 ∝ 梯度量（与 GPU 数量无关）                       │
│    ZeRO-3: 通信量 ∝ 参数量 / GPU数（GPU越多单次通信越少，但次数更多）│
│                                                                     │
│  结论：                                                             │
│    - GPU 数少（≤4）：ZeRO-2 通常更快                                │
│    - GPU 数多（≥8）或模型极大：ZeRO-3 是唯一选择                    │
│    - 开启 overlap_comm 可缓解 50%+ 通信延迟                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### C.4 ZeRO-3 + CPU Offload 极限省钱模式

```python
# === 极限省钱：用 CPU 内存扩展显存 ===
# deepspeed_config_z3_offload.json

# 核心配置差异：
zero_offload_optimizer = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",           // 优化器状态放 CPU
            "pin_memory": true         // 锁页内存加速传输
        },
        "offload_param": {
            "device": "cpu",           // 参数也放 CPU（仅需要时加载到 GPU）
            "pin_memory": true
        }
    }
}

# 效果：用 CPU 内存扩展显存！
# 8GB 显存 + 64GB 内存 → 可以训练 7B 模型
# 代价：速度降低 3-5 倍（CPU-GPU 数据传输瓶颈）
print("""
╔════════════════════════════════════════════════════════╗
║  ZeRO-3 + CPU Offload 性能估算                          ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  配置: 1×RTX 4090 (24GB) + 64GB CPU RAM               ║
║                                                        ║
║  ┌──────────────────┬────────────┬──────────┐          ║
║  │ 方案             │ 可训练模型 │ 相对速度 │          ║
║  ├──────────────────┼────────────┼──────────┤          ║
║  │ 纯 GPU (ZeRO-2)  │ ~7B FP16   │ 1.0×     │          ║
║  │ ZeRO-3 纯 GPU    │ ~14B FP16  │ 0.5×     │          ║
║  │ Z3 + Opt Offload │ ~30B FP16  │ 0.25×    │          ║
║  │ Z3 + Param OFload│ ~65B FP16  │ 0.1×     │          ║
║  └──────────────────┴────────────┴──────────┘          ║
║                                                        ║
║  炼丹口诀：                                              ║
║  "心急吃不了热豆腐，CPU Offload 能跑就行。"              ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
""")
```

---

## 附录 D：Colossal-AI — 另一派分布式功法

> *"DeepSpeed 是微软的深渊之力，Colossal-AI 是开源社区的混元大阵。两者殊途同归，各有千秋。"*

### D.1 Colossal-AI 简介

Colossal-AI（由 HPC-AI Tech 开发）提供了一套与 DeepSpeed 并列的分布式训练框架，特色功能包括：

| 特性 | Colossal-AI | DeepSpeed |
|------|-------------|-----------|
| ZeRO-1/2/3 | ✅ | ✅ |
| 张量并行 (TP) | ✅ | ✅ |
| 流水线并行 (PP) | ✅ | ✅ |
| 3D 并行 | ✅ | ✅ |
| Gemini (异构内存) | ✅ 独有 | ❌ |
| Sequence Parallelism | ✅ | 部分支持 |
| MoE 并行 | ✅ | ✅ |
| ease of use | 配置式 | 配置式 |
| 社区 | 学术导向 | 工业导向 |

### D.2 Colossal-AI 快速上手

```bash
# 安装
pip install colossalai

# 检查安装
python -c "import colossalai; print(colossalai.__version__)"
```

```python
# === Colossal-AI ZeRO 训练 ===
import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin

# 方式一：Gemini 插件（自动异构内存管理，类似 ZeRO-3 + Offload）
gemini_plugin = GeminiPlugin(
    precision="fp16",
    placement_policy="auto",      # 自动决定参数放 GPU 还是 CPU
    pin_memory=True,
    search_range_mb=64,           # 搜索最优分块大小
)

# 方式二：ZeRO 插件（类似 DeepSpeed ZeRO）
zero_plugin = LowLevelZeroPlugin(
    stage=3,                      # ZeRO Stage 3
    precision="fp16",
)

# 选择一个插件
booster = Booster(plugin=zero_plugin)

# 准备模型、优化器、数据加载器
model, optimizer, _, _, lr_scheduler = booster.boost(
    model=model,
    optimizer=optimizer,
    dataloader=train_dataloader,
    lr_scheduler=lr_scheduler,
)

# 训练循环
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    booster.backward(loss, optimizer)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

# Colossal-AI 的 Gemini 会自动在 GPU 和 CPU 内存之间分配参数，
# 无需手动配置 offload 策略，比 DeepSpeed 更智能化。
```

### D.3 Gemini 异构内存管理 — Colossal-AI 独门绝技

```
DeepSpeed ZeRO-3 + Offload:
  需要手动配置哪些放 CPU、哪些放 GPU
  配置固定，运行时不调整

Colossal-AI Gemini:
  自动分析每个参数的访问频率和重要性
  热门参数 → GPU（快速访问）
  冷门参数 → CPU / NVMe（节省显存）
  运行时动态调整分配策略！

┌──────────────────────────────────────────────┐
│            Gemini 内存管理示意图              │
│                                              │
│  GPU 显存 (24GB):                            │
│  ┌────────────────────────────────┐          │
│  │ Embedding层 (频繁访问) ████████│          │
│  │ Attention层 (中等频率) █████   │          │
│  │ FFN前几层 (常用) ████          │          │
│  │ [空闲空间用于计算]  ░░░░░░░░░  │          │
│  └────────────────────────────────┘          │
│                                              │
│  CPU 内存 (128GB):                           │
│  ┌────────────────────────────────┐          │
│  │ FFN后几层 (偶尔访问) ████      │          │
│  │ LayerNorm等小参数 ███          │          │
│  │ [空闲空间充足] ░░░░░░░░░░░░░  │          │
│  └────────────────────────────────┘          │
│                                              │
│  自动调度：热点参数在GPU，冷门参数在CPU       │
│  目标：在有限GPU显存下最大化训练速度          │
└──────────────────────────────────────────────┘
```

---

## 附录 E：MoE 分布式训练 — 万火同炼

> *"混合专家(MoE)系统，如同将万千异火分给不同的炼丹师，每道火焰只炼制自己擅长的丹药。当所有丹药汇聚，便是万火归一的至高境界。"*

### E.1 MoE 的基本原理

混合专家模型（Mixture of Experts）的核心思想：不增加总计算量的前提下，通过稀疏激活扩大模型参数量。

```
Dense 模型（密炼）：
  输入 → [Layer 1 (全参数)] → [Layer 2 (全参数)] → ... → 输出
  每一层都使用全部参数进行计算
  7B 模型 = 每层 7B 参数参与计算

MoE 模型（稀疏炼）：
  输入 → [Router 选专家] → [Expert 1 (1B)] → 输出
                          → [Expert 2 (1B)] → 输出  (未选中，不计算)
                          → [Expert 3 (1B)] → 输出  (未选中，不计算)
                          → [Expert 4 (1B)] → 输出  (未选中，不计算)
  每一层只激活部分专家（如 Top-2）
  总参数 28B，但每次只计算 2B → 计算量 ≈ 7B Dense

优势：
  参数量是 Dense 模型的 4 倍
  计算量与 Dense 模型相当
  推理速度几乎不增加
```

### E.2 Top-K Router 机制

```python
# === MoE Router 实现 ===
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """MoE 路由器 — 决定每个 token 由哪些专家处理"""

    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape

        # 计算每个 token 对每个专家的得分
        logits = self.gate(x)  # [batch_size, seq_len, num_experts]

        # 选择 Top-K 专家
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax 归一化权重
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices


# 简单的 MoE 层
class SimpleMoE(nn.Module):
    def __init__(self, hidden_size, num_experts=4, top_k=2):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        weights, indices = self.router(x)
        batch_size, seq_len, hidden = x.shape

        output = torch.zeros_like(x)

        for k in range(self.top_k):
            for expert_id in range(self.router.num_experts):
                # 找到被路由到当前专家的 token
                mask = (indices[:, :, k] == expert_id)
                if mask.sum() == 0:
                    continue

                # 获取这些 token 及其权重
                expert_input = x[mask]
                expert_weight = weights[:, :, k][mask].unsqueeze(-1)

                # 专家计算
                expert_output = self.experts[expert_id](expert_input)

                # 加权累加
                output[mask] += expert_weight * expert_output

        return output


# 测试
moe = SimpleMoE(hidden_size=512, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)  # batch=2, seq=10, hidden=512
out = moe(x)
print("MoE output shape:", out.shape)  # [2, 10, 512]
print("Total params: %.1fM" % (sum(p.numel() for p in moe.parameters()) / 1e6))
# 8 个专家，每个 512×512 ≈ 0.26M × 8 = 2.1M 参数
# 但每个 token 只激活 2 个专家
```

### E.3 MoE 分布式训练策略

```
MoE 分布式训练的挑战：

1. 负载不均衡（Load Imbalance）
   - Router 倾向于选择少数专家 → 这些专家过载
   - 解决：辅助损失 (Auxiliary Loss) 鼓励均匀分配

2. 通信量大
   - Token 需要在 GPU 之间传输到对应的专家
   - 解决：EP (Expert Parallelism) 每个专家独占一块 GPU

3. 显存不均衡
   - 各专家参数相同，但激活值分布不同
   - 解决：Expert Parallelism + Capacity Factor

┌──────────────────────────────────────────────────────────┐
│          MoE 并行策略对比                                 │
├────────────────┬─────────────────────────────────────────┤
│ 策略           │ 说明                                    │
├────────────────┼─────────────────────────────────────────┤
│ EP             │ 不同专家放不同 GPU，通信最少             │
│ (Expert Par.)  │ 适合：专家数能被 GPU 数整除             │
│                │ 代表：Megatron-LM MoE, DeepSpeed-MoE    │
├────────────────┼─────────────────────────────────────────┤
│ TP + EP        │ 张量并行 + 专家并行                      │
│                │ 适合：模型极大 + 专家数多                │
│                │ Mixtral-8x7B 常用此方案                 │
├────────────────┼─────────────────────────────────────────┤
│ DP + EP        │ 数据并行 + 专家并行                      │
│                │ 适合：增大全局 batch size                │
├────────────────┼─────────────────────────────────────────┤
│ 3D (TP+EP+PP) │ 三维并行 + 专家并行                      │
│                │ 适合：千亿级 MoE 模型                   │
└────────────────┴─────────────────────────────────────────┘
```

### E.4 著名 MoE 模型一览

| 模型 | 参数量 | 专家数 | Top-K | 活跃参数 | 特点 |
|------|--------|--------|-------|---------|------|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~12.9B | 开源标杆，性能超 Llama 2 70B |
| Mixtral 8x22B | 141B | 8 | 2 | ~39B | 更大版本 |
| DeepSeek-V2 | 236B | 160 | 6 | ~21B | 超多专家 + 共享专家 |
| DeepSeek-V3 | 671B | 256 | 8 | ~37B | 多头潜在注意力(MLA) + MoE |
| Qwen1.5-MoE-A2.7B | 14.3B | 60 | 4 | ~2.7B | 小模型大容量 |
| Grok-1 | 314B | 8 | 2 | ~ unknown | xAI 首个模型 |

```python
# === 使用 HuggingFace 加载 Mixtral ===
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 MoE 模型（需要多卡或量化）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分配专家到不同 GPU
)

# 推理
inputs = tokenizer("What is Mixture of Experts?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Mixtral 在 4×24GB GPU 上即可运行
# 每个专家约 7B 参数，8 个专家分散到 4 张卡
```

---

## 修炼总结

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   化 翼 篇 · 修 炼 总 结                                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 功法速查表

| 功法 | 技术名 | 作用 | 何时使用 |
|------|--------|------|---------|
| 混元真气 | 混合精度 (AMP) | 显存减半，速度提升 | **始终启用** |
| 聚火成阵 | 分布式数据并行 (DDP) | 多卡并行，吞吐翻倍 | 有多卡时 |
| 梯度聚灵 | 梯度累加 | 模拟大 batch | 显存不够大 batch 时 |
| 深渊之力 Stage 1 | ZeRO-1 | 分片优化器状态 | 初步显存优化 |
| 深渊之力 Stage 2 | ZeRO-2 | 分片优化器 + 梯度 | **大多数场景推荐** |
| 深渊之力 Stage 3 | ZeRO-3 | 分片一切 | 极大模型 |
| 深渊借力 | ZeRO-Offload | 卸载至 CPU | 显存极度紧张 |
| 时空互换 | 梯度检查点 | 以时间换显存 | 显存不够时 |

### 修炼路线推荐

```
               ┌──────────────────────────────────────────┐
               │        你的模型和硬件配置是？              │
               └────────────────┬─────────────────────────┘
                                │
               ┌────────────────┼───────────────────┐
               │                │                   │
         小模型 + 单卡     中模型 + 多卡          大模型 + 多卡
         (1~3B, 1×24GB)   (7B, 4×80GB)          (13B+, 8×80GB)
               │                │                   │
               ▼                ▼                   ▼
          AMP + LoRA      AMP + DDP + GA       AMP + DeepSpeed
       (上一卷功法即可)    + ZeRO Stage 2        ZeRO Stage 2/3
                               │                   │
                          若显存仍不足 ────────→ + CPU Offload
                               │                   + 梯度检查点
                          + 梯度检查点
                          + 减小 batch
```

### 下一卷预告

掌握了多卡训练的功法，你已经能够驾驭相当规模的模型了。但真正的挑战才刚刚开始——

**第六卷：凌空篇（斗皇）** 将带你进入更高层次的修炼：
- 模型并行（Tensor Parallelism）—— 当单卡连模型参数都放不下时
- 流水线并行（Pipeline Parallelism）—— 将模型切成多段，流水线式训练
- 3D 并行策略 —— 数据并行 × 张量并行 × 流水线并行
- FSDP（Fully Sharded Data Parallel）—— PyTorch 原生的 ZeRO-3 实现

从斗王迈向斗皇，你将学会在多维度上分割和并行化一切。

---


## 附录 F：FSDP 深度实战 — 碎丹重铸术

> *"ZeRO-3 将修炼者的灵力分散到不同丹田，FSDP 则更进一步——自动管理一切。"*
> *"PyTorch 原生的 FSDP，是分布式训练的现代标准。"*

### F.1 FSDP vs DeepSpeed ZeRO-3

```python
# ========== FSDP vs DeepSpeed 对比 ==========

comparison = """
=== FSDP vs DeepSpeed ZeRO-3 ===

┌─────────────┬──────────────────────┬──────────────────────┐
│     维度     │       FSDP           │    DeepSpeed ZeRO-3  │
├─────────────┼──────────────────────┼──────────────────────┤
│ 来源        │ PyTorch 原生         │ 微软开源             │
│ 安装        │ pip install torch    │ pip install deepspeed│
│ 配置复杂度  │ 中等                 │ 较高                 │
│ 兼容性      │ 与 PyTorch 生态完全  │ 需要适配             │
│ 通信优化    │ 自动                 │ 手动配置             │
│ CPU Offload │ PyTorch 原生支持     │ 原生支持             │
│ 激活检查点  │ 手动集成             │ 内置支持             │
│ 社区活跃度  │ PyTorch 官方维护     │ 微软维护             │
│ 推荐场景    │ PyTorch 为主的技术栈 │ 需要极致优化的场景   │
└─────────────┴──────────────────────┴──────────────────────┘

选择建议：
- 95% 的场景推荐 FSDP（PyTorch 原生，维护更活跃）
- 需要 CPU Offload 的极致省钱场景：两者均可
- 已有 DeepSpeed 生态的项目：继续使用 DeepSpeed
"""
print(comparison)
```

### F.2 FSDP 完整实战

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

# ========== 1. FSDP 环境初始化 ==========

def setup_fsdp():
    """
    初始化 FSDP 分布式环境
    """
    import os
    import torch.distributed as dist
    
    # 使用 torchrun 启动时自动设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


# ========== 2. FSDP 配置 ==========

def create_fsdp_config(model):
    """
    创建 FSDP 配置
    """
    # 混合精度配置
    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,       # 参数精度
        reduce_dtype=torch.float16,       # 梯度规约精度
        buffer_dtype=torch.float16,       # 缓冲区精度
    )
    
    # 分片策略
    sharding_strategy = ShardingStrategy.FULL_SHARD  # 等同 ZeRO-3
    # 其他选项:
    # - ShardingStrategy.NO_SHARD: 等同 DDP
    # - ShardingStrategy.SHARD_GRAD_OP: 等同 ZeRO-2
    
    # 自动包裹策略：按参数量决定哪些层单独分片
    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e7  # 参数量 > 10M 的层单独分片
    )
    
    # CPU Offload（显存不足时的保命策略）
    cpu_offload = CPUOffload(offload_params=True)  # 参数卸载到 CPU
    
    fsdp_config = {
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": cpu_offload,  # 仅在显存紧张时启用
        "device_id": torch.cuda.current_device(),
    }
    
    return fsdp_config


# ========== 3. FSDP 模型包裹 ==========

def wrap_model_with_fsdp(model, fsdp_config=None):
    """
    用 FSDP 包裹模型
    """
    if fsdp_config is None:
        fsdp_config = create_fsdp_config(model)
    
    fsdp_model = FSDP(model, **fsdp_config)
    return fsdp_model


# ========== 4. FSDP 训练循环 ==========

def train_with_fsdp():
    """
    完整的 FSDP 训练示例
    """
    import os
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    
    rank, world_size, _ = setup_fsdp()
    is_main = rank == 0
    
    # 创建模型
    model = nn.Transformer(
        d_model=512, nhead=8, num_encoder_layers=6,
        num_decoder_layers=6, dim_feedforward=2048,
        batch_first=True
    )
    
    # FSDP 包裹
    fsdp_model = wrap_model_with_fsdp(model)
    
    # 优化器（注意：在 FSDP 包裹之后创建）
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), lr=1e-4, weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000
    )
    
    # 模拟训练循环
    print(f"Rank {rank}: FSDP 训练已启动")
    
    for step in range(100):
        # 前向传播
        x = torch.randn(8, 128, 512).cuda()
        y = torch.randn(8, 128, 512).cuda()
        
        output = fsdp_model(x, y)
        loss = output.mean()
        
        # 反向传播（FSDP 自动处理梯度规约）
        loss.backward()
        
        # 梯度裁剪
        fsdp_model.clip_grad_norm_(max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if is_main and step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")
    
    if world_size > 1:
        dist.destroy_process_group()

# 启动命令: torchrun --nproc_per_node=4 train_fsdp.py
```

### F.3 FSDP 显存分析

```python
def fsdp_memory_analysis():
    """
    FSDP 各分片策略的显存分析
    """
    # 假设模型参数量 7B, FP16
    model_params_gb = 7e9 * 2 / 1024**3  # ~14 GB
    
    # 优化器状态 (AdamW): 2 份参数量
    optimizer_states_gb = model_params_gb * 2  # ~28 GB
    
    # 梯度: 1 份参数量
    gradients_gb = model_params_gb  # ~14 GB
    
    analysis = """
    === FSDP 显存分析 (7B 模型, FP16, AdamW) ===
    
    单卡总显存需求（不分片）:
    - 参数:     {params:.1f} GB
    - 优化器:   {opt:.1f} GB
    - 梯度:     {grad:.1f} GB
    - 激活值:   ~8 GB (取决于 batch/seq_len)
    - 总计:     ~{total:.0f} GB  ← 远超单卡显存!
    
    FSDP FULL_SHARD (ZeRO-3) 分片后 (4 卡):
    - 参数:     {params:.1f} / 4 = {params/4:.1f} GB
    - 优化器:   {opt:.1f} / 4 = {opt/4:.1f} GB
    - 梯度:     {grad:.1f} / 4 = {grad/4:.1f} GB
    - 激活值:   ~8 GB (不分片)
    - 总计:     ~{total_fsdp:.0f} GB  ← 可在 4×24GB 上运行!
    
    FSDP + CPU Offload (极限模式):
    - GPU 显存: 仅需 ~20 GB (参数和优化器在 CPU)
    - 代价: 训练速度降低 30-50%
    """.format(
        params=model_params_gb,
        opt=optimizer_states_gb,
        grad=gradients_gb,
        total=model_params_gb + optimizer_states_gb + gradients_gb + 8,
        total_fsdp=(model_params_gb + optimizer_states_gb + gradients_gb)/4 + 8,
    )
    
    print(analysis)

fsdp_memory_analysis()
```

---

## 附录 G：分布式通信优化 — 灵力传导加速

### G.1 通信原语详解

```python
"""
分布式训练中的通信原语
"""

# ========== 1. 点对点通信 (Point-to-Point) ==========

def point_to_point_communication():
    """
    send/recv: 进程间直接通信
    """
    import torch.distributed as dist
    
    rank = dist.get_rank()
    
    if rank == 0:
        # 发送数据
        data = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dist.send(data, dst=1)
        print("Rank 0: 已发送数据")
    elif rank == 1:
        # 接收数据
        data = torch.zeros(3).cuda()
        dist.recv(data, src=0)
        print(f"Rank 1: 收到数据 {data}")


# ========== 2. 集合通信 (Collective Communication) ==========

def collective_communication():
    """
    集合通信：所有进程参与的通信操作
    """
    import torch.distributed as dist
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data = torch.tensor([rank * 1.0]).cuda()
    
    # --- Broadcast: 一对多广播 ---
    # Rank 0 广播数据给所有进程
    dist.broadcast(data, src=0)
    print(f"Rank {rank}: Broadcast 结果 = {data.item()}")
    
    # --- Reduce: 多对一规约 ---
    # 所有进程的数据求和，结果到 Rank 0
    result = torch.zeros(1).cuda()
    dist.reduce(result, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Reduce 结果 = {result.item()}")  # 0+1+...+(N-1)
    
    # --- All-Reduce: 多对多规约 ---
    # 所有进程的数据求和，结果发给所有进程
    result = torch.zeros(1).cuda()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: All-Reduce 结果 = {result.item()}")
    
    # --- All-Gather: 收集所有进程的数据 ---
    gathered = [torch.zeros(1).cuda() for _ in range(world_size)]
    dist.all_gather(gathered, data)
    print(f"Rank {rank}: All-Gather = {[g.item() for g in gathered]}")
    
    # --- Scatter: 分发数据 ---
    scatter_data = torch.tensor([10.0, 20.0, 30.0, 40.0]).cuda()
    received = torch.zeros(1).cuda()
    dist.scatter(received, scatter_list=list(
        scatter_data.chunk(world_size)) if rank == 0 else None, src=0)
    print(f"Rank {rank}: Scatter 收到 = {received.item()}")
```

### G.2 通信优化技术

```python
def communication_optimization():
    """
    通信优化策略总结
    """
    optimization = """
    === 分布式训练通信优化 ===
    
    1. 梯度压缩
       - 通信量减少 10-100 倍
       - 精度损失可控 (< 1%)
       - 常用: FP8 压缩、随机量化、Top-K稀疏化
    
    2. 通信与计算重叠
       - All-Reduce 与反向传播重叠
       - 减少等待时间 30-50%
       - FSDP: backward_prefetch 参数控制
    
    3. 通信后端选择
       - NCCL (GPU): NVIDIA GPU 专用，最快
       - Gloo (CPU/GPU): CPU 为主，跨平台
       - MPI: 传统 HPC 环境
    
    4. 拓扑感知
       - 同节点内: NVLink (300 GB/s)
       - 跨节点间: InfiniBand (100-400 Gbps)
       - 避免跨交换机通信
    
    5. 梯度累积减少通信
       - 累积多步再同步
       - 通信次数减少 N 倍
       - 但延迟增加
    
    6. 通信分桶 (Bucket)
       - 将小张量的梯度合并成大桶
       - 减少通信次数
       - FSDP/DDP 默认启用
    """
    print(optimization)

communication_optimization()
```

### G.3 通信瓶颈诊断

```python
import torch

class CommunicationProfiler:
    """
    通信瓶颈诊断工具
    """
    
    @staticmethod
    def estimate_communication_time(
        data_size_gb: float,
        bandwidth_gbps: float = 100.0,
        latency_ms: float = 0.1
    ):
        """
        估算通信时间
        Args:
            data_size_gb: 数据大小 (GB)
            bandwidth_gbps: 网络带宽 (Gbps)
            latency_ms: 延迟 (ms)
        """
        # 传输时间
        transfer_time = data_size_gb * 8 / bandwidth_gbps  # 秒
        
        # 延迟（All-Reduce 需要 log(N) 次通信）
        import math
        num_hops = math.ceil(math.log2(8))  # 假设 8 卡
        total_latency = latency_ms / 1000 * num_hops
        
        total = transfer_time + total_latency
        
        print(f"=== 通信时间估算 ===")
        print(f"数据大小: {data_size_gb:.3f} GB")
        print(f"带宽: {bandwidth_gbps:.0f} Gbps")
        print(f"传输时间: {transfer_time*1000:.1f} ms")
        print(f"延迟开销: {total_latency*1000:.1f} ms ({num_hops} hops)")
        print(f"总通信时间: {total*1000:.1f} ms")
        return total
    
    @staticmethod
    def compute_to_comm_ratio(
        compute_time_ms: float,
        comm_time_ms: float
    ):
        """
        计算通信比 — 判断是否通信瓶颈
        """
        ratio = compute_time_ms / max(comm_time_ms, 0.001)
        print(f"\n计算/通信比: {ratio:.2f}")
        if ratio > 10:
            print("✅ 计算主导，通信不是瓶颈")
        elif ratio > 2:
            print("🟡 通信有一定影响，可优化")
        else:
            print("🔴 通信瓶颈！需要减少通信量")
        return ratio

# 使用示例
profiler = CommunicationProfiler()

# DDP: 每步需要 All-Reduce 梯度 (7B FP16 模型)
comm_time = profiler.estimate_communication_time(
    data_size_gb=14.0,  # 7B * 2 bytes
    bandwidth_gbps=300,  # NVLink
    latency_ms=0.01
)

profiler.compute_to_comm_ratio(
    compute_time_ms=500,  # 前向+反向
    comm_time_ms=comm_time * 1000
)
```

---

## 附录 H：容错与断点续训 — 修炼不中断术

### H.1 断点续训实现

```python
import os
import torch
import random
from pathlib import Path

class CheckpointManager:
    """
    断点续训管理器
    """
    
    def __init__(self, save_dir: str = "checkpoints",
                 max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_keep = max_keep  # 最多保留的检查点数量
    
    def save(self, model, optimizer, scheduler, epoch, step,
             best_metric=None, is_best=False):
        """
        保存训练检查点
        """
        # 获取模型状态（处理 FSDP/DDP）
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': best_metric,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': __import__('numpy').random.get_state(),
                'random': random.getstate(),
            }
        }
        
        # 常规检查点
        filepath = self.save_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, filepath)
        
        # 最佳模型
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  最佳模型已保存: {best_path}")
        
        # 清理旧检查点
        self._cleanup()
        
        print(f"  检查点已保存: {filepath}")
        return str(filepath)
    
    def load(self, model, optimizer=None, scheduler=None,
             resume_path: str = None):
        """
        加载检查点（断点续训）
        """
        if resume_path is None:
            # 找到最新的检查点
            checkpoints = sorted(self.save_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                print("未找到检查点，从头开始训练")
                return 0, 0
            resume_path = str(checkpoints[-1])
        
        print(f"从检查点恢复: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        # 加载模型权重
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复随机状态（确保可复现）
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng['cuda'])
            __import__('numpy').random.set_state(rng['numpy'])
            random.setstate(rng['random'])
        
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        best = checkpoint.get('best_metric', None)
        
        print(f"  已恢复到 epoch {epoch}, step {step}")
        if best is not None:
            print(f"  最佳指标: {best:.4f}")
        
        return epoch, step
    
    def _cleanup(self):
        """清理旧检查点，保留最新的 max_keep 个"""
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        while len(checkpoints) > self.max_keep:
            old = checkpoints.pop(0)
            old.unlink()
            print(f"  已清理旧检查点: {old.name}")

# 使用示例
model = torch.nn.Linear(100, 10)
optimizer = torch.optim.Adam(model.parameters())
ckpt_mgr = CheckpointManager(save_dir="train_checkpoints", max_keep=3)

# 保存检查点
ckpt_mgr.save(model, optimizer, None, epoch=5, step=500,
              best_metric=0.95, is_best=True)

# 断点续训
epoch, step = ckpt_mgr.load(model, optimizer)
print(f"从 epoch={epoch}, step={step} 继续训练")
```

### H.2 训练监控与告警

```python
import time
import json
from pathlib import Path

class TrainingMonitor:
    """
    训练监控器 — 实时追踪训练状态
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_log = self.log_dir / "metrics.jsonl"
        self.alerts = []
        self.history = {
            'loss': [],
            'lr': [],
            'grad_norm': [],
            'gpu_memory': [],
            'throughput': [],
        }
    
    def log_step(self, step, loss, lr, grad_norm=None,
                 gpu_memory_gb=None, samples_per_sec=None):
        """
        记录单步指标
        """
        record = {
            'step': step,
            'loss': loss,
            'lr': lr,
            'timestamp': time.time(),
        }
        if grad_norm is not None:
            record['grad_norm'] = grad_norm
            self.history['grad_norm'].append(grad_norm)
        if gpu_memory_gb is not None:
            record['gpu_memory_gb'] = gpu_memory_gb
            self.history['gpu_memory'].append(gpu_memory_gb)
        if samples_per_sec is not None:
            record['samples_per_sec'] = samples_per_sec
            self.history['throughput'].append(samples_per_sec)
        
        self.history['loss'].append(loss)
        self.history['lr'].append(lr)
        
        # 追加写入 JSONL
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        # 检查异常
        self._check_anomalies(step, loss, grad_norm)
    
    def _check_anomalies(self, step, loss, grad_norm):
        """异常检测"""
        import math
        
        # Loss 异常
        if math.isnan(loss) or math.isinf(loss):
            alert = f"[Step {step}] Loss = {loss}! 立即停止训练!"
            self.alerts.append(alert)
            print(f"  🔴 {alert}")
        
        # Loss 突增
        if len(self.history['loss']) >= 10:
            recent = self.history['loss'][-5:]
            prev = self.history['loss'][-10:-5]
            recent_mean = sum(recent) / len(recent)
            prev_mean = sum(prev) / len(prev)
            if prev_mean > 0 and recent_mean > prev_mean * 3:
                alert = f"[Step {step}] Loss 突增 {recent_mean/prev_mean:.1f}x!"
                self.alerts.append(alert)
                print(f"  🟡 {alert}")
        
        # 梯度爆炸
        if grad_norm is not None and grad_norm > 1000:
            alert = f"[Step {step}] 梯度范数过大: {grad_norm:.1f}"
            self.alerts.append(alert)
            print(f"  🟡 {alert}")
    
    def summary(self):
        """生成训练摘要"""
        if not self.history['loss']:
            return
        
        print("\n=== 训练摘要 ===")
        print(f"总步数: {len(self.history['loss'])}")
        print(f"初始 Loss: {self.history['loss'][0]:.4f}")
        print(f"最终 Loss: {self.history['loss'][-1]:.4f}")
        print(f"最低 Loss: {min(self.history['loss']):.4f}")
        
        if self.history['throughput']:
            avg_tp = sum(self.history['throughput']) / len(self.history['throughput'])
            print(f"平均吞吐量: {avg_tp:.1f} samples/sec")
        
        if self.alerts:
            print(f"\n异常告警 ({len(self.alerts)} 次):")
            for a in self.alerts[-5:]:
                print(f"  {a}")
        else:
            print("\n✅ 无异常告警")

# 使用示例
monitor = TrainingMonitor("experiment_logs")
import random
for step in range(50):
    loss = 2.0 * (0.95 ** step) + random.uniform(-0.05, 0.05)
    monitor.log_step(step, loss, lr=0.001 * (0.99 ** step),
                     grad_norm=random.uniform(0.5, 5.0),
                     gpu_memory_gb=random.uniform(8, 12))
monitor.summary()
```

### H.3 大规模训练容错策略

```python
def fault_tolerance_strategies():
    """
    大规模分布式训练的容错策略
    """
    strategies = """
    === 大规模训练容错策略 ===
    
    1. 检查点频率策略
       - 小模型: 每 5-10 个 epoch 保存
       - 大模型: 每 1000-2000 步保存
       - 关键节点: 每个 epoch 结束时保存
       - 成本: 每次保存 7B 模型约 14GB, 耗时 30-60s
    
    2. 异步检查点
       - 在训练继续时，后台线程保存检查点
       - 减少训练中断时间
       - FSDP: use_dist_checkpointing=True
    
    3. 节点故障恢复
       - 检测失败节点: 心跳机制 (30s 超时)
       - 自动重试: 最多 3 次
       - 从最新检查点恢复
       - 丢弃故障节点的未同步梯度
    
    4. 数据一致性
       - 确保所有 rank 保存相同的 RNG 状态
       - 恢复后跳过已完成的步骤
       - 避免数据重复/丢失
    
    5. 健康检查脚本
       - 定期检查 GPU 温度/显存/利用率
       - 检查网络延迟和丢包率
       - 磁盘空间监控
       - 自动降级策略
    
    6. 预算与时间控制
       - 最大训练时间限制
       - 最大计算预算限制
       - 优雅退出: 保存检查点后再终止
    """
    print(strategies)

fault_tolerance_strategies()
```


*焚诀曰：修炼之道，在于突破。当你感到被束缚，那就是突破的时刻。*

*斗气化翼，翱翔九天。*
