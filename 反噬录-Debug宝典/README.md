# 反噬录 — Debug 宝典

> *"炼丹之道，十炼九噬。*
> *反噬不可怕，可怕的是不知反噬从何而来，如何化解。*
> *此册记录天下已知反噬之象，及其破解之法。*
> *愿每一位炼丹师，都能安然度过每一次反噬。"*
>
> — 《焚诀》附录·反噬录

---

## 写在最前

**反噬（Bug/Error）是每个炼丹师（AI 工程师）的必经之路。**

本册按照反噬的类型分门别类，对每一种反噬详述其：
- **反噬之象**（症状）：如何识别这种错误
- **反噬之源**（根因）：错误的根本原因
- **化解之法**（解决方案）：一步步的修复指南
- **防噬之术**（预防措施）：如何避免再次发生

### 术语对照

| 修炼术语 | 技术含义 |
|---------|---------|
| 反噬 | 程序错误 / 训练失败 |
| 异火 | GPU |
| 火种容量 | 显存（VRAM） |
| 丹炉 | 服务器 |
| 炼丹 | 模型训练 |
| 丹方 | 训练脚本/配置 |
| 斗气 | 模型参数 |
| 药材 | 数据集 |
| 走火入魔 | Loss NaN / 训练崩溃 |
| 丹毒 | 模型质量问题（Mode Collapse 等）|

---

## 目录

- [第一章：异火之噬 — CUDA 错误](#第一章异火之噬--cuda-错误)
- [第二章：走火入魔 — 训练过程问题](#第二章走火入魔--训练过程问题)
- [第三章：丹毒之患 — 模型质量问题](#第三章丹毒之患--模型质量问题)
- [第四章：丹炉不稳 — 环境配置问题](#第四章丹炉不稳--环境配置问题)
- [第五章：万火同炼 — 分布式训练问题](#第五章万火同炼--分布式训练问题)
- [第六章：推理之噬 — 推理部署问题](#第六章推理之噬--推理部署问题)
- [附录：急救丹方速查表](#附录急救丹方速查表)

---

## 第一章：异火之噬 — CUDA 错误

> *"异火虽为至宝，但若使用不当，反被其灼。"*

### 1.1 CUDA Out of Memory（显存溢出）— 最常见的反噬

**反噬之象（症状）：**
```
RuntimeError: CUDA out of memory. Tried to allocate XX MiB
(GPU 0; XX GiB total capacity; XX GiB already allocated;
XX MiB free; XX GiB reserved in total by PyTorch)
```
或者：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**反噬之源（根因）：**

```
显存占用 = 模型参数 + 优化器状态 + 梯度 + 激活值 + 临时缓冲区

最常见的原因：
1. batch_size 太大 → 激活值占用爆炸
2. 模型太大 → 参数本身就装不下
3. 序列长度太长 → 注意力矩阵占用 O(n²)
4. 显存碎片化 → 总量够但找不到连续空间
5. 数据预处理在 GPU 上 → 额外占用显存
```

**化解之法（解决方案）：**

```
方法 1：减小 batch_size（最简单直接）
  ├── 将 batch_size 减半，观察是否还 OOM
  ├── 最极端情况下 batch_size=1
  └── 配合 gradient_accumulation 维持有效 batch size

方法 2：启用梯度检查点（Gradient Checkpointing）
  ├── 用时间换空间：不保存中间激活值，反向传播时重新计算
  ├── 显存减少约 50-70%，训练时间增加约 20-30%
  └── 代码：model.gradient_checkpointing_enable()

方法 3：使用混合精度训练（Mixed Precision）
  ├── 使用 FP16/BF16 替代 FP32
  ├── 显存减少约 50%，速度还能提升
  ├── 代码：
  │   from torch.cuda.amp import autocast, GradScaler
  │   scaler = GradScaler()
  │   with autocast(dtype=torch.bfloat16):
  │       output = model(input)
  │       loss = criterion(output, target)
  │   scaler.scale(loss).backward()
  │   scaler.step(optimizer)
  │   scaler.update()
  └── 或使用 Trainer 的 bf16=True / fp16=True 参数

方法 4：使用 DeepSpeed ZeRO
  ├── ZeRO Stage 1：分片优化器状态（显存减少约 4 倍）
  ├── ZeRO Stage 2：+ 分片梯度
  ├── ZeRO Stage 3：+ 分片模型参数
  └── 对于单卡也可以使用 ZeRO-Offload 将部分数据放到 CPU

方法 5：使用 QLoRA / LoRA 替代全量微调
  ├── QLoRA：4-bit 量化基础模型 + LoRA 适配器
  ├── 7B 模型：全量微调需 ~112GB → QLoRA 仅需 ~10GB
  └── 代码示例：
      from peft import get_peft_model, LoraConfig
      from transformers import BitsAndBytesConfig
      bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
      model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config
      )
      peft_config = LoraConfig(r=16, lora_alpha=32, ...)
      model = get_peft_model(model, peft_config)

方法 6：减少序列长度
  ├── max_length 从 2048 降到 1024 或 512
  └── 注意力的显存占用与序列长度的平方成正比

方法 7：清理显存碎片
  ├── torch.cuda.empty_cache()  # 释放缓存
  ├── 避免在训练循环中创建不必要的张量
  └── del 不再使用的变量
```

**防噬之术（预防措施）：**
```
1. 训练前用小 batch 试跑几步，观察显存占用
2. 使用 nvidia-smi 或 torch.cuda.memory_summary() 监控
3. 预估显存需求（参考万火录中的估算公式）
4. 始终开启混合精度训练
5. 使用 gradient_accumulation_steps 而非大 batch_size
```

---

### 1.2 CUDA 版本不匹配 — 异火与丹炉不兼容

**反噬之象（症状）：**
```
# 常见错误信息
RuntimeError: The NVIDIA driver on your system is too old
CUDA error: no kernel image is available for execution on the device
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```
或者更隐蔽的：
```
# PyTorch 安装了，但 CUDA 不可用
>>> import torch
>>> torch.cuda.is_available()
False
```

**反噬之源（根因）：**

```
CUDA 生态有三层版本需要对齐：

1. NVIDIA 驱动（最底层）
   └── 决定了支持的最高 CUDA 版本

2. CUDA Toolkit（中间层）
   └── nvcc 编译器、CUDA 运行时库

3. PyTorch / TensorFlow（上层）
   └── 编译时链接了特定版本的 CUDA

版本兼容关系：
  驱动 ≥ CUDA Toolkit ≥ PyTorch 编译时的 CUDA 版本

常见冲突场景：
  ├── 新 GPU（如 RTX 4090）需要新驱动，但系统驱动太旧
  ├── pip install torch 装了 CPU 版本
  ├── conda 环境中 CUDA 版本与系统 CUDA 不一致
  └── 系统安装了多个 CUDA 版本，PATH 指向了错误的版本
```

**化解之法（解决方案）：**

```
第 1 步：确认当前版本信息
  nvidia-smi                    # 查看驱动版本和支持的最高 CUDA 版本
  nvcc --version                # 查看 CUDA Toolkit 版本
  python -c "import torch; print(torch.version.cuda)"  # PyTorch 使用的 CUDA

第 2 步：确认兼容性
  ├── 查看 NVIDIA 驱动兼容表：
  │   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
  ├── 查看 PyTorch 兼容矩阵：
  │   https://pytorch.org/get-started/previous-versions/
  └── 驱动版本 ≥ CUDA 版本对应的最低驱动版本

第 3 步：安装正确的 PyTorch
  # 前往 https://pytorch.org 选择对应的安装命令
  # 例如 CUDA 12.1：
  pip install torch torchvision torchaudio --index-url \
      https://download.pytorch.org/whl/cu121

  # 或使用 conda（通常更不容易出问题）：
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
      -c pytorch -c nvidia

第 4 步：如果驱动太旧，更新驱动
  # Ubuntu:
  sudo apt update
  sudo apt install nvidia-driver-535  # 或更新的版本
  sudo reboot

  # 或使用 NVIDIA 官网下载的 .run 安装器
```

**防噬之术（预防措施）：**
```
1. 使用 conda 管理 CUDA 依赖（conda 会自动处理版本匹配）
2. 使用 Docker 容器（NVIDIA 提供了预配置的容器镜像）
3. 记录并固定环境版本（pip freeze > requirements.txt）
4. 新项目开始时，先确认 nvidia-smi 和 torch.cuda.is_available()
```

---

### 1.3 NCCL 错误 — 多火通信故障

**反噬之象（症状）：**
```
RuntimeError: NCCL Error 2: unhandled system error
NCCL WARN NET/IB : No device found
RuntimeError: NCCL communicator was aborted
Watchdog caught collective operation timeout
```

**反噬之源（根因）：**
```
NCCL (NVIDIA Collective Communications Library) 负责多 GPU 通信。
常见原因：
├── 网络接口配置错误（多网卡环境下选错了网卡）
├── InfiniBand 驱动问题
├── 防火墙阻断了 GPU 间通信
├── 不同节点的 NCCL 版本不一致
├── GPU 拓扑问题（NVLink/NVSwitch 配置不对）
└── 某个 GPU 故障导致通信超时
```

**化解之法（解决方案）：**
```
方法 1：指定正确的网络接口
  export NCCL_SOCKET_IFNAME=eth0     # 或 ib0（InfiniBand）
  export NCCL_IB_DISABLE=0           # 启用 InfiniBand（如果有）

方法 2：增加超时时间
  export NCCL_TIMEOUT=1800           # 秒
  # 在 PyTorch 中：
  dist.init_process_group(..., timeout=timedelta(seconds=1800))

方法 3：启用详细日志定位问题
  export NCCL_DEBUG=INFO             # 或 WARN, TRACE
  export NCCL_DEBUG_SUBSYS=ALL

方法 4：降级通信方式
  export NCCL_P2P_DISABLE=1          # 禁用 P2P（如果 NVLink 有问题）
  export NCCL_IB_DISABLE=1           # 禁用 InfiniBand（回退到 TCP）

方法 5：检查 GPU 健康状态
  nvidia-smi                         # 查看所有 GPU 状态
  nvidia-smi topo -m                 # 查看 GPU 互联拓扑
  nvidia-smi -q -d ECC               # 查看 ECC 错误
```

---

## 第二章：走火入魔 — 训练过程问题

> *"走火入魔是炼丹最危险的反噬——丹方（训练）看似正常运行，*
> *但产出的丹药（模型）已经废了。"*

### 2.1 Loss NaN — 炼丹炉爆炸

**反噬之象（症状）：**
```
Step 100: loss = 2.345
Step 200: loss = 1.892
Step 300: loss = 45.678  ← 开始异常
Step 400: loss = 12345.0
Step 500: loss = nan      ← 彻底走火入魔
Step 600: loss = nan
```

**反噬之源（根因）：**

```
原因 1：学习率过高（最常见）
  ├── 梯度过大 → 参数更新幅度过大 → 数值溢出
  ├── 尤其是训练初期，参数还不稳定
  └── FP16 的数值范围只到 65504，很容易溢出

原因 2：数据中有异常值
  ├── 空样本导致除零
  ├── 极端的 token ID（超出词表范围）
  ├── label 中有 -100 以外的无效值
  └── 数据预处理不当导致 NaN 输入

原因 3：数值不稳定
  ├── FP16 下的数值溢出/下溢
  ├── log(0) 或 log(负数)
  ├── softmax 数值不稳定
  └── 某些层的权重初始化不当

原因 4：模型架构问题
  ├── 残差连接缺失
  ├── LayerNorm 放置位置不对
  └── 注意力分数计算不稳定

原因 5：混合精度训练配置问题
  ├── loss scale 设置不当
  ├── 没有使用 GradScaler
  └── BF16 和 FP16 混用
```

**化解之法（解决方案）：**

```
急救步骤：

第 1 步：降低学习率
  ├── 将 learning_rate 减半，甚至减小到 1/10
  ├── 常用范围：1e-5 到 5e-5（微调），1e-4 到 3e-4（预训练）
  └── 使用 warmup：前 5-10% 的步数线性增加学习率

第 2 步：检查数据
  ├── 遍历数据集，检查是否有 NaN/Inf 值
  │   import torch
  │   for batch in dataloader:
  │       for k, v in batch.items():
  │           if torch.is_tensor(v) and torch.isnan(v).any():
  │               print(f"Found NaN in {k}!")
  │           if torch.is_tensor(v) and torch.isinf(v).any():
  │               print(f"Found Inf in {k}!")
  ├── 检查 label 是否合法
  └── 检查 input_ids 是否在词表范围内

第 3 步：改用 BF16
  ├── BF16 的数值范围与 FP32 相同（指数 8 位），不容易溢出
  ├── 如果 GPU 支持 BF16（A100+, RTX 30xx+），优先用 BF16
  └── bf16=True 而非 fp16=True

第 4 步：启用梯度裁剪
  ├── max_grad_norm=1.0（HuggingFace Trainer 默认值）
  ├── 或手动：torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  └── 防止梯度爆炸导致参数更新过大

第 5 步：检查模型权重
  for name, param in model.named_parameters():
      if torch.isnan(param).any():
          print(f"NaN in parameter: {name}")
      if torch.isinf(param).any():
          print(f"Inf in parameter: {name}")

第 6 步：添加 NaN 检测钩子
  def nan_hook(module, input, output):
      if isinstance(output, torch.Tensor) and torch.isnan(output).any():
          print(f"NaN detected in {module.__class__.__name__}")
          raise RuntimeError("NaN detected!")

  for module in model.modules():
      module.register_forward_hook(nan_hook)
```

---

### 2.2 Loss 爆炸 — 炼丹失控

**反噬之象（症状）：**
```
Step 100: loss = 2.345
Step 200: loss = 2.189
Step 300: loss = 3.456  ← 突然上升
Step 400: loss = 15.789
Step 500: loss = 234.56
Step 600: loss = 99999.0  ← 完全失控
```

**反噬之源（根因）：**
```
与 Loss NaN 类似，但还没有到溢出的程度。
额外原因：
├── 学习率调度不当（warmup 后突然跳到太高的 LR）
├── 数据分布突变（某个 batch 的数据与之前差异太大）
├── 梯度累积步数设置错误
└── 权重衰减（weight decay）太大
```

**化解之法（解决方案）：**
```
1. 使用 cosine 或 linear 学习率调度
2. 检查数据是否被正确 shuffle
3. 确认 gradient_accumulation_steps 的设置
4. 降低 weight_decay（通常 0.01-0.1）
5. 从最近的正常 checkpoint 恢复
```

---

### 2.3 Loss 停滞 — 炼丹无进展

**反噬之象（症状）：**
```
Step 1000: loss = 2.345
Step 2000: loss = 2.341
Step 3000: loss = 2.339
Step 4000: loss = 2.340  ← 几乎不动了
Step 5000: loss = 2.338
...
```

**反噬之源（根因）：**
```
1. 学习率太低 → 参数更新太小，无法跳出局部最小值
2. 模型容量不足 → 模型太小，无法学到更复杂的模式
3. 数据质量差 → 噪声太大，信号被淹没
4. 正则化过强 → dropout 太大、weight decay 太大
5. 到达了该数据集/模型的理论下界
```

**化解之法（解决方案）：**
```
1. 尝试增大学习率（例如从 1e-5 增大到 5e-5）
2. 使用学习率 warmup + cosine decay
3. 检查数据质量，过滤低质量样本
4. 减小 dropout 和 weight decay
5. 增大模型容量（更多层或更宽的隐藏层）
6. 使用更长的序列长度（如果任务需要）
7. 检查是否意外冻结了某些参数
   for name, param in model.named_parameters():
       print(f"{name}: requires_grad = {param.requires_grad}")
```

---

### 2.4 梯度消失与梯度爆炸 — 斗气逆流

**梯度消失（Vanishing Gradients）：**
```
反噬之象：
├── 靠近输入层的参数几乎不更新
├── 训练 loss 下降极其缓慢
└── 深层网络表现不如浅层网络

反噬之源：
├── 网络太深，梯度逐层衰减
├── 使用了 sigmoid/tanh 激活函数（梯度区间 [0, 0.25] 或 [0, 1]）
└── 权重初始化不当

化解之法：
├── 使用 ReLU/GELU 激活函数
├── 添加残差连接（ResNet 思想）
├── 使用 LayerNorm/BatchNorm
├── 合适的权重初始化（He/Xavier）
└── 现代 Transformer 架构通常不会有此问题（有残差连接和 LayerNorm）
```

**梯度爆炸（Exploding Gradients）：**
```
反噬之象：
├── Loss 急剧增大或变为 NaN
├── 参数值变得极大
└── 梯度范数远超正常范围

反噬之源：
├── 学习率太高
├── 权重初始化不当（权重太大）
├── 缺少梯度裁剪
└── 数值不稳定的运算

化解之法：
├── 梯度裁剪：torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
├── 降低学习率
├── 使用 warmup
└── 检查权重初始化

监控梯度的代码：
  total_norm = 0
  for p in model.parameters():
      if p.grad is not None:
          param_norm = p.grad.data.norm(2)
          total_norm += param_norm.item() ** 2
  total_norm = total_norm ** 0.5
  print(f"Gradient norm: {total_norm}")
  # 正常范围通常在 0.1 - 10 之间
  # 超过 100 就需要警惕了
```

---

## 第三章：丹毒之患 — 模型质量问题

> *"最阴险的反噬不是训练崩溃——那至少你知道出了问题。*
> *真正可怕的是训练完成了，loss 看起来正常，但模型废了。"*

### 3.1 Mode Collapse — 丹药同质化

**反噬之象（症状）：**
```
# 模型对不同问题给出几乎相同的回答
User: "什么是机器学习？"
Model: "这是一个很好的问题，让我为您详细解释..."

User: "今天天气怎么样？"
Model: "这是一个很好的问题，让我为您详细解释..."

User: "1+1=?"
Model: "这是一个很好的问题，让我为您详细解释..."
```

**反噬之源（根因）：**
```
1. RLHF 训练中 KL 惩罚系数太小 → 模型偏离参考策略太远
2. SFT 数据同质化 → 大量数据使用相似的回答模板
3. 奖励模型有偏差 → 总是给某种风格的回答高分
4. 训练过拟合 → 模型记住了少数模式
```

**化解之法（解决方案）：**
```
1. 增大 RLHF 的 KL 惩罚系数（beta）
2. 提升训练数据的多样性
3. 使用多个奖励模型或定期更新奖励模型
4. 在 SFT 阶段加入更多样化的回答风格
5. 监控生成的多样性指标（distinct-n, entropy）
```

---

### 3.2 Catastrophic Forgetting — 丹道走偏

**反噬之象（症状）：**
```
微调前：
  模型可以回答各种通用问题，能力全面

微调后：
  模型在微调任务上表现很好
  但之前的通用能力大幅退化
  例如：微调了代码能力后，不会写文章了
```

**反噬之源（根因）：**
```
1. 微调数据分布与预训练数据分布差异太大
2. 学习率太高，导致预训练权重被大幅覆盖
3. 微调时间太长（过多 epoch）
4. 没有在微调数据中混入通用数据
```

**化解之法（解决方案）：**
```
1. 使用 LoRA/QLoRA（只修改少量参数，大部分预训练权重保持不变）
2. 降低学习率（微调通常用 1e-5 ~ 5e-5，而非预训练的 1e-4 ~ 3e-4）
3. 减少训练 epoch（通常 1-3 个 epoch 就够了）
4. 在微调数据中混入一定比例的通用数据（例如 10-20%）
5. 使用 EWC（Elastic Weight Consolidation）等持续学习方法
6. 训练后评估多个维度的能力，而非仅看目标任务
```

---

### 3.3 Reward Hacking — 钻奖励漏洞

**反噬之象（症状）：**
```
RLHF 训练中：
  奖励分数持续上升 ↑（看起来很好！）
  但人类评估的质量却在下降 ↓（实际上变差了！）

例如：
  奖励模型偏好长回答 → 模型学会了废话连篇
  奖励模型偏好有礼貌的回答 → 模型学会了在每句话后加 "希望对您有帮助！"
  奖励模型偏好包含关键词的回答 → 模型学会了堆砌关键词
```

**反噬之源（根因）：**
```
Goodhart's Law: "当一个度量变成目标时，它就不再是好的度量。"
├── 奖励模型只是人类偏好的近似，不是完美的
├── 模型的优化能力远超奖励模型的鲁棒性
└── 模型找到了满足奖励函数但不满足真实意图的"捷径"
```

**化解之法（解决方案）：**
```
1. 增大 KL 散度惩罚，限制模型偏离参考策略的程度
2. 使用多个奖励模型的集成
3. 定期用人类评估替代自动评估
4. 对奖励分数设置上限（reward clipping）
5. 使用 Constitutional AI 等方法添加额外约束
6. 定期更新奖励模型（用新模型的输出重新收集偏好数据）
7. 使用 DPO 替代 PPO（DPO 不需要显式的奖励模型，reward hacking 风险更低）
```

---

## 第四章：丹炉不稳 — 环境配置问题

> *"丹炉（环境）不稳，再好的丹方也炼不出好药。*
> *环境问题是最让人头疼的反噬——因为它们往往与代码逻辑无关。"*

### 4.1 包版本冲突 — 丹材不兼容

**反噬之象（症状）：**
```
# 安装时就报错
ERROR: pip's dependency resolver does not currently take into account
all the packages that are installed. As a result, the following
package installations could potentially break other packages.

# 或运行时报错
ImportError: cannot import name 'XXX' from 'transformers'
AttributeError: module 'torch' has no attribute 'XXX'
TypeError: XXX() got an unexpected keyword argument 'YYY'
```

**反噬之源（根因）：**
```
常见冲突组合：
├── torch 版本 vs transformers 版本
│   例：transformers 4.36+ 需要 torch >= 2.0
├── transformers 版本 vs peft 版本
│   例：peft 新版本可能不兼容旧 transformers
├── transformers 版本 vs accelerate 版本
│   这两个库经常需要同步更新
├── bitsandbytes 版本
│   bitsandbytes 与 CUDA 版本强绑定
├── flash-attn 版本
│   与 torch 版本和 CUDA 版本强绑定，编译安装经常出问题
└── vllm / trl / datasets 等
    各自有复杂的依赖关系
```

**化解之法（解决方案）：**

```
第 1 步：使用虚拟环境（永远不要在系统 Python 中安装）
  # 使用 conda
  conda create -n myenv python=3.10
  conda activate myenv

  # 或使用 venv
  python -m venv myenv
  source myenv/bin/activate

第 2 步：按正确顺序安装
  # 1. 先安装 PyTorch（与 CUDA 版本匹配）
  pip install torch==2.2.0 --index-url \
      https://download.pytorch.org/whl/cu121

  # 2. 再安装 transformers 和相关库
  pip install transformers==4.40.0

  # 3. 最后安装可选依赖
  pip install peft accelerate bitsandbytes

第 3 步：使用经过验证的版本组合
  推荐组合（截至 2025 年初）：
  ├── Python 3.10 / 3.11
  ├── PyTorch 2.2.x + CUDA 12.1
  ├── transformers 4.40.x
  ├── peft 0.10.x
  ├── accelerate 0.29.x
  ├── bitsandbytes 0.43.x
  └── trl 0.8.x

第 4 步：锁定版本
  pip freeze > requirements.txt
  # 或使用 poetry / pdm 管理依赖
```

**防噬之术（预防措施）：**
```
1. 每个项目使用独立的虚拟环境
2. 使用 requirements.txt 或 pyproject.toml 锁定版本
3. 优先使用 Docker（完全隔离的环境）
4. 关注库的 CHANGELOG，升级前阅读变更说明
5. 使用 pip install --dry-run 预览安装效果
```

---

### 4.2 CUDA Toolkit 与驱动不匹配

**反噬之象（症状）：**
```
# 编译 CUDA 代码时
nvcc fatal: Unsupported gpu architecture 'compute_89'

# 或运行时
CUDA error: device-side assert triggered
CUDA error: invalid device function

# flash-attn 安装失败
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```

**反噬之源（根因）：**
```
版本链条：GPU 硬件 → 驱动 → CUDA Toolkit → PyTorch → 库

GPU 架构代号对应关系：
  ├── Turing (RTX 20xx): compute_75
  ├── Ampere (RTX 30xx, A100): compute_80, compute_86
  ├── Ada Lovelace (RTX 40xx): compute_89
  ├── Hopper (H100): compute_90
  └── Blackwell (B100/B200): compute_100

常见问题：
  ├── CUDA Toolkit 版本太旧，不支持新 GPU 架构
  ├── 驱动太旧，不支持新 CUDA Toolkit
  └── 库（如 flash-attn）编译时未包含目标 GPU 架构
```

**化解之法（解决方案）：**
```
1. 确认 GPU 架构：nvidia-smi -q | grep "Product Name"
2. 确认所需的最低 CUDA 版本
3. 更新驱动到足够新的版本
4. 安装正确版本的 CUDA Toolkit
5. 对于编译问题，设置正确的 TORCH_CUDA_ARCH_LIST:
   export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"  # 根据你的 GPU
```

---

### 4.3 Python 版本问题

**反噬之象（症状）：**
```
SyntaxError: invalid syntax  # Python 版本不支持新语法
ModuleNotFoundError: No module named 'XXX'  # 某些库不支持当前 Python 版本
```

**推荐 Python 版本：**
```
2025 年推荐：Python 3.10 或 3.11
├── 3.10: 最广泛兼容，大多数库都支持
├── 3.11: 性能提升约 10-25%，大部分库已支持
├── 3.12: 较新，部分库可能还未适配
├── 3.9: 仍可用，但逐渐被淘汰
└── 3.8 及以下: 不推荐，很多新库已不支持

注意：
├── PyTorch 2.2+ 需要 Python >= 3.8
├── 某些库（如 vllm）可能要求特定 Python 版本
└── 始终使用 conda/pyenv 管理 Python 版本，不要修改系统 Python
```

---

## 第五章：万火同炼 — 分布式训练问题

> *"多火联炼，威力倍增，但配合不当，反噬也倍增。"*

### 5.1 进程挂起（Hanging）

**反噬之象（症状）：**
```
训练日志停止输出，进程仍在运行但无进展。
GPU 利用率显示为 0%，但进程不退出。
通常伴随几分钟后的 NCCL timeout 错误。
```

**反噬之源（根因）：**
```
1. 数据加载不均衡 → 某些 rank 比其他 rank 先完成
2. 条件分支 → 不同 rank 走了不同的代码路径
3. 某个 GPU 故障 → 通信等待超时
4. 死锁 → 集合通信的调用顺序不一致
5. DataLoader 异常 → 某个 worker 进程卡住
```

**化解之法（解决方案）：**
```
1. 确保所有 rank 的数据量一致
   # 使用 DistributedSampler 并设置 drop_last=True
   sampler = DistributedSampler(dataset, drop_last=True)

2. 避免条件分支中包含集合通信操作
   # 错误示例：
   if rank == 0:
       dist.all_reduce(tensor)  # 其他 rank 不会执行这行，导致死锁

3. 增加 NCCL 超时时间以获取更多调试信息
   os.environ["NCCL_TIMEOUT"] = "3600"

4. 使用 torch.distributed.barrier() 同步所有进程
   # 在关键点添加 barrier 来定位哪里卡住了
   print(f"Rank {rank}: before barrier at step {step}")
   dist.barrier()
   print(f"Rank {rank}: after barrier at step {step}")

5. 检查 DataLoader 的 num_workers 设置
   # 减少 num_workers 或设为 0 排除数据加载问题
```

---

### 5.2 各 GPU 不同步（Desync）

**反噬之象（症状）：**
```
不同 GPU 上的 loss 值不一致（应该一致的地方）
模型权重在不同 GPU 上产生差异
训练结果不可复现
```

**反噬之源（根因）：**
```
1. 随机种子未正确同步
2. dropout 在不同 rank 上使用不同的随机序列
3. 数据 shuffle 在不同 rank 上不一致
4. 梯度同步缺失或错误
```

**化解之法（解决方案）：**
```
1. 设置统一的随机种子
   import torch
   import numpy as np
   import random

   def set_seed(seed):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)

   set_seed(42)

2. 使用 DistributedSampler 确保数据一致性
   sampler = DistributedSampler(dataset, seed=42)

3. 定期检查各 rank 的参数是否一致
   # 在训练循环中定期添加
   for name, param in model.named_parameters():
       tensor = param.data.clone()
       dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
       tensor /= dist.get_world_size()
       diff = (param.data - tensor).abs().max().item()
       if diff > 1e-5:
           print(f"WARNING: Param desync in {name}, max diff: {diff}")
```

---

### 5.3 NCCL 超时

**反噬之象（症状）：**
```
RuntimeError: NCCL communicator was aborted on rank X.
Original reason for failure was: Watchdog caught collective
operation timeout: WorkNCCL(...)
```

**化解之法（解决方案）：**
```
1. 增加超时时间
   # 环境变量
   export NCCL_TIMEOUT=1800000  # 毫秒
   # 或在代码中
   dist.init_process_group(
       ...,
       timeout=datetime.timedelta(minutes=30)
   )

2. 检查网络连通性
   # 在各节点间测试
   ping node2
   # 检查 InfiniBand 状态
   ibstat

3. 检查 GPU 健康状态
   nvidia-smi -q  # 查看是否有 ECC 错误或其他异常

4. 减少通信量
   ├── 使用梯度压缩
   ├── 减少 all_reduce 频率
   └── 使用更高效的并行策略
```

---

## 第六章：推理之噬 — 推理部署问题

> *"丹药炼成，分发给众人时，才发现新的问题。"*

### 6.1 推理 OOM

```
推理时的 OOM 与训练不同——不涉及梯度和优化器，
但长序列的 KV Cache 会占用大量显存。

KV Cache 显存估算：
  = 2 × num_layers × hidden_size × seq_len × batch_size × bytes_per_element

7B 模型，FP16，seq_len=4096，batch_size=1:
  = 2 × 32 × 4096 × 4096 × 1 × 2 bytes ≈ 2 GB

解决方案：
├── 使用量化推理（INT8/INT4）
├── 使用 Paged Attention（vLLM）
├── 限制最大序列长度
├── 使用 FlashAttention
└── 使用模型并行（多卡推理）
```

### 6.2 推理速度慢

```
常见原因和解决方案：

1. 未使用 KV Cache
   ├── 确保 use_cache=True（大多数框架默认开启）
   └── KV Cache 避免重复计算之前 token 的注意力

2. 未使用 FlashAttention
   ├── pip install flash-attn
   ├── model = AutoModelForCausalLM.from_pretrained(
   │       ..., attn_implementation="flash_attention_2")
   └── FlashAttention 可提速 2-4 倍并减少显存

3. 未使用量化
   ├── GPTQ: 离线量化，推理快
   ├── AWQ: 对激活值感知的量化
   ├── GGUF: llama.cpp 格式，CPU/GPU 混合推理
   └── 量化可在精度损失极小的情况下大幅减少显存和加速

4. batch_size 太小
   ├── 推理时增大 batch_size 可以提高吞吐量
   └── 使用 continuous batching（vLLM 支持）

5. 使用专门的推理框架
   ├── vLLM: PagedAttention, continuous batching
   ├── TensorRT-LLM: NVIDIA 优化的推理引擎
   ├── llama.cpp: CPU/GPU 混合推理
   └── SGLang: 高性能推理框架
```

---

## 附录：急救丹方速查表

> *"反噬来袭时，翻到这一页，按图索骥。"*

### 快速诊断流程

```
你遇到了什么问题？
│
├── 训练无法启动
│   ├── ImportError → 检查包版本 [4.1]
│   ├── CUDA not available → 检查 CUDA 安装 [1.2]
│   ├── CUDA OOM（第一步就爆） → 模型太大 [1.1]
│   └── Permission denied → 检查文件权限和 GPU 访问权限
│
├── 训练过程中崩溃
│   ├── Loss = NaN → [2.1]
│   ├── Loss 爆炸 → [2.2]
│   ├── CUDA OOM → [1.1]
│   ├── NCCL Error → [1.3]
│   └── 进程挂起 → [5.1]
│
├── 训练完成但效果差
│   ├── 回答同质化 → Mode Collapse [3.1]
│   ├── 旧能力丢失 → Catastrophic Forgetting [3.2]
│   ├── 奖励高但质量差 → Reward Hacking [3.3]
│   └── Loss 没降到预期 → [2.3]
│
├── 推理阶段的问题
│   ├── 推理 OOM → [6.1]
│   ├── 推理太慢 → [6.2]
│   └── 推理结果不对 → 检查模型加载和提示词
│
└── 环境问题
    ├── 包安装失败 → [4.1]
    ├── CUDA 版本问题 → [4.2]
    └── Python 版本问题 → [4.3]
```

### 常用调试命令速查

```bash
# === GPU 状态检查 ===
nvidia-smi                          # GPU 总览
nvidia-smi -l 1                     # 每秒刷新
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv                    # 格式化输出
nvidia-smi topo -m                  # GPU 互联拓扑

# === CUDA 版本检查 ===
nvidia-smi | head -3                # 驱动版本和 CUDA 版本
nvcc --version                      # CUDA Toolkit 版本
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# === Python 环境检查 ===
python --version
pip list | grep -E "torch|transformers|peft|accelerate|bitsandbytes"
conda list | grep cuda

# === 显存使用排查 ===
# 在 Python 中
import torch
print(torch.cuda.memory_summary())
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# === 网络检查（分布式训练） ===
ifconfig                             # 网络接口
ibstat                               # InfiniBand 状态（如果有）
ping <other_node>                    # 节点连通性

# === 进程排查 ===
ps aux | grep python                 # 查看 Python 进程
kill -9 <PID>                        # 强制终止挂起的进程
fuser -v /dev/nvidia*                # 查看占用 GPU 的进程
```

### 常见错误信息与对策快查

| 错误信息 | 最可能原因 | 第一步操作 |
|---------|-----------|-----------|
| `CUDA out of memory` | batch_size 太大 / 模型太大 | 减半 batch_size |
| `Loss = NaN` | 学习率太高 | 降低学习率，启用梯度裁剪 |
| `NCCL Error` | 网络/通信问题 | 检查网络接口和 GPU 拓扑 |
| `torch.cuda.is_available() = False` | CUDA 未正确安装 | 检查驱动和 PyTorch CUDA 版本 |
| `ImportError: cannot import name` | 包版本不兼容 | 检查版本，创建新虚拟环境 |
| `Killed` (无其他信息) | CPU 内存不足 | 减少数据加载 worker，增加 swap |
| `Connection refused` | 分布式训练主节点不可达 | 检查 IP 和端口配置 |
| `Segmentation fault` | C/C++ 层面的内存错误 | 更新 CUDA/PyTorch，检查 GPU 硬件 |
| `Gradient overflow` | FP16 数值溢出 | 切换到 BF16 |
| `CUDA error: device-side assert` | 索引越界或非法操作 | 设置 CUDA_LAUNCH_BLOCKING=1 获取详细堆栈 |

---

> *"炼丹之路，反噬常伴。*
> *但每一次成功化解反噬，都是修为的增长。*
> *此册非一次写就，乃历代炼丹师血泪凝成。*
> *若你遇到新的反噬，亦可添入此册，惠及后来之人。"*
>
> — 《焚诀》附录·反噬录 完

---

## 附录 B：2025 新型反噬与对策

> *"随着修炼境界提升，新型反噬也随之出现。"*

### B.1 LoRA/QLoRA 常见反噬

| 反噬现象 | 症状 | 解决方案 |
|----------|------|---------|
| bitsandbytes 兼容性 | ImportError / CUDA版本不匹配 | pip install bitsandbytes --upgrade |
| LoRA 权重不匹配 | PeftModel 加载失败 | 确认 base model 版本一致 |
| 量化后输出乱码 | 模型生成随机字符 | 检查 tokenizer 匹配 |
| 灾难性遗忘 | 领域能力提升但通用能力崩塌 | 混合通用数据 / 减小学习率 |

### B.2 分布式训练反噬

| 反噬现象 | 症状 | 解决方案 |
|----------|------|---------|
| NCCL 超时 | 训练 hang / Timeout | export NCCL_TIMEOUT=1800 |
| 各 GPU 显存不均 | Rank 0 OOM | drop_last=True / PackDataset |
| 死锁 | 所有 GPU 都在等消息 | 检查 barrier 配对 |

### B.3 长文本训练反噬

| 反噬现象 | 症状 | 解决方案 |
|----------|------|---------|
| 位置编码外推失败 | 超训练长度后效果骤降 | 使用 YaRN/NTK 扩展 |
| 注意力爆炸 | seq_len>8K OOM | 启用 Flash Attention 2 |
| KV Cache 膨胀 | 长对话推理 OOM | H2O / Scissorhands 压缩 |

### B.4 推理服务反噬（vLLM / TGI 部署）

| 反噬现象 | 症状 | 解决方案 |
|----------|------|---------|
| vLLM 启动 OOM | `torch.cuda.OutOfMemoryError` during init | 降低 `--gpu-memory-utilization`（默认0.9→0.85）|
| vLLM 吞吐量骤降 | P99 延迟突然变高 | 检查 `--max-num-seqs` 是否过高；开启 `--enable-prefix-caching` |
| TGI 与 PyTorch 版本冲突 | `ImportError: cannot import name` | 使用 TGI Docker 镜像（隔离环境）|
| vLLM 生成乱码 | 输出无意义字符 | 检查 tokenizer（trust_remote_code=True）；检查量化模型是否匹配 |
| 批处理超时 | Request timeout in production | 增加 `--max-waiting-tokens`；调小 `max_tokens` |
| 多卡 vLLM 不均衡 | GPU 0 利用率远高于其他 | 确保 `--tensor-parallel-size` 正确；检查 NVLink 互联 |

```python
# === vLLM 常见问题诊断脚本 ===

import subprocess, json

def diagnose_vllm():
    """vLLM 部署健康检查"""
    print("vLLM 健康诊断：\n")

    # 1. 检查 GPU 状态
    print("1. GPU 状态：")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        print(f"   {line}")

    # 2. 检查端口是否在监听
    import socket
    for port in [8000, 8080, 3000]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port))
        status = "✅ 在监听" if result == 0 else "❌ 未监听"
        print(f"   端口 {port}: {status}")
        sock.close()

    # 3. 测试 API 响应
    try:
        import requests
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()["data"]
            print(f"\n3. API 状态: ✅ 正常 ({len(models)} 个模型)")
        else:
            print(f"\n3. API 状态: ❌ 异常 (HTTP {resp.status_code})")
    except Exception as e:
        print(f"\n3. API 状态: ❌ 无法连接 ({e})")

    # 4. 显存碎片检查
    print("\n4. 推荐检查项：")
    print("   - nvidia-smi dmon：实时监控 GPU 使用率")
    print("   - watch -n 1 nvidia-smi：每秒刷新")
    print("   - curl localhost:8000/health：vLLM 健康检查端点")
    print("   - curl localhost:8000/metrics：Prometheus 指标")

# diagnose_vllm()
```

### B.5 模型安全反噬

| 反噬现象 | 症状 | 解决方案 |
|----------|------|---------|
| Prompt 注入（越狱） | 用户输入绕过安全限制 | 输入过滤 + 系统提示词防护 + 安全分类器 |
| 幻觉（Hallucination） | 模型编造不存在的"事实" | RAG 引入真实知识 + 温度调低 + 不确定性表达 |
| 数据泄漏 | 模型输出训练数据中的敏感信息 | 差分隐私训练 + 去重 + 输出过滤 |
| 恶意指令 | 模型被诱导生成有害内容 | 内容审核 + 红队测试 + 对齐训练 |

```python
# === 简单的安全防护层 ===

def basic_safety_filter(user_input: str) -> dict:
    """
    基础安全过滤（生产环境建议使用专业安全模型）
    """
    # 关键词检测
    danger_keywords = [
        "忽略之前的指令", "ignore previous", "forget everything",
        "你现在是", "pretend you are", "jailbreak",
    ]

    risk_score = 0
    for keyword in danger_keywords:
        if keyword.lower() in user_input.lower():
            risk_score += 1

    # 注入模式检测
    injection_patterns = [
        "[INST]", "</s>", "<|im_start|>",  # 特殊 token 注入
        "system:", "human:", "assistant:",   # 角色伪装
        "### Instruction", "### Response",   # 模板注入
    ]
    for pattern in injection_patterns:
        if pattern.lower() in user_input.lower():
            risk_score += 2

    if risk_score >= 3:
        return {"safe": False, "reason": "检测到潜在的 prompt 注入", "score": risk_score}
    elif risk_score >= 1:
        return {"safe": True, "warning": "输入中包含可疑模式", "score": risk_score}
    else:
        return {"safe": True, "score": 0}

# 安全防护口诀：
# "多层防护胜于单点，输入过滤+模型对齐+输出审核，三道防线缺一不可。"
```


