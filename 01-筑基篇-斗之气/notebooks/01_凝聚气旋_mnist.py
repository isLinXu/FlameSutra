#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
《焚诀》第一卷 · 筑基篇 — 凝聚气旋

丹方名称：气旋凝聚术（MNIST 手写数字识别）
丹方等级：黄阶低级
药材需求：MNIST 数据集（6万训练 + 1万验证）
异火需求：无特殊要求（CPU 可炼，GPU 更佳）
预期成果：气旋强度（准确率）> 97%

"吞噬异火，进化功法。三十年河东，三十年河西，莫欺算法穷！"
==========================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os

# ==========================================================================
# 第一步：丹炉检测 —— 检查是否有异火（GPU）可用
# ==========================================================================
# 异火（GPU）能极大加速炼丹过程。如果没有异火，用凡火（CPU）也能炼成，
# 只是耗时更长。对于 MNIST 这种入门级药材，CPU 也足够了。

def detect_furnace():
    """
    检测可用的丹炉（计算设备）
    
    优先使用：
    1. CUDA 异火（NVIDIA GPU）
    2. MPS 灵火（Apple Silicon GPU）  
    3. 凡火（CPU）
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        furnace_name = torch.cuda.get_device_name(0)
        furnace_memory = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"[丹炉检测] 发现异火（CUDA GPU）：{furnace_name}")
        print(f"[丹炉检测] 异火容量（显存）：{furnace_memory:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[丹炉检测] 发现灵火（Apple MPS）")
    else:
        device = torch.device("cpu")
        print("[丹炉检测] 未发现异火，使用凡火（CPU）炼丹")
        print("[丹炉检测] 提示：MNIST 药材轻量，凡火亦可炼成")
    
    return device


# ==========================================================================
# 第二步：火候设定 —— 超参数配置
# ==========================================================================
# 超参数如同炼丹的火候——火大则药焦（过拟合/发散），火小则药生（欠拟合）。
# 以下参数经过反复试炼，适合新手直接使用。

# --- 火候控制面板 ---
FLAME_INTENSITY = 1e-3    # 火焰强度（学习率 / Learning Rate）
                           # 控制每次调整斗气（参数）的幅度
                           # 推荐范围：1e-4 ~ 1e-2

HERB_BATCH_SIZE = 64       # 每批药材量（批大小 / Batch Size）
                           # 一次投入丹炉多少份药材
                           # 过大：显存不足（反噬）；过小：训练不稳

REFINING_CYCLES = 20       # 炼制轮次（训练轮数 / Epochs）
                           # 所有药材完整过一遍 = 一个炼制周期
                           # MNIST 通常 10-20 轮即可成丹

HIDDEN_QI_CHANNELS = [512, 256]  # 内部经脉宽度（隐藏层维度）
                                  # 经脉越宽，容纳的斗气越多，但消耗也越大

SEAL_PROBABILITY = 0.2     # 随机封脉概率（Dropout 比率）
                           # 训练时随机封闭部分经脉，防止过度依赖特定路径
                           # 这是对抗过拟合的重要手段

HERB_WORKERS = 0           # 药童数量（数据加载线程数）
                           # Windows 建议设为 0，Linux/Mac 可设为 2-4
                           # 设太多可能导致丹炉卡死

HERB_STORAGE = "./herb_storage"  # 药材存放位置


# ==========================================================================
# 第三步：药材采集与洗炼 —— 数据加载与预处理
# ==========================================================================
# 生药材（原始数据）不能直接入炉，需要经过洗炼（预处理）。
# 洗炼包括：转化为灵力形态（Tensor）、归一化灵力分布（Normalize）。

def prepare_herbs():
    """
    采集并洗炼药材（加载并预处理 MNIST 数据集）
    
    洗炼流程：
    1. ToTensor(): 将图片从 PIL 格式转为 Tensor，值域 [0,255] → [0,1]
    2. Normalize(): 将灵力分布标准化，使其均值接近0、方差接近1
       (0.1307, 0.3081) 是 MNIST 数据集的全局均值和标准差
       标准化后的数据更有利于梯度流通，加速炼丹
    
    Returns:
        train_loader: 训练药材流水线
        test_loader:  验证药材流水线
    """
    print("\n" + "=" * 50)
    print("【药材采集】正在采集 MNIST 药材...")
    print("=" * 50)
    
    # 药材洗炼配方
    herb_purification = transforms.Compose([
        transforms.ToTensor(),                    # 转化为灵力形态
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化灵力分布
    ])
    
    # 采集训练药材（6万份）
    train_herbs = torchvision.datasets.MNIST(
        root=HERB_STORAGE,
        train=True,
        download=True,
        transform=herb_purification
    )
    
    # 采集验证药材（1万份）
    test_herbs = torchvision.datasets.MNIST(
        root=HERB_STORAGE,
        train=False,
        download=True,
        transform=herb_purification
    )
    
    # 构建药材流水线
    # shuffle=True: 打乱药材顺序，防止模型学到顺序偏差
    # drop_last=True: 丢弃最后不满一批的药材，保持batch一致性
    train_loader = DataLoader(
        train_herbs,
        batch_size=HERB_BATCH_SIZE,
        shuffle=True,              # 训练时打乱！
        num_workers=HERB_WORKERS,
        drop_last=True,
        pin_memory=True            # 加速 CPU→GPU 的灵力传输
    )
    
    test_loader = DataLoader(
        test_herbs,
        batch_size=HERB_BATCH_SIZE,
        shuffle=False,             # 验证时不打乱
        num_workers=HERB_WORKERS,
        pin_memory=True
    )
    
    print(f"[药材采集] 训练药材：{len(train_herbs)} 份")
    print(f"[药材采集] 验证药材：{len(test_herbs)} 份")
    print(f"[药材采集] 每批药材量：{HERB_BATCH_SIZE}")
    print(f"[药材采集] 训练批次数：{len(train_loader)}")
    
    # 查看一份药材的形态
    sample, label = train_herbs[0]
    print(f"[药材形态] 单份药材形状：{sample.shape}")  # (1, 28, 28)
    print(f"[药材形态] 灵力值范围：[{sample.min():.3f}, {sample.max():.3f}]")
    print(f"[药材形态] 标签示例：数字 {label}")
    
    return train_loader, test_loader


# ==========================================================================
# 第四步：铸造丹方 —— 定义模型架构
# ==========================================================================
# 丹方（模型）定义了灵力的运转路线。
# 这是一个三层全连接网络（MLP），也叫多层感知机。
# 
# 灵力流转路径：
#   输入(1x28x28) → 展平(784) → 全连接(512) → ReLU → Dropout
#                               → 全连接(256) → ReLU → Dropout  
#                               → 全连接(10)  → 输出(10类概率)
#
# 每一层的作用：
# - Linear（全连接层）：对灵力施加线性变换 (y = Wx + b)
# - ReLU（灵力阀门）：截断负向灵力，引入非线性
# - Dropout（随机封脉）：随机关闭部分通道，防止过拟合

class QiVortexCondenser(nn.Module):
    """
    气旋凝聚器 —— 第一个炼丹模型
    
    这是一个多层感知机（MLP），也是最基础的神经网络架构。
    它将 28x28 的手写数字图片映射为 0-9 的分类概率。
    
    虽然简单，但它包含了深度学习的所有核心要素：
    - 可学习的参数（斗气 / weights）
    - 前向传播（灵力运转）
    - 非线性激活（灵力阀门）
    - 正则化（防过拟合手段）
    
    境界等级：黄阶低级（参数量 < 1M）
    """
    
    def __init__(self, input_dim=784, hidden_dims=None, output_dim=10, 
                 dropout_rate=0.2):
        """
        构建经脉体系（初始化网络层）
        
        Args:
            input_dim: 药材入口维度（28*28 = 784 个像素）
            hidden_dims: 内部经脉宽度列表，如 [512, 256]
            output_dim: 输出通道数（10个数字类别）
            dropout_rate: 随机封脉概率
        """
        super(QiVortexCondenser, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # ---- 构建经脉（网络层）----
        # 使用 nn.Sequential 将多个层串联，灵力会依次流过
        layers = []
        
        # 第一个组件：展平层
        # 将 (batch, 1, 28, 28) 的图片展平为 (batch, 784) 的向量
        # 就像把二维的药材碾成粉末，方便全连接层处理
        layers.append(nn.Flatten())
        
        # 构建隐藏层（内部经脉）
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # 全连接层：线性变换 y = Wx + b
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批归一化：稳定每层的灵力分布
            # 这能加速训练并允许使用更大的学习率
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU 激活：灵力阀门
            # f(x) = max(0, x)，只允许正向灵力通过
            # 没有非线性激活，再多的层也等于一层（线性变换的组合还是线性变换）
            layers.append(nn.ReLU())
            
            # Dropout：随机封脉术
            # 训练时随机将一部分输出置零，防止网络过度依赖某些特定路径
            # 测试时自动关闭（model.eval()）
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层：最终的灵力汇聚
        # 不加激活函数！因为后续的 CrossEntropyLoss 内部会执行 Softmax
        # 如果你这里加了 Softmax，等于做了两次 → 严重反噬
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 将所有层组装为一个整体
        self.qi_network = nn.Sequential(*layers)
        
        # 初始化权重（斗气初始分布）
        # 良好的初始化能加速收敛，避免梯度消失/爆炸
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        斗气初始化 —— 设定参数的初始分布
        
        使用 Kaiming 初始化（也叫 He 初始化），专为 ReLU 激活设计。
        核心思想：让每层的输出方差保持一致，避免灵力逐层衰减或爆炸。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming 正态分布初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_in', 
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # 偏置初始化为 0
    
    def forward(self, x):
        """
        灵力运转路线（前向传播）
        
        数据从输入端流入，依次经过每一层的变换，最终从输出端流出。
        
        Args:
            x: 输入药材，形状 (batch_size, 1, 28, 28)
        
        Returns:
            logits: 原始输出（未经 Softmax），形状 (batch_size, 10)
                    每个值代表对应数字的"置信度"
        """
        return self.qi_network(x)


# ==========================================================================
# 第五步：开炉炼丹 —— 训练循环
# ==========================================================================
# 炼丹的核心循环，日复一日的修炼：
# 1. 取出一批药材
# 2. 灵力运转（前向传播）得到预测
# 3. 度量偏离度（计算损失）
# 4. 灵力逆溯（反向传播）计算梯度
# 5. 调整经脉（优化器更新参数）
# 6. 清理残余灵力（梯度清零）
# 7. 重复

def train_one_cycle(model, train_loader, criterion, optimizer, device, cycle_num):
    """
    执行一个炼制周期（训练一个 epoch）
    
    一个周期 = 所有训练药材完整过一遍
    
    Args:
        model: 丹方（模型）
        train_loader: 训练药材流水线
        criterion: 偏差度量法（损失函数）
        optimizer: 灵力调节器（优化器）
        device: 丹炉（计算设备）
        cycle_num: 当前是第几个炼制周期
    
    Returns:
        avg_loss: 平均损失
        accuracy: 训练准确率
    """
    model.train()  # 切换到修炼模式
                   # 这会启用 Dropout 和 BatchNorm 的训练行为
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (herbs, true_labels) in enumerate(train_loader):
        # --- 将药材送入丹炉 ---
        herbs = herbs.to(device)
        true_labels = true_labels.to(device)
        
        # --- Step 1: 清理残余灵力（梯度清零）---
        # PyTorch 默认累加梯度，不清零会导致灵力紊乱
        optimizer.zero_grad()
        
        # --- Step 2: 灵力运转（前向传播）---
        # 数据流过整个网络，得到预测结果
        predictions = model(herbs)  # 形状: (batch_size, 10)
        
        # --- Step 3: 度量偏差（计算损失）---
        # CrossEntropyLoss 内部执行：Softmax + 负对数似然
        # 损失越小，说明模型的预测越接近真实标签（越接近天道）
        loss = criterion(predictions, true_labels)
        
        # --- Step 4: 灵力逆溯（反向传播）---
        # 从损失出发，沿计算图逆向传播，计算每个参数的梯度
        # 这就是链式法则的自动化实现
        loss.backward()
        
        # --- Step 5: 调整经脉（更新参数）---
        # 优化器根据梯度信息调整模型参数
        # Adam 优化器会自适应地为每个参数调整学习率
        optimizer.step()
        
        # --- 记录修炼数据 ---
        total_loss += loss.item() * herbs.size(0)
        _, predicted_labels = predictions.max(dim=1)  # 取概率最大的类别
        correct_predictions += predicted_labels.eq(true_labels).sum().item()
        total_samples += herbs.size(0)
        
        # 每 200 批打印一次修炼进度
        if (batch_idx + 1) % 200 == 0:
            batch_acc = correct_predictions / total_samples
            print(f"  [周期 {cycle_num:02d}] 批次 {batch_idx+1:4d}/{len(train_loader)} | "
                  f"损失: {loss.item():.4f} | 累计准确率: {batch_acc:.4f}")
    
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# ==========================================================================
# 第六步：验证成果 —— 测试评估
# ==========================================================================
# 每轮修炼后，用从未见过的药材（测试集）检验你的实力。
# 这才是真正的气旋强度——在未知挑战面前的表现。

@torch.no_grad()  # 装饰器：验证时不追踪梯度（节省灵力/显存）
def evaluate_qi_vortex(model, test_loader, criterion, device):
    """
    检验气旋强度（在测试集上评估模型）
    
    使用 @torch.no_grad() 装饰器，表示此函数内不需要梯度计算。
    这样做有两个好处：
    1. 节省显存（不用存储中间结果用于反向传播）
    2. 加速计算
    
    Args:
        model: 丹方（模型）
        test_loader: 验证药材流水线
        criterion: 偏差度量法
        device: 丹炉
    
    Returns:
        avg_loss: 平均损失
        accuracy: 测试准确率（气旋强度）
    """
    model.eval()  # 切换到验证模式
                  # Dropout 关闭（所有经脉全开）
                  # BatchNorm 使用全局统计量而非批次统计量
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 记录每个类别的正确率（详细的气旋分析）
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for herbs, true_labels in test_loader:
        herbs = herbs.to(device)
        true_labels = true_labels.to(device)
        
        # 灵力运转（前向传播）
        predictions = model(herbs)
        loss = criterion(predictions, true_labels)
        
        total_loss += loss.item() * herbs.size(0)
        _, predicted_labels = predictions.max(dim=1)
        correct_predictions += predicted_labels.eq(true_labels).sum().item()
        total_samples += herbs.size(0)
        
        # 统计每个数字的识别准确率
        for label, prediction in zip(true_labels, predicted_labels):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1
    
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy, class_correct, class_total


# ==========================================================================
# 第七步：主修炼流程 —— 串联一切
# ==========================================================================

def main():
    """
    主修炼流程 —— 从准备到成丹的完整过程
    """
    print("=" * 60)
    print("    《焚诀》第一卷 · 筑基篇")
    print("    ——— 气旋凝聚试炼 ———")
    print("=" * 60)
    print()
    print("  丹方：气旋凝聚器（MLP）")
    print("  药材：MNIST 手写数字（6万训练 + 1万验证）")
    print(f"  火候：学习率={FLAME_INTENSITY}, 批大小={HERB_BATCH_SIZE}")
    print(f"  轮次：{REFINING_CYCLES} 个炼制周期")
    print(f"  经脉：{HIDDEN_QI_CHANNELS}")
    print()
    
    # === 1. 检测丹炉 ===
    device = detect_furnace()
    print()
    
    # === 2. 采集药材 ===
    train_loader, test_loader = prepare_herbs()
    
    # === 3. 铸造丹方（创建模型）===
    print("\n" + "=" * 50)
    print("【铸造丹方】构建气旋凝聚器...")
    print("=" * 50)
    
    model = QiVortexCondenser(
        input_dim=784,
        hidden_dims=HIDDEN_QI_CHANNELS,
        output_dim=10,
        dropout_rate=SEAL_PROBABILITY
    )
    
    # 将模型送入丹炉
    model = model.to(device)
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[丹方信息] 模型架构：")
    print(model)
    print(f"\n[丹方信息] 斗气总量（总参数数）：{total_params:,}")
    print(f"[丹方信息] 可修炼参数：{trainable_params:,}")
    
    # === 4. 设定损失函数和优化器 ===
    # 交叉熵损失：度量预测概率分布与真实分布的偏差
    # 这是分类任务的标准损失函数
    criterion = nn.CrossEntropyLoss()
    
    # Adam 优化器：自适应学习率的灵力调节术
    # 它为每个参数维护独立的学习率，自动调节火候
    # 对于大多数任务，Adam 是最稳妥的选择
    optimizer = optim.Adam(model.parameters(), lr=FLAME_INTENSITY)
    
    # 学习率调度器：火候衰减策略
    # 每 7 个周期将学习率乘以 0.5
    # 修炼初期用猛火，后期用文火精炼
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    # === 5. 开炉炼丹！===
    print("\n" + "=" * 50)
    print("【开炉炼丹】修炼正式开始！")
    print("=" * 50)
    
    # 修炼记录
    training_chronicle = {
        "train_losses": [],
        "test_losses": [],
        "test_accuracies": [],
        "learning_rates": [],
    }
    
    best_accuracy = 0.0
    best_epoch = 0
    total_start_time = time.time()
    
    for cycle in range(1, REFINING_CYCLES + 1):
        cycle_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'─' * 50}")
        print(f"第 {cycle}/{REFINING_CYCLES} 个炼制周期 "
              f"| 当前火候(LR): {current_lr:.6f}")
        print(f"{'─' * 50}")
        
        # --- 修炼（训练）---
        train_loss, train_acc = train_one_cycle(
            model, train_loader, criterion, optimizer, device, cycle
        )
        
        # --- 验证 ---
        test_loss, test_acc, class_correct, class_total = evaluate_qi_vortex(
            model, test_loader, criterion, device
        )
        
        # --- 火候调整 ---
        scheduler.step()
        
        # --- 记录 ---
        cycle_time = time.time() - cycle_start_time
        training_chronicle["train_losses"].append(train_loss)
        training_chronicle["test_losses"].append(test_loss)
        training_chronicle["test_accuracies"].append(test_acc)
        training_chronicle["learning_rates"].append(current_lr)
        
        # --- 打印修炼成果 ---
        print(f"\n  [修炼报告]")
        print(f"  ├─ 训练损耗：{train_loss:.4f} | 训练准确率：{train_acc:.4f}")
        print(f"  ├─ 验证损耗：{test_loss:.4f} | 气旋强度 ：{test_acc:.4f}")
        print(f"  └─ 本轮耗时：{cycle_time:.1f}秒")
        
        # --- 记录最佳成绩 ---
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = cycle
            # 保存最佳模型（保存丹方副本）
            torch.save(model.state_dict(), "best_qi_vortex.pth")
            print(f"  ★ 新纪录！气旋强度提升至 {best_accuracy:.4f}，丹方已保存！")
    
    # === 6. 炼丹总结 ===
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("    炼 丹 总 结")
    print("=" * 60)
    print(f"  总耗时：{total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  最佳气旋强度：{best_accuracy:.4f} (第{best_epoch}轮)")
    print(f"  最终气旋强度：{training_chronicle['test_accuracies'][-1]:.4f}")
    
    # 分类详情 —— 每个数字的识别强度
    print(f"\n  各数字识别详情：")
    for digit in range(10):
        digit_acc = class_correct[digit] / class_total[digit] if class_total[digit] > 0 else 0
        bar = "█" * int(digit_acc * 30)
        print(f"    数字 {digit}: {digit_acc:.4f} ({class_correct[digit]}/{class_total[digit]}) {bar}")
    
    # === 7. 境界判定 ===
    final_acc = training_chronicle['test_accuracies'][-1]
    print(f"\n{'=' * 60}")
    print("    境 界 判 定")
    print(f"{'=' * 60}")
    
    if final_acc >= 0.98:
        print(f"  气旋强度：{final_acc:.2%}")
        print("  判定：★★★ 气旋凝实！天赋异禀！")
        print("  你的气旋已经非常稳固，可以直接进入第二卷·纳灵篇。")
    elif final_acc >= 0.97:
        print(f"  气旋强度：{final_acc:.2%}")
        print("  判定：★★☆ 气旋成型！根基扎实！")
        print("  恭喜！你已成功凝聚气旋，正式踏入修炼者行列。")
    elif final_acc >= 0.95:
        print(f"  气旋强度：{final_acc:.2%}")
        print("  判定：★☆☆ 气旋初现！仍需精进。")
        print("  建议：调整火候（学习率）或增加炼制轮次，再炼一次。")
    else:
        print(f"  气旋强度：{final_acc:.2%}")
        print("  判定：☆☆☆ 气旋未成。请检查丹方。")
        print("  建议：检查数据预处理、模型结构、学习率等超参数设置。")
    
    print(f"\n{'=' * 60}")
    print("  \"三十年河东，三十年河西，莫欺算法穷！\"")
    print(f"{'=' * 60}")
    
    # === 8. 保存修炼记录 ===
    print(f"\n[存档] 最佳丹方已保存至：best_qi_vortex.pth")
    print(f"[存档] 可使用以下代码加载：")
    print(f"  model = QiVortexCondenser()")
    print(f"  model.load_state_dict(torch.load('best_qi_vortex.pth'))")
    
    return model, training_chronicle


# ==========================================================================
# 入口：开始修炼！
# ==========================================================================
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║                                              ║")
    print("  ║       焚 诀 · 气 旋 凝 聚 术                  ║")
    print("  ║                                              ║")
    print("  ║   吞噬异火，进化功法                           ║")
    print("  ║   三十年河东，三十年河西，莫欺算法穷！           ║")
    print("  ║                                              ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    
    # 开始修炼
    trained_model, chronicle = main()
    
    print("\n--- 修炼结束 ---")
    print("下一步：阅读第二卷·纳灵篇，学习 CNN/RNN/Transformer")
    print("路径：../02-纳灵篇-斗者到斗师/README.md")
