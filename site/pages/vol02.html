# 第二卷：纳灵篇（斗者 - 斗师）— 炼气入体

> *"斗之气，三段！"*
> *——萧炎初次展现斗气外放之时*

> *"能将斗气凝于体表，只是斗者；能将斗气化为万千形态，方为斗师。"*
> *——《焚诀》第二卷 卷首语*

---

## 开篇引言：斗气外放，万象初生

修炼者，你已走过漫漫启程路。

在第一卷《吞噬篇》中，你完成了最关键的第一步——**凝聚气旋**。你学会了搭建炼丹炉（开发环境），
掌握了最基本的丹方（训练脚本），在 MNIST 这批入门级药材上成功炼出了第一炉丹药（训练出第一个模型）。
你的斗气不再是体内一团混沌的能量，而是有了雏形，有了方向。

但这仅仅是开始。

**斗者**之境，斗气只能在体内流转，勉强外放。就像你在第一卷中训练的全连接网络——它能工作，
但粗糙、笨拙，对输入数据的理解停留在最浅层。它把一张图片拍平成一维向量，丢失了所有空间信息，
就像一个武者闭着眼睛挥拳，虽有力道，却无章法。

要突破到**斗师**之境，你需要掌握三门大斗技（三大核心架构），让你的斗气化为不同形态，
应对不同的战斗场景：

| 斗技名称 | 技术对应 | 核心能力 | 比喻 |
|---------|---------|---------|------|
| **火眼金睛** | CNN（卷积神经网络） | 空间感知，视觉理解 | 看穿万物表象的灵瞳 |
| **时间长河** | RNN/LSTM（循环神经网络） | 时序感知，序列记忆 | 感知时间之流的神识 |
| **天道法则** | Transformer（自注意力架构） | 全局关联，并行感知 | 超越时空的天道感悟 |

这三门斗技，难度递增，威力递增。尤其是第三门——**天道法则（Transformer）**——它彻底改变了
修炼界的格局，是当今一切最强功法（GPT、BERT、LLaMA、Stable Diffusion 等）的根基。

本卷结束时，你将能够：

- 理解 CNN 如何"看"图像，并用它完成图像分类
- 理解 RNN/LSTM 如何处理序列数据，并用它生成文本
- **从数学原理到代码实现，彻底拆解 Transformer 架构**
- **亲手从零实现一个完整的 Transformer block**
- 追踪张量在每一层中的维度变化，做到心中有数

准备好了吗？让我们开始修炼。

---

## 第一章：火眼金睛 — 卷积神经网络 (CNN)

> *"修炼火眼金睛者，能从纷繁万象中捕捉到最细微的灵力波动，*
> *从一片落叶的纹理中推断出整座森林的气场。"*

### 1.1 为何需要新的斗技？

回忆第一卷中我们用全连接网络处理 MNIST 图片的方式：将 28x28 的图片**拍平**成 784 维向量，
然后喂入线性层。

这就像是你把一幅精美的画卷撕成碎片，搅成一堆，然后试图从这堆碎片中理解画的内容。
它丢失了一个至关重要的信息——**空间结构**。

图像中，相邻的像素往往高度相关。一只猫的眼睛、鼻子、耳朵是按特定空间关系排列的。
全连接网络完全无视这种关系，因为在它眼中，像素 (0,0) 和像素 (27,27) 之间没有任何
"距离"概念。

**卷积神经网络（CNN）** 正是为解决这个问题而生的斗技。它不撕碎画卷，而是用一双
**灵瞳**去**扫描**画卷，在扫描过程中捕捉局部模式。

### 1.2 卷积操作：灵瞳扫描术

想象你拥有一双修炼到极致的灵瞳。你不是一次性看完整幅画面，而是用一个小小的
**感知窗口**（卷积核 / kernel）在画面上滑动扫描。

```
输入图像（灵力场）          卷积核（灵瞳模式）         输出特征图（感知结果）
┌─────────────┐          ┌───────┐              ┌───────────┐
│ 1  0  1  0  │          │ 1  0  │              │ 2   1   │
│ 0  1  0  1  │    *     │ 0  1  │      =       │ 1   2   │
│ 1  0  1  0  │          └───────┘              │ 2   1   │
│ 0  1  0  1  │                                 └───────────┘
└─────────────┘
```

**具体过程**：

1. 灵瞳（kernel）放在图像的左上角
2. 将灵瞳覆盖区域的值与灵瞳自身的值逐元素相乘，然后求和
3. 得到的值填入输出特征图的对应位置
4. 灵瞳向右滑动一步（stride），重复计算
5. 到达右边界后，向下滑动一行，从左边重新开始

这个过程的数学表达：

```
Output[i, j] = Sum over (m, n) of Input[i+m, j+n] * Kernel[m, n]
```

**核心要义**：卷积核的参数是**共享**的——同一个灵瞳模式在整张图上滑动使用。
这意味着：

- **参数效率极高**：一个 3x3 的卷积核只有 9 个参数，却能扫描任意大小的图像
- **平移不变性**：无论猫出现在图像的左上角还是右下角，同一个卷积核都能检测到它

### 1.3 核心概念详解

#### 滤波器 / 卷积核（灵瞳修炼模式）

每个卷积核（filter）就是一种**特定的感知模式**。不同的卷积核检测不同的特征：

- 有的灵瞳善于感知**水平边缘**（横向灵力流动）
- 有的灵瞳善于感知**垂直边缘**（纵向灵力流动）
- 有的灵瞳善于感知**对角纹理**（斜向灵力漩涡）

一层卷积通常有**多个**卷积核（比如 32 个或 64 个），它们并行工作，各自检测不同的模式。
输出的通道数 = 卷积核的数量。

#### 步幅 Stride（灵瞳扫描速度）

步幅决定了灵瞳每次移动的距离：

- **stride=1**：每次移动 1 个像素，扫描最精细，输出特征图最大
- **stride=2**：每次移动 2 个像素，扫描速度翻倍，输出尺寸减半

步幅越大，感知越粗糙，但计算越快。就像修炼者快速扫视一个区域时，
能覆盖更大的范围，但可能遗漏细微的灵力波动。

#### 填充 Padding（外围感知域）

当灵瞳滑到图像边缘时，它的一部分会"悬空"在图像之外。填充就是在图像周围
补一圈零值（虚空灵力），使得：

- **Valid padding (无填充)**：输出尺寸 < 输入尺寸
- **Same padding**：输出尺寸 = 输入尺寸（最常用）

```
无填充：                      Same 填充（补零）：
┌─────────┐                 ┌─────────────┐
│ 图  像  │    →  更小输出    │ 0  0  0  0  │
│ 数  据  │                 │ 0  图  像  0  │  → 同尺寸输出
└─────────┘                 │ 0  数  据  0  │
                            │ 0  0  0  0  │
                            └─────────────┘
```

#### 池化 Pooling（精华提取术）

池化是一种**降维**操作，从一片区域中提取最关键的信息：

- **最大池化 (Max Pooling)**：取区域中的最大值——"只保留最强的灵力波动"
- **平均池化 (Average Pooling)**：取区域的平均值——"感知区域的整体灵力水平"

```
最大池化 (2x2, stride 2)：
┌─────┬─────┐          ┌─────┐
│ 1 3 │ 2 1 │          │  4  │   3
│ 4 2 │ 3 1 │    →     ├─────┤
├─────┼─────┤          │  6  │   4
│ 6 5 │ 1 4 │          └─────┘
│ 3 2 │ 2 3 │
└─────┴─────┘
```

池化的作用：
- **减小特征图尺寸**（减少计算量）
- **增加感受野**（让后续层能"看到"更大范围）
- **提供微小的平移不变性**

#### 输出尺寸计算公式

这个公式你必须烂熟于心，它是追踪张量维度变化的基础：

```
Output_size = (Input_size - Kernel_size + 2 * Padding) / Stride + 1
```

例如：输入 32x32，卷积核 3x3，padding=1，stride=1：
```
Output = (32 - 3 + 2*1) / 1 + 1 = 32
```
尺寸不变！这就是 same padding 的效果。

### 1.4 特征层级：从灵力涟漪到完整阵法

CNN 最深刻的洞见之一：**浅层检测简单模式，深层组合出复杂概念**。

```
第1层（浅层灵瞳）    第2-3层（中层感知）    第4-5层（深层领悟）     输出层（判断）
───────────────    ──────────────────    ─────────────────    ──────────
边缘 / 角点         纹理 / 形状组合         物体部件              "这是一只猫"
灵力涟漪            灵力旋涡               阵法片段              完整阵法辨识
```

这就像修炼火眼金睛的境界递进：
- **初窥**：能感知到最基本的灵力涟漪（边缘、色彩变化）
- **小成**：能辨识灵力旋涡的类型（纹理、简单形状）
- **大成**：能看出阵法的局部结构（物体的组成部分——眼睛、耳朵、轮子）
- **圆满**：能一眼看穿整个阵法的本质（这是猫、这是车、这是人）

### 1.5 经典架构演进

#### LeNet-5（1998）— 火眼金睛的创始功法

LeCun 大师创造的第一个实用 CNN,用于手写数字识别。结构简单但开创性：

```
Input(1x32x32) → Conv(6@5x5) → Pool → Conv(16@5x5) → Pool → FC(120) → FC(84) → Output(10)
```

#### AlexNet（2012）— 功法复兴

在 ImageNet 大赛上碾压传统方法，让整个修炼界重新重视 CNN：
- 更深的网络（8 层）
- 首次使用 ReLU 激活（灵力激发函数）
- Dropout 正则化（随机封闭经脉防止过拟合）
- GPU 训练（异火加速炼丹）

#### VGGNet（2014）— 大道至简

核心思想：**只用 3x3 的小卷积核，但堆叠很多层**。
两个 3x3 卷积等效于一个 5x5 卷积的感受野，但参数更少。

这就像修炼界的一个领悟：与其练一招威力巨大但难以控制的大招，
不如精通多招简洁的小技，组合使用反而更强。

#### ResNet（2015）— 残差连接，斗气循环捷径

**这是 CNN 历史上最重要的突破之一。**

当网络变得很深时（比如 50 层、100 层），会出现一个违反直觉的现象：
**更深的网络反而表现更差**。这不是过拟合，而是**优化困难**——梯度在
反向传播过程中逐层衰减，到达浅层时已近乎消失。

这就像斗气在漫长的经脉中流转，每经过一个穴位都会损耗一些，
经脉越长，到达末端的斗气越微弱。

**ResNet 的解决方案：残差连接（Skip Connection）**

```
          ┌──────────────────────────┐
          │      斗气捷径通道          │
          │   (Identity Shortcut)     │
          │                          ↓
输入 x ──→ ├──→ [卷积层] → [卷积层] → (+) ──→ 输出 = F(x) + x
```

简单到令人惊叹：**让输入直接跳过几层，与输出相加**。

这样网络只需要学习**残差** F(x) = H(x) - x，而不是完整的映射 H(x)。
如果某些层不需要做任何变换，F(x) 只需要学成零即可——这比学习恒等映射容易得多。

就像在漫长的经脉上开辟了一条**斗气捷径**，让能量可以直接跳过中间节点，
既保证了斗气不会在传递中完全消散，又让深层经脉也能得到充足的斗气滋养（梯度）。

### 1.6 火眼金睛的进阶修炼术

#### 1.6.1 批归一化（Batch Normalization）— 稳定灵力之法

**问题**：在深层网络训练中，每层的输入分布会不断变化，导致：
- 训练不稳定
- 必须使用很小的学习率
- 对初始化极其敏感

**批归一化（BatchNorm, BN）** 是 Sergey Ioffe 和 Christian Szegedy 在 2015 年提出的解决方案：

```
输入 x: (batch_size, C, H, W)
归一化：x_normalized = (x - mean(x)) / sqrt(var(x) + eps)
缩放：y = γ * x_normalized + β（γ 和 β 是可学习参数）
```

**核心思想**：对每个 batch 的特征进行标准化，使其均值为 0，方差为 1。

**修炼比喻**：就像在经脉中安装了"灵力稳定器"，确保流经每个穴位的斗气
都保持在合理的范围内，避免斗气过度波动导致经脉受损。

**BatchNorm 的完整实现**：

```python
class BatchNorm2d(nn.Module):
    """
    批归一化 —— 稳定深层网络修炼
    
    参数:
        num_features: 特征通道数
        eps: 数值稳定性常数
        momentum: 移动平均的动量
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 可学习的缩放和偏移参数（γ 和 β）
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 移动平均的均值和方差（用于推理）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = momentum
        
    def forward(self, x):
        """
        x: (batch_size, num_features, H, W)
        """
        if self.training:
            # 训练模式：使用当前 batch 的统计量
            # 计算每个通道的均值和方差
            mean = x.mean(dim=[0, 2, 3], keepdim=True)  # (1, C, 1, 1)
            var = x.var(dim=[0, 2, 3], keepdim=True)    # (1, C, 1, 1)
            
            # 更新移动平均
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # 推理模式：使用移动平均的统计量
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        y = self.gamma.view(1, -1, 1, 1) * x_normalized + self.beta.view(1, -1, 1, 1)
        
        return y
```

**BatchNorm 的独特优势**：
- **加速收敛**：可以使用更大的学习率
- **减少对初始化的依赖**：对参数初始化不敏感
- **轻微的正则化效果**：引入了噪声（batch 统计量），有类似 Dropout 的效果

**注意事项**：
- Batch size 太小（如 2、4）时，batch 统计量不稳定
- 在某些任务（如 GAN）中可能不适用
- 现代 GPU 推理时可能带来额外开销

#### 1.6.2 深度可分离卷积（Depthwise Separable Convolution）— 轻量化神术

**问题**：标准卷积的计算量很大，难以在移动端部署。

**深度可分离卷积**（MobileNet 的核心技术）将标准卷积分解为两步：

**标准卷积**：
```
输入: (B, C_in, H, W)
卷积核: (C_out, C_in, K_h, K_w)
输出: (B, C_out, H', W')
计算量: C_out × C_in × K_h × K_w × H' × W'
```

**深度可分离卷积**：
1. **Depthwise 卷积**（逐通道卷积）：
   - 每个输入通道独立应用一个卷积核
   - 卷积核: (C_in, 1, K_h, K_w)
   - 输出: (B, C_in, H', W')

2. **Pointwise 卷积**（1×1 卷积）：
   - 将深度卷积的输出进行通道混合
   - 卷积核: (C_out, C_in, 1, 1)
   - 输出: (B, C_out, H', W')

**计算量对比**：
```
标准卷积: C_out × C_in × K_h × K_w × H' × W'
深度可分离: C_in × K_h × K_w × H' × W' + C_out × C_in × H' × W'

节省比例 ≈ 1 / (C_out + K_h × K_w / C_in)
```

对于典型的 3×3 卷积和 C_out = C_in = 256：
```
标准卷积: 256 × 256 × 3 × 3 = 589,824
深度可分离: 256 × 3 × 3 + 256 × 256 = 71,680
节省: 88% ！
```

**完整实现**：

```python
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 —— 轻量化感知阵法
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步幅
        padding: 填充
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Depthwise 卷积：逐通道卷积
        # groups=in_channels 表示每个通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise 卷积：1×1 卷积进行通道混合
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: (B, C_in, H, W)
        输出: (B, C_out, H', W')
        """
        # 1. Depthwise 卷积
        x = self.depthwise(x)  # (B, C_in, H', W')
        x = self.bn1(x)
        x = self.relu(x)
        
        # 2. Pointwise 卷积
        x = self.pointwise(x)  # (B, C_out, H', W')
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
```

**修炼心得**：深度可分离卷积的核心哲学是**分解与重组**——将复杂操作分解为更简单的步骤，
在不损失太多性能的前提下大幅降低计算成本。这是资源受限环境下的智慧。

#### 1.6.3 Vision Transformer (ViT) — 火眼金睛的天道法则版

**问题**：CNN 的局部感受野虽然有效，但难以捕捉全局依赖。

**Vision Transformer (ViT)**（2020）将 Transformer 引入视觉领域：

```
图像 (H×W×3) → 切分 patches (N×P×P×3) → 展平 + 线性投影 → [CLS] token + 位置编码 → Transformer → [CLS] 表示 → 分类器
```

**核心步骤**：

1. **切分图像为 patches**：
   ```
   图像: 224×224×3
   Patch 大小: 16×16
   Patch 数量: (224/16) × (224/16) = 14×14 = 196
   每个 patch: 16×16×3 = 768 维
   ```

2. **线性投影**：
   ```
   将每个 patch (768维) 投影到 embedding 维度 (如 768)
   得到序列: (196, 768)
   ```

3. **添加 [CLS] token**：
   ```
   在序列前面添加一个可学习的 [CLS] token
   最终的 [CLS] 表示用于分类
   ```

4. **添加位置编码**：
   ```
   为每个 patch 添加位置信息（可学习或正弦编码）
   ```

5. **Transformer 编码器**：
   ```
   多层 Transformer block 处理序列
   ```

**ViT 的关键优势**：
- **全局感受野**：每个 patch 都能关注所有其他 patch
- **统一架构**：文本和图像可以使用相同的 Transformer
- **大规模预训练**：在 ImageNet-21K 等大规模数据上预训练后迁移学习效果很好

**ViT 的完整实现（简化版）**：

```python
class VisionTransformer(nn.Module):
    """
    Vision Transformer —— 将天道法则应用于视觉
    
    参数:
        image_size: 图像大小
        patch_size: patch 大小
        num_classes: 类别数
        d_model: embedding 维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        d_ff: FFN 中间维度
    """
    
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 d_model=768, num_heads=12, num_layers=12, d_ff=3072):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 计算 patch 数量
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding：将每个 patch 投影到 d_model 维
        self.patch_embed = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.mlp_head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        x: (B, 3, H, W)
        输出: (B, num_classes)
        """
        B = x.shape[0]
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # 2. 添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, d_model)
        
        # 3. 添加位置编码
        x = x + self.pos_embed
        
        # 4. Transformer 编码
        x = self.transformer(x)  # (B, num_patches+1, d_model)
        
        # 5. 使用 [CLS] token 进行分类
        cls_output = x[:, 0]  # (B, d_model)
        logits = self.mlp_head(cls_output)  # (B, num_classes)
        
        return logits
```

**修炼心得**：ViT 证明了 Transformer 的**通用性**——它不仅能处理文本，也能处理图像。
这是一种境界的飞跃：从"每种数据类型需要专门设计的架构"，到"统一的天道法则适用于万法"。

### 1.8 数据增强与正则化 — 锤炼斗气的秘术

#### 1.8.1 数据增强（Data Augmentation）

**问题**：数据量太少，模型容易过拟合。就像修炼者只在有限的几种药材上炼丹，遇到新药材就手忙脚乱。

**数据增强**：通过对现有数据进行各种变换，"创造"出更多的训练样本。

**常见图像数据增强方法**：

```python
import torch
import torchvision.transforms as T
from PIL import Image

class ImageAugmentation:
    """
    图像数据增强工具集
    
    策略：
    - 几何变换：翻转、旋转、缩放、裁剪
    - 颜色变换：亮度、对比度、饱和度、色相
    - 噪声注入：高斯噪声、椒盐噪声
    """
    
    def __init__(self, image_size=224, mode='train'):
        self.image_size = image_size
        self.mode = mode
        
        if mode == 'train':
            # 训练时使用更强的增强
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),  # 随机裁剪
                T.RandomHorizontalFlip(p=0.5),  # 水平翻转
                T.RandomRotation(degrees=15),  # 随机旋转
                T.ColorJitter(brightness=0.2, contrast=0.2,  # 颜色抖动
                            saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                          std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时不使用增强
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image):
        """
        image: PIL Image 或 Tensor
        """
        return self.transform(image)
```

**Cutout 和 Mixup — 高级增强技术**：

```python
import numpy as np

class Cutout:
    """
    Cutout: 随机遮挡图像的一部分
    
    核心：强制模型学习全局特征，而不是依赖局部特征
    
    参数:
        n_holes: 遮挡块的数量
        length: 遮挡块的边长
    """
    
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        """
        img: Tensor (C, H, W)
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            # 随机选择遮挡中心
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # 计算遮挡边界
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # 应用遮挡
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


def mixup_data(x, y, alpha=1.0):
    """
    Mixup: 混合两张图像和它们的标签
    
    公式：
        x_new = λ * x_i + (1 - λ) * x_j
        y_new = λ * y_i + (1 - λ) * y_j
        
    参数:
        x: (B, C, H, W) 输入图像
        y: (B,) 标签
        alpha: Beta 分布的参数
    
    返回:
        mixed_x: (B, C, H, W) 混合后的图像
        y_a, y_b: (B,) 原始标签（混合比例）
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    # 混合图像
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # 混合标签
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup 损失函数
    
    公式：
        loss = λ * loss(pred, y_a) + (1 - λ) * loss(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**数据增强的修炼心得**：
- **训练阶段**：使用强增强（翻转、旋转、裁剪、颜色抖动、Cutout、Mixup）
- **验证/测试阶段**：不使用增强，只做标准化和中心裁剪
- **增强强度**：需要通过验证集调优，过强的增强可能导致欠拟合

#### 1.8.2 正则化技术

**正则化（Regularization）**：防止过拟合的技术，就像修炼者修炼时需要有人监督，避免走火入魔。

**L1 和 L2 正则化**：

```python
def l1_regularization(model, lambda_l1=1e-5):
    """
    L1 正则化
    
    公式：
        Loss_total = Loss_data + λ * Σ|w|
    
    作用：
        - 鼓励稀疏权重（很多权重变为 0）
        - 特征选择（不重要的特征权重为 0）
    """
    l1_loss = 0.
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss


def l2_regularization(model, lambda_l2=1e-4):
    """
    L2 正则化（权重衰减）
    
    公式：
        Loss_total = Loss_data + λ * Σw²
    
    作用：
        - 防止权重过大
        - 更平滑的决策边界
        - 对异常值更鲁棒
    """
    l2_loss = 0.
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return lambda_l2 * l2_loss


def train_with_regularization(model, dataloader, optimizer, criterion, 
                             lambda_l1=1e-5, lambda_l2=1e-4, device='cuda'):
    """
    带 L1 和 L2 正则化的训练
    
    注意：PyTorch 的 optimizer 自带 L2 正则化（weight_decay 参数），
    但 L1 正则化需要手动添加。
    """
    model.train()
    total_loss = 0.
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 数据损失
        loss = criterion(outputs, targets)
        
        # 添加 L1 正则化
        loss += l1_regularization(model, lambda_l1)
        
        # L2 正则化通常通过 optimizer 的 weight_decay 实现
        # 这里不手动添加
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**Dropout — 随机失活**：

```python
class MLPWithDropout(nn.Module):
    """
    带 Dropout 的全连接网络
    
    Dropout 训练时：随机将一部分神经元的输出置为 0
    Dropout 推理时：所有神经元都工作，输出乘以 (1 - p)
    
    参数 p: 失活概率（通常 0.2 ~ 0.5）
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # 训练时随机失活，推理时自动处理
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

**Dropout 的修炼心得**：
- **训练时**：每个 batch 随机失活不同的神经元，相当于训练了多个"子网络"
- **推理时**：相当于多个子网络的集成（平均），提高泛化能力
- **失活率**：通常 0.2 ~ 0.5，过大会导致欠拟合
- **位置**：通常放在全连接层后，CNN 中较少使用（有 BatchNorm）

**Early Stopping — 早停**：

```python
class EarlyStopping:
    """
    早停：当验证损失不再下降时停止训练
    
    参数:
        patience: 容忍验证损失不下降的 epoch 数
        min_delta: 最小改善量（小于这个值视为没改善）
        verbose: 是否打印信息
        mode: 'min'（损失）或 'max'（准确率）
    """
    
    def __init__(self, patience=10, min_delta=0, verbose=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        
        self.counter = 0  # 计数器
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0
    
    def save_checkpoint(self, val_metric, model):
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation metric improved. Saving model...')
        self.best_model_state = model.state_dict().copy()
    
    def load_best_model(self, model):
        """加载最佳模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            return model
        return model
```

### 1.9 目标检测与分割 — 火眼金睛的终极形态

#### 1.9.1 目标检测（Object Detection）

#### 1.7.1 目标检测（Object Detection）

目标检测不仅要识别图像中有什么，还要定位它们的位置。

**Two-Stage 方法**（如 Faster R-CNN）：
1. **Region Proposal Network (RPN)**：生成候选区域
2. **ROI Pooling + 分类**：对每个区域分类和边界框回归

**One-Stage 方法**（如 YOLO、SSD）：
- 直接在特征图上预测边界框和类别
- 速度更快，适合实时应用

**YOLO（You Only Look Once）核心思想**：

```python
class YOLO(nn.Module):
    """
    YOLO v1 简化版 —— 一次前向同时检测所有目标
    
    参数:
        grid_size: 网格大小（如 7×7）
        num_boxes: 每个网格预测的边界框数量
        num_classes: 类别数
    """
    
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # 每个网格单元预测：num_boxes × (5 + num_classes)
        # 5 = (x, y, w, h, confidence)
        self.predictor = nn.Sequential(
            nn.Linear(1024, 4960),  # 7×7×2×(5+20) = 4960
        )
    
    def forward(self, x):
        """
        x: (B, 3, 448, 448)
        输出: (B, 7, 7, 2, 25)  # 7×7 网格，每个位置 2 个框，每个框 25 维
        """
        x = self.predictor(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_boxes, 5 + self.num_classes)
        return x
```

#### 1.7.2 语义分割（Semantic Segmentation）

语义分割为图像的每个像素分配类别标签。

**U-Net（医学图像分割经典）**：

```python
class UNet(nn.Module):
    """
    U-Net —— 编码器-解码器架构
    
    特点：
    - 对称的 U 形结构
    - 编码器：收缩路径，提取特征
    - 解码器：扩张路径，恢复空间信息
    - 跳跃连接（Skip Connections）：连接编码器和解码器
    """
    
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        
        # 编码器
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._double_conv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._double_conv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._double_conv(128, 64)
        
        # 输出层
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        x1 = self.enc1(x)
        x2 = self.pool(self.enc2(x1))
        x3 = self.pool(self.enc3(x2))
        x4 = self.pool(self.enc4(x3))
        
        # 解码器（带跳跃连接）
        x = self.up4(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec2(x)
        
        # 输出
        x = self.final(x)
        return x
```

### 1.6 实战：用 CNN 修炼 CIFAR-10 分类

CIFAR-10 是比 MNIST 更具挑战性的药材（数据集）：10 类彩色图像，每张 32x32x3。

```python
"""
=============================================================
  《焚诀》第二卷 - 火眼金睛修炼实录
  丹方：CNN 分类 CIFAR-10
  药材：CIFAR-10（十类彩色灵图）
  异火：CUDA（如果可用）
=============================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ============ 第一步：炮制药材（数据预处理） ============
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),    # 随机翻转——增加药材多样性
    transforms.RandomCrop(32, padding=4), # 随机裁剪——扩展灵图视角
    transforms.ToTensor(),
    transforms.Normalize(                 # 标准化——平衡灵力分布
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# 加载药材
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

classes = ('飞机', '汽车', '鸟', '猫', '鹿',
           '狗', '蛙', '马', '船', '卡车')


# ============ 第二步：构建斗技（定义模型） ============
class FireEyeCNN(nn.Module):
    """
    火眼金睛 CNN —— 灵瞳感知阵法

    架构设计思路：
    - 3 个卷积块，每块包含 2 个卷积层 + 批归一化 + 池化
    - 通道数逐步增加：32 → 64 → 128（感知逐层深化）
    - 使用残差连接（斗气捷径）增强梯度流动
    - 最后用全局平均池化 + 全连接层输出
    """

    def __init__(self, num_classes=10):
        super(FireEyeCNN, self).__init__()

        # 第一层灵瞳：初窥——检测基础灵力涟漪（边缘、色彩）
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3通道→32通道
            nn.BatchNorm2d(32),                            # 稳定灵力流
            nn.ReLU(inplace=True),                         # 灵力激发
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 精华提取：32→16
            nn.Dropout2d(0.2),                             # 随机封脉防过拟合
        )

        # 第二层灵瞳：小成——辨识灵力旋涡类型（纹理、形状）
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32通道→64通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 精华提取：16→8
            nn.Dropout2d(0.3),
        )

        # 第三层灵瞳：大成——看穿阵法本质（物体整体特征）
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64通道→128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 精华提取：8→4
            nn.Dropout2d(0.4),
        )

        # 全局灵力汇聚 + 判断输出
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化→(128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x 的维度变化追踪：
        # 输入: (batch, 3, 32, 32) —— 一批彩色灵图
        x = self.block1(x)   # → (batch, 32, 16, 16) —— 初窥：基础特征
        x = self.block2(x)   # → (batch, 64, 8, 8)   —— 小成：中级特征
        x = self.block3(x)   # → (batch, 128, 4, 4)  —— 大成：高级特征
        x = self.global_pool(x)  # → (batch, 128, 1, 1) —— 灵力汇聚
        x = self.classifier(x)   # → (batch, 10)       —— 阵法判断
        return x


# ============ 第三步：开炉炼丹（训练） ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的异火：{device}")

model = FireEyeCNN().to(device)
criterion = nn.CrossEntropyLoss()           # 天罚函数（损失函数）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 自适应斗气调节
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()           # 清除残余斗气
        outputs = model(inputs)         # 灵瞳感知
        loss = criterion(outputs, targets)  # 天罚降临
        loss.backward()                 # 反向追溯因果
        optimizer.step()                # 调整经脉参数

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    train_acc = 100. * correct / total

    # 每 10 轮检验修炼成果
    if (epoch + 1) % 10 == 0:
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"第 {epoch+1} 轮 | "
              f"炼丹损耗: {running_loss/len(trainloader):.3f} | "
              f"修炼准确率: {train_acc:.2f}% | "
              f"试炼准确率: {test_acc:.2f}%")

print("火眼金睛修炼完成！")
```

> **修炼笔记**：这个简单的 CNN 在 CIFAR-10 上应能达到 ~88-90% 的准确率。
> 如果使用更先进的架构（如 ResNet-18），可以轻松突破 93%。

---

## 附录 A：斗气修炼心法 — 优化算法与损失函数

> * "炼丹之法，在于火候。火太猛则药力流失，火太弱则药效不显。*
> * 修炼亦是如此，优化器就是控制火候的心法，损失函数就是衡量药力的标尺。"*
> * —— 《焚诀》附录篇*

### A.1 优化算法详解

优化算法（Optimizer）决定了模型如何更新参数，就像修炼者如何吸收和运用天地灵气。

#### A.1.1 梯度下降（Gradient Descent）— 最基础的心法

**核心思想**：沿着梯度的反方向移动参数，使损失函数最小化。

```python
def gradient_descent_step(model, learning_rate):
    """
    手动实现梯度下降
    
    公式：
        θ_new = θ_old - lr * ∇L(θ)
    
    参数:
        model: PyTorch 模型
        learning_rate: 学习率（步长）
    """
    with torch.no_grad():  # 不记录梯度
        for param in model.parameters():
            param -= learning_rate * param.grad  # 梯度反方向更新
```

**梯度下降的变体**：

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **批量梯度下降（Batch GD）** | 使用全部数据计算梯度 | 数据集小，稳定但慢 |
| **随机梯度下降（SGD）** | 每次使用一个样本 | 数据集大，快但波动大 |
| **小批量梯度下降（Mini-batch GD）** | 每次使用一个小 batch | **最常用**，平衡速度和稳定性 |

**SGD 实现**：

```python
def sgd_step(model, learning_rate, batch_size):
    """
    随机梯度下降（SGD）
    
    PyTorch 的 SGD 优化器使用的是 Mini-batch SGD
    """
    with torch.no_grad():
        for param in model.parameters():
            param.grad = param.grad / batch_size  # 梯度平均
            param -= learning_rate * param.grad
```

#### A.1.2 动量（Momentum）— 惯性助推

**问题**：SGD 在平坦区域下降慢，在峡谷区域震荡，容易陷入局部最小值。

**动量（Momentum）**：引入"惯性"，让更新方向更稳定。

```python
class MomentumOptimizer:
    """
    动量优化器
    
    公式：
        v_t = γ * v_{t-1} + lr * ∇L(θ)
        θ_new = θ_old - v_t
    
    参数:
        γ: 动量系数（通常 0.9）
        v: 速度向量（累积的梯度历史）
    """
    
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in model.parameters()]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                # 更新速度
                self.velocities[i] = (self.momentum * self.velocities[i] + 
                                    self.lr * param.grad)
                # 更新参数
                param -= self.velocities[i]
```

**动量的修炼心得**：
- 就像推球下山，一开始速度慢，但随着惯性积累，速度越来越快
- 能更快通过平坦区域（速度累积）
- 能减少峡谷震荡（惯性保持方向）

#### A.1.3 AdaGrad — 自适应学习率

**问题**：不同参数需要不同的学习率。

**AdaGrad**：为每个参数自适应调整学习率，频繁更新的参数学习率减小，不频繁更新的参数学习率增大。

```python
class AdaGradOptimizer:
    """
    AdaGrad 优化器
    
    公式：
        G_t = G_{t-1} + (∇L(θ))²
        θ_new = θ_old - (lr / sqrt(G_t + ε)) * ∇L(θ)
    
    参数:
        G: 累积的梯度平方和
        ε: 数值稳定性常数
    """
    
    def __init__(self, model, learning_rate=0.01, epsilon=1e-8):
        self.model = model
        self.lr = learning_rate
        self.epsilon = epsilon
        self.G = [torch.zeros_like(p) for p in model.parameters()]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                # 累积梯度平方
                self.G[i] += param.grad ** 2
                # 自适应学习率更新
                adaptive_lr = self.lr / torch.sqrt(self.G[i] + self.epsilon)
                param -= adaptive_lr * param.grad
```

**AdaGrad 的问题**：学习率持续递减，后期可能过小。

#### A.1.4 RMSprop — AdaGrad 的改进

**RMSprop**：使用指数移动平均代替累积和，避免学习率过小。

```python
class RMSpropOptimizer:
    """
    RMSprop 优化器
    
    公式：
        E[g²]_t = β * E[g²]_{t-1} + (1 - β) * (∇L(θ))²
        θ_new = θ_old - (lr / sqrt(E[g²]_t + ε)) * ∇L(θ)
    
    参数:
        β: 衰减率（通常 0.9）
        E[g²]: 梯度平方的指数移动平均
    """
    
    def __init__(self, model, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.model = model
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.E_g2 = [torch.zeros_like(p) for p in model.parameters()]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                # 指数移动平均
                self.E_g2[i] = (self.beta * self.E_g2[i] + 
                              (1 - self.beta) * param.grad ** 2)
                # 自适应学习率更新
                adaptive_lr = self.lr / torch.sqrt(self.E_g2[i] + self.epsilon)
                param -= adaptive_lr * param.grad
```

#### A.1.5 Adam — 自适应矩估计（最常用）

**Adam（Adaptive Moment Estimation）**：结合动量和 RMSprop 的优点，目前最流行的优化器。

```python
class AdamOptimizer:
    """
    Adam 优化器
    
    公式：
        m_t = β1 * m_{t-1} + (1 - β1) * g_t  # 一阶矩（梯度均值）
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²  # 二阶矩（梯度方差）
        m_t_hat = m_t / (1 - β1^t)  # 偏差修正
        v_t_hat = v_t / (1 - β2^t)
        θ_new = θ_old - lr * m_t_hat / (sqrt(v_t_hat) + ε)
    
    默认超参数：
        lr = 0.001
        β1 = 0.9（一阶矩衰减率）
        β2 = 0.999（二阶矩衰减率）
        ε = 1e-8
    """
    
    def __init__(self, model, learning_rate=0.001, beta1=0.9, 
                 beta2=0.999, epsilon=1e-8):
        self.model = model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = [torch.zeros_like(p) for p in model.parameters()]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in model.parameters()]  # 二阶矩
        self.t = 0  # 时间步
    
    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                # 更新一阶矩和二阶矩
                self.m[i] = (self.beta1 * self.m[i] + 
                           (1 - self.beta1) * param.grad)
                self.v[i] = (self.beta2 * self.v[i] + 
                           (1 - self.beta2) * param.grad ** 2)
                
                # 偏差修正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
```

**PyTorch 中的 Adam 使用**：

```python
import torch.optim as optim

# 创建 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练循环
for inputs, targets in dataloader:
    optimizer.zero_grad()  # 清零梯度
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
```

**Adam 的修炼心得**：
- **自适应**：每个参数有自己的学习率
- **动量**：有惯性，能加速收敛
- **鲁棒**：对初始学习率不敏感
- **默认选择**：大多数情况下使用 Adam 效果都不错

#### A.1.6 AdamW — Adam + 权重衰减

**AdamW**：将权重衰减与梯度解耦，是 Adam 的改进版本。

```python
# PyTorch 中使用 AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Adam vs AdamW 对比**：

| 特性 | Adam | AdamW |
|------|------|-------|
| 权重衰减方式 | 与梯度耦合 | 与梯度解耦 |
| 泛化能力 | 较好 | **更好** |
| 推荐场景 | 通用 | **大模型、预训练** |

### A.2 损失函数详解

损失函数（Loss Function）衡量模型预测与真实值的差距，就像衡量丹药药效的标尺。

#### A.2.1 回归问题损失

**均方误差（Mean Squared Error, MSE）**：

```python
def mse_loss(pred, target):
    """
    均方误差
    
    公式：
        MSE = (1/n) * Σ(y_pred - y_true)²
    
    特点：
        - 对大误差敏感（平方放大）
        - 可导
        - 假设误差服从高斯分布
    """
    return torch.mean((pred - target) ** 2)

# PyTorch 内置
loss = nn.MSELoss()(pred, target)
```

**平均绝对误差（Mean Absolute Error, MAE）**：

```python
def mae_loss(pred, target):
    """
    平均绝对误差
    
    公式：
        MAE = (1/n) * Σ|y_pred - y_true|
    
    特点：
        - 对异常值不敏感
        - 在导数不连续点（0）次优
    """
    return torch.mean(torch.abs(pred - target))

# PyTorch 内置
loss = nn.L1Loss()(pred, target)
```

**Huber Loss**：

```python
def huber_loss(pred, target, delta=1.0):
    """
    Huber Loss：结合 MSE 和 MAE 的优点
    
    公式：
        if |y_pred - y_true| <= delta:
            loss = 0.5 * (y_pred - y_true)²
        else:
            loss = delta * |y_pred - y_true| - 0.5 * delta²
    
    特点：
        - 误差小时用 MSE（平滑）
        - 误差大时用 MAE（对异常值不敏感）
    """
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(loss)
```

#### A.2.2 分类问题损失

**交叉熵损失（Cross-Entropy Loss）**：

```python
def cross_entropy_loss(pred, target):
    """
    交叉熵损失
    
    公式：
        CE = -Σ y_true * log(y_pred)
    
    特点：
        - 概率分布差异的衡量
        - 对预测错误惩罚大
    """
    # pred: (B, C) 未经 softmax 的 logits
    # target: (B,) 类别索引
    return nn.CrossEntropyLoss()(pred, target)

# 或者手动实现
def manual_cross_entropy_loss(pred, target):
    pred_softmax = F.softmax(pred, dim=-1)
    log_pred = torch.log(pred_softmax + 1e-10)  # 防止 log(0)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[-1]).float()
    loss = -torch.sum(target_one_hot * log_pred, dim=-1).mean()
    return loss
```

**二元交叉熵损失（Binary Cross-Entropy）**：

```python
def binary_cross_entropy_loss(pred, target):
    """
    二元交叉熵损失
    
    适用：
        - 二分类问题
        - 多标签分类（每个样本可以有多个标签）
    """
    # pred: (B,) 经过 sigmoid 的概率
    # target: (B,) 0 或 1
    return nn.BCELoss()(pred, target)

# 或者使用 logits 版本
def binary_cross_entropy_with_logits_loss(pred, target):
    # pred: (B,) 未经 sigmoid 的 logits
    return nn.BCEWithLogitsLoss()(pred, target)
```

**Focal Loss**：

```python
class FocalLoss(nn.Module):
    """
    Focal Loss：解决类别不平衡问题
    
    公式：
        FL = -α * (1 - p_t)^γ * log(p_t)
    
    参数:
        alpha: 类别权重
        gamma: 聚焦参数（通常 2.0）
    
    核心思想：
        - 容易分类的样本降低权重（(1 - p_t)^γ 小）
        - 困难分类的样本提高权重（(1 - p_t)^γ 大）
    """
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: (B, C) logits
        # target: (B,) 类别索引
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

#### A.2.3 对抗损失（Adversarial Loss）

**GAN 的损失函数**：

```python
def gan_loss(logits_real, logits_fake, mode='discriminator'):
    """
    GAN 损失
    
    判别器目标：
        - 真实样本判为 1
        - 生成样本判为 0
    
    生成器目标：
        - 生成样本判为 1
    """
    if mode == 'discriminator':
        # 判别器损失
        loss_real = F.binary_cross_entropy_with_logits(
            logits_real, torch.ones_like(logits_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            logits_fake, torch.zeros_like(logits_fake)
        )
        return (loss_real + loss_fake) / 2
    else:
        # 生成器损失
        loss = F.binary_cross_entropy_with_logits(
            logits_fake, torch.ones_like(logits_fake)
        )
        return loss
```

### A.3 高级训练技巧 — 突破瓶颈的秘术

> *"基础功法人人可学，但要在斗师之境更进一步，必须掌握那些不传之秘。*
> *这些技巧，或许能让你在关键时刻突破瓶颈，达到更高的境界。"*

#### A.3.1 学习率调度 — 温火慢炖的艺术

**学习率 warmup（预热）**：

```python
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR

class WarmupScheduler(LambdaLR):
    """
    Warmup 调度器
    
    原理：
        - 前期学习率从小到大，避免模型初期震荡
        - 后期按正常策略衰减
        
    公式：
        lr = base_lr * min(step / warmup_steps, 1.0)
    """
    
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        super().__init__(optimizer, lr_lambda)

# 使用示例
model = nn.Linear(100, 10)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=0.001)

# 训练循环
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        warmup_scheduler.step()  # 每个 step 更新学习率
```

**Cosine 退火调度**：

```python
def cosine_annealing_with_warmup(optimizer, warmup_steps, max_steps, min_lr=0):
    """
    Cosine 退火 + Warmup
    
    公式：
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
        其中 progress = (step - warmup_steps) / (max_steps - warmup_steps)
    
    优点：
        - 平滑的学习率变化
        - 后期逐渐降低，有助于收敛到更优解
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    
    return LambdaLR(optimizer, lr_lambda)

# 完整示例
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = cosine_annealing_with_warmup(
    optimizer, 
    warmup_steps=100, 
    max_steps=1000, 
    min_lr=0.0001
)
```

#### A.3.2 梯度裁剪 — 避免走火入魔

**梯度爆炸的识别与处理**：

```python
def check_gradient_explosion(model, threshold=100.0):
    """
    检查梯度爆炸
    
    指标：
        - 最大梯度范数
        - 平均梯度范数
        - 超过阈值的参数数量
    """
    max_norm = 0.0
    avg_norm = 0.0
    param_count = 0
    explosion_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            max_norm = max(max_norm, param_norm)
            avg_norm += param_norm
            param_count += 1
            
            if param_norm > threshold:
                explosion_count += 1
                print(f"⚠️  梯度爆炸: {name}, norm={param_norm:.2f}")
    
    if param_count > 0:
        avg_norm /= param_count
    
    return {
        'max_norm': max_norm,
        'avg_norm': avg_norm,
        'explosion_count': explosion_count
    }

# 梯度裁剪的几种方式

# 1. 按范数裁剪（推荐）
def clip_grad_norm_(model, max_norm):
    """
    按范数裁剪梯度
    
    公式：
        if total_norm > max_norm:
            grad = grad * (max_norm / total_norm)
    
    优点：
        - 保持梯度方向不变
        - 只是缩放梯度大小
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# 2. 按值裁剪
def clip_grad_value_(model, clip_value):
    """
    按值裁剪梯度
    
    将每个梯度元素限制在 [-clip_value, clip_value] 范围内
    """
    return torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

# 使用示例
model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 较大的学习率容易爆炸

for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 训练前检查
        grad_stats = check_gradient_explosion(model, threshold=10.0)
        print(f"梯度统计: max={grad_stats['max_norm']:.2f}, avg={grad_stats['avg_norm']:.2f}")
        
        # 梯度裁剪
        clip_grad_norm_(model, max_norm=1.0)
        
        optimizer.step()
```

**自适应梯度裁剪（AGC）**：

```python
class AdaptiveGradientClipping:
    """
    自适应梯度裁剪
    
    核心思想：
        - 每个参数的裁剪阈值 = λ * ||W||
        - 大权重的参数允许更大的梯度
    
    公式：
        g_clip = g * min(1, λ * ||W|| / ||g||)
    
    论文：
        AGC: Adaptive Gradient Clipping (https://arxiv.org/abs/2102.06171)
    """
    
    def __init__(self, model, clip_factor=0.01, eps=1e-3):
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps
    
    def step(self):
        """
        执行自适应梯度裁剪
        """
        for param in self.model.parameters():
            if param.grad is None:
                continue
            
            # 计算权重范数
            weight_norm = torch.norm(param.data, p=2)
            # 计算梯度范数
            grad_norm = torch.norm(param.grad.data, p=2)
            
            # 计算裁剪阈值
            clip_factor = torch.clamp(
                self.clip_factor * weight_norm / (grad_norm + self.eps),
                max=1.0
            )
            
            # 裁剪梯度
            param.grad.data.mul_(clip_factor)

# 使用示例
model = nn.Linear(1000, 1000)
optimizer = optim.Adam(model.parameters(), lr=0.01)
agc = AdaptiveGradientClipping(model, clip_factor=0.01)

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # 自适应梯度裁剪
    agc.step()
    
    optimizer.step()
```

#### A.3.3 混合精度训练 — 节省灵力的法门

**FP16 训练的优势**：

| 优势 | 说明 |
|------|------|
| **显存占用减半** | FP16: 2 bytes, FP32: 4 bytes |
| **计算速度提升** | Tensor Core 专门优化 FP16 运算 |
| **更大的 batch size** | 显存节省，可以增加 batch |

**原生 AMP（Automatic Mixed Precision）训练**：

```python
from torch.cuda.amp import autocast, GradScaler

def train_amp(model, train_loader, optimizer, criterion, epochs=5):
    """
    AMP 训练流程
    
    核心组件：
        1. autocast: 自动选择 FP16/FP32
        2. GradScaler: 梯度缩放，防止下溢出
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 梯度缩放器
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # autocast 上下文：自动转换为 FP16
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # 梯度缩放 + 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪（可选，需要在 unscale 之前）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            
            # 更新缩放因子
            scaler.update()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 使用示例
model = nn.Sequential(
    nn.Linear(784, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_amp(model, train_loader, optimizer, criterion, epochs=10)
```

**梯度缩放的原理**：

```python
class ManualGradScaler:
    """
    手动梯度缩放器（理解原理）
    
    问题：
        FP16 的范围 [-65504, 65504] 远小于 FP32
        小梯度可能下溢出为 0
        
    解决：
        将梯度乘以一个大的缩放因子（如 2^16）
        反向传播后再除回去
        
    公式：
        scaled_loss = loss * scale_factor
        scaled_grad = backward(scaled_loss)
        grad = scaled_grad / scale_factor
    """
    
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self._growth_tracker = 0
    
    def scale(self, loss):
        """缩放损失"""
        return loss * self.scale
    
    def step(self, optimizer):
        """
        更新参数 + 调整缩放因子
        
        策略：
            - 如果没有 inf/nan，逐步增大缩放因子
            - 如果出现 inf/nan，缩小缩放因子
        """
        # 检查是否有 inf/nan
        has_inf_nan = False
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                has_inf_nan = torch.isinf(param.grad).any() or torch.isnan(param.grad).any()
                if has_inf_nan:
                    break
        
        if has_inf_nan:
            # 缩小缩放因子
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            print(f"⚠️  检测到 inf/nan，缩放因子调整为: {self.scale:.2f}")
        else:
            # 正常更新参数
            optimizer.step()
            
            # 逐步增大缩放因子
            self._growth_tracker += 1
            if self._growth_tracker >= 2000:  # 每 2000 steps 增大一次
                self.scale *= self.growth_factor
                self._growth_tracker = 0
```

#### A.3.4 标签平滑 — 避免过度自信

**为什么需要标签平滑**：

```python
def label_smoothing_loss(pred, target, smoothing=0.1):
    """
    标签平滑损失
    
    问题：
        - 传统 one-hot 标签让模型过度自信
        - 模型可能记住训练集噪声
        - 泛化性能下降
    
    解决：
        - one-hot: [0, 0, 1, 0, 0]
        - 平滑后: [0.025, 0.025, 0.9, 0.025, 0.025]
    
    公式：
        y_smooth = (1 - ε) * y_one_hot + ε / K
        其中 ε 是平滑系数，K 是类别数
    """
    num_classes = pred.shape[-1]
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    # 创建平滑标签
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()
    target_smooth = target_one_hot * confidence + smooth_value * (1 - target_one_hot)
    
    # 计算交叉熵
    log_probs = F.log_softmax(pred, dim=-1)
    loss = -torch.sum(target_smooth * log_probs, dim=-1).mean()
    
    return loss

# 使用示例
model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    
    # 使用标签平滑
    loss = label_smoothing_loss(output, target, smoothing=0.1)
    
    loss.backward()
    optimizer.step()
```

**PyTorch 内置标签平滑**：

```python
# PyTorch 1.10+ 内置支持
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)  # 自动应用标签平滑
    loss.backward()
    optimizer.step()
```

#### A.3.5 梯度累积 — 突破显存限制

**为什么需要梯度累积**：

```python
def train_with_gradient_accumulation(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    accumulation_steps=4, 
    epochs=5
):
    """
    梯度累积训练
    
    问题：
        - 小显存只能跑 batch_size=4
        - 但大 batch size（如 16）收敛更快
    
    解决：
        - 分 4 个 mini-batch，每个 batch_size=4
        - 累积 4 次梯度后，再更新参数
        - 相当于 batch_size=16 的效果
    
    公式：
        loss_accumulated = (loss_1 + loss_2 + ... + loss_N) / N
        grad_accumulated = (grad_1 + grad_2 + ... + grad_N) / N
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 重要：除以 accumulation_steps
            loss = loss / accumulation_steps
            
            # 反向传播（梯度累积）
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # 每个 accumulation_steps 更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪（可选）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
        
        # 处理剩余的 batch
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 使用示例
model = nn.Linear(784, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设显存只能跑 batch_size=32，但想要 batch_size=128 的效果
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

train_with_gradient_accumulation(
    model, 
    train_loader, 
    optimizer, 
    criterion=nn.CrossEntropyLoss(),
    accumulation_steps=4,  # 32 * 4 = 128
    epochs=10
)
```

**梯度累积 + AMP 组合**：

```python
def train_amp_with_accumulation(
    model, 
    train_loader, 
    optimizer, 
    criterion, 
    accumulation_steps=4,
    epochs=5
):
    """
    AMP + 梯度累积组合
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        if len(train_loader) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
```

#### A.3.6 Early Stopping — 见好就收的智慧

```python
class EarlyStopping:
    """
    早停机制
    
    核心思想：
        - 监控验证集指标
        - 如果连续 patience 个 epoch 没有提升，停止训练
    
    参数：
        - patience: 容忍的 epoch 数
        - min_delta: 最小提升阈值
        - mode: 'min' 或 'max'
    """
    
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
        else:
            self.monitor_op = lambda x, y: x > y
    
    def __call__(self, score):
        """
        调用早停检查
        
        返回：
            True: 应该停止训练
            False: 继续训练
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score - self.min_delta):
            # 有提升，重置计数器
            self.best_score = score
            self.counter = 0
            return False
        else:
            # 没有提升，增加计数器
            self.counter += 1
            print(f"早停计数: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                print("触发早停，停止训练")
                self.early_stop = True
                return True
            
            return False

# 使用示例
def train_with_early_stopping(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion,
    epochs=100
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    early_stopping = EarlyStopping(patience=10, mode='min')
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # 早停检查
        if early_stopping(val_loss):
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
```

#### A.3.7 数据增强 — 扩充修炼资源

**图像数据增强**：

```python
import torchvision.transforms as transforms

def get_image_augmentation():
    """
    图像数据增强流程
    
    策略：
        - 随机翻转
        - 随机旋转
        - 随机裁剪
        - 颜色抖动
        - 标准化
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),           # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),             # 随机垂直翻转
        transforms.RandomRotation(degrees=15),            # 随机旋转 ±15 度
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机裁剪 + 缩放
        transforms.ColorJitter(brightness=0.2, contrast=0.2,    # 颜色抖动
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# 使用示例
train_dataset = datasets.ImageFolder('data/train', transform=get_image_augmentation()[0])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**文本数据增强**：

```python
import random
import nlpaug.augmenter.word as naw

def text_augmentation(text, aug_type='synonym'):
    """
    文本数据增强
    
    方法：
        1. 同义词替换
        2. 随机插入
        3. 随机删除
        4. 回译
    """
    if aug_type == 'synonym':
        # 同义词替换
        aug = naw.SynonymAug(aug_src='wordnet')
        return aug.augment(text)[0]
    
    elif aug_type == 'random_insert':
        # 随机插入
        aug = naw.RandomWordAug(action='insert')
        return aug.augment(text)[0]
    
    elif aug_type == 'random_delete':
        # 随机删除
        aug = naw.RandomWordAug(action='delete')
        return aug.augment(text)[0]
    
    elif aug_type == 'back_translation':
        # 回译
        aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )
        return aug.augment(text)[0]

# 在训练循环中使用
def augment_text_batch(texts, augment_prob=0.3):
    """
    批量文本增强
    """
    augmented_texts = []
    for text in texts:
        if random.random() < augment_prob:
            aug_type = random.choice(['synonym', 'random_insert', 'random_delete'])
            aug_text = text_augmentation(text, aug_type=aug_type)
            augmented_texts.append(aug_text)
        else:
            augmented_texts.append(text)
    return augmented_texts
```

**Mixup 和 CutMix 增强技术**：

```python
def mixup_data(x, y, alpha=0.4):
    """
    Mixup 数据增强
    
    原理：
        - 将两张图片线性插值
        - 标签也相应插值
        - 公式：x_mix = λ * x_1 + (1-λ) * x_2
    
    参数：
        alpha: Beta 分布的参数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup 损失函数
    
    损失 = λ * loss(y_a) + (1-λ) * loss(y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 使用示例
def train_with_mixup(model, train_loader, optimizer, criterion, alpha=0.4, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Mixup 增强
            data, target_a, target_b, lam = mixup_data(data, target, alpha)
            
            optimizer.zero_grad()
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# CutMix: 剪切混合（类似 Mixup，但直接在图片上剪切一块区域）
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix 数据增强
    
    原理：
        - 将一张图片的一块矩形区域剪切
        - 用另一张图片的对应区域填充
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    # 生成剪切区域
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机位置
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行剪切混合
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整 lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam
```

#### A.3.8 权重初始化 — 良好的开端

```python
def init_weights(m):
    """
    权重初始化策略
    
    目标：
        - 避免梯度消失/爆炸
        - 保持激活值的方差稳定
    """
    if isinstance(m, nn.Linear):
        # Xavier/Glorot 初始化
        # 适用于 tanh, sigmoid 等对称激活函数
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Conv2d):
        # Kaiming/He 初始化
        # 适用于 ReLU, LeakyReLU 等非对称激活函数
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

# 使用示例
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 应用初始化
model.apply(init_weights)

# 预训练模型的初始化
def init_from_pretrained(model, pretrained_path):
    """
    从预训练模型初始化
    """
    state_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    
    # 1. 加载匹配的权重
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    # 2. 更新模型权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # 3. 打印未初始化的层
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
    if missing_keys:
        print(f"未初始化的层: {missing_keys}")
        # 随机初始化剩余层
        model.apply(init_weights)
    
    return model

# Layer-wise Learning Rate Decay
def get_layerwise_lr(model, base_lr=0.001, decay=0.95):
    """
    层级学习率衰减
    
    原理：
        - 浅层（靠近输入）使用较小的学习率
        - 深层（靠近输出）使用较大的学习率
    
    公式：
        lr_layer = base_lr * decay^(depth - 1)
    """
    layers = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            # 计算层级深度
            depth = len(name.split('.'))
            lr = base_lr * (decay ** (depth - 1))
            layers.append({'params': param, 'lr': lr})
    
    return layers

# 使用示例
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(
    get_layerwise_lr(model, base_lr=0.001, decay=0.9),
    weight_decay=0.01
)
```

### A.4 修炼心得总结

| 技术 | 核心思想 | 适用场景 |
|------|----------|----------|
| **SGD** | 简单直接，稳定收敛 | 小数据集、需要精确控制 |
| **Momentum** | 引入惯性，加速收敛 | 有平坦区域/峡谷的损失曲面 |
| **AdaGrad** | 自适应学习率 | 稀疏特征 |
| **RMSprop** | 避免学习率过小 | RNN、在线学习 |
| **Adam** | 结合动量 + 自适应 | **默认选择**，大多数场景 |
| **AdamW** | 解耦权重衰减 | 大模型、预训练 |
| **MSE** | 对大误差敏感 | 回归问题 |
| **MAE** | 对异常值不敏感 | 有噪声的回归 |
| **Huber** | 结合 MSE/MAE 优点 | 鲁棒回归 |
| **CrossEntropy** | 概率分布差异 | 多分类 |
| **Focal** | 聚焦困难样本 | 类别不平衡 |

---

## 附录 B：灵力精炼术 — 模型压缩与优化

> *"炼丹之术，不在于炼出多么强大的丹药，而在于能否用最少的灵力，炼出效力最高的成丹。*
> *模型亦是如此，压缩与优化，是修炼者必须掌握的精炼之术。"*
> *——《焚诀》精炼篇*

### B.1 为何需要模型压缩？

**问题背景**：

随着模型规模不断增大，面临三个严峻挑战：

| 挑战 | 具体表现 | 影响 |
|------|---------|------|
| **存储成本高** | 7B 参数模型需要 14GB（FP32） | 部署成本高昂 |
| **推理速度慢** | GPT-3 单次推理需要数秒 | 用户体验差 |
| **资源受限** | 移动端设备显存/内存有限 | 无法部署大模型 |

**压缩目标**：

- **减少模型体积**：FP32 → INT8，体积缩小 4 倍
- **加速推理速度**：速度提升 2-5 倍
- **保持模型精度**：精度损失 < 1-2%

### B.2 权重量化（Quantization）

量化是将高精度浮点数（FP32）转换为低精度整数（INT8/INT4）的技术，就像将灵力压缩为更精炼的形式。

#### B.2.1 训练后量化（Post-Training Quantization, PTQ）

**核心思想**：训练完成后，将模型权重转换为低精度格式，无需重新训练。

**量化公式**：

```
q = round(f / scale + zero_point)
```

其中：
- `f`：原始 FP32 数值
- `q`：量化后的整数
- `scale`：缩放因子
- `zero_point`：零点偏移

**完整实现**：

```python
import torch
import torch.quantization

def post_training_quantize(model, example_input):
    """
    训练后量化 — PTQ
    
    参数:
        model: 训练好的模型（FP32）
        example_input: 示例输入（用于校准）
    """
    # 1. 设置模型为评估模式
    model.eval()
    
    # 2. 配置量化（对称量化，无需 zero_point）
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 3. 准备量化（插入量化/反量化节点）
    model_prepared = torch.quantization.prepare(model, inplace=False)
    
    # 4. 校准（使用代表性数据）
    print("正在校准量化模型...")
    with torch.no_grad():
        for _ in range(100):  # 使用 100 个样本进行校准
            output = model_prepared(example_input)
    
    # 5. 转换为量化模型
    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    
    # 6. 对比性能
    print(f"原始模型大小: {get_model_size(model):.2f} MB")
    print(f"量化模型大小: {get_model_size(quantized_model):.2f} MB")
    
    return quantized_model


def get_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024
```

**量化效果示例**：

```
原始模型大小: 125.34 MB
量化模型大小: 31.42 MB  ← 缩小 4 倍！

推理速度:
  原始模型: 15.2 ms
  量化模型: 3.8 ms       ← 加速 4 倍！
```

#### B.2.2 量化感知训练（Quantization-Aware Training, QAT）

**问题**：PTQ 可能导致较大精度损失（尤其是小模型）。

**解决方案**：在训练时就模拟量化误差，让模型适应量化后的精度。

**完整实现**：

```python
import torch
import torch.nn as nn
import torch.quantization

def quantization_aware_training(model, train_loader, val_loader, num_epochs=10):
    """
    量化感知训练 — QAT
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
    """
    # 1. 配置 QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 2. 准备 QAT（插入 fake quant 节点）
    model_prepared = torch.quantization.prepare_qat(model, inplace=False)
    
    # 3. 正常训练流程（前向/反向传播）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_prepared = model_prepared.to(device)
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model_prepared.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model_prepared(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证
        acc = validate(model_prepared, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {acc:.4f}")
    
    # 4. 转换为真正的量化模型
    quantized_model = torch.quantization.convert(model_prepared.eval(), inplace=False)
    
    return quantized_model


def validate(model, val_loader, device):
    """验证模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total
```

**QAT vs PTQ 对比**：

| 方法 | 精度损失 | 训练时间 | 适用场景 |
|------|---------|---------|---------|
| **PTQ** | 1-5% | 无需训练 | 大模型、快速部署 |
| **QAT** | < 1% | 需要重新训练 | 小模型、高精度要求 |

#### B.2.3 动态量化 vs 静态量化

| 类型 | 权重量化 | 激活量化 | 速度 | 适用场景 |
|------|---------|---------|------|---------|
| **动态量化** | ✅ | ❌（推理时量化） | 中等 | LSTM/RNN、NLP 模型 |
| **静态量化** | ✅ | ✅（需要校准） | 快 | CNN、推荐模型 |

### B.3 知识蒸馏（Knowledge Distillation）

**核心思想**：用大型"教师"模型（Teacher）的知识，训练小型的"学生"模型（Student），让小模型达到接近大模型的性能。

这就像让一位大师（Teacher）指点年轻弟子（Student），弟子可以更快地掌握精华。

#### B.3.1 软标签（Soft Labels）vs 硬标签

**硬标签（Hard Labels）**：
```
[0, 0, 1, 0]  ← 只知道正确答案是第 3 类
```

**软标签（Soft Labels）**：
```
[0.01, 0.05, 0.92, 0.02]  ← 知道各个类别的置信度
                              - 第 3 类置信度最高
                              - 第 2 类也有一定可能（"猫"和"狗"相似）
```

**软标签的优势**：
- 包含更多信息（类间相似性）
- 指导学生模型学习"类间关系"
- 提高泛化能力

#### B.3.2 温度参数（Temperature）

**公式**：

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

- **T=1**：正常 softmax（硬标签）
- **T > 1**：分布更平滑（软标签），信息更丰富
- **T → ∞**：所有类别的概率趋于均匀

**温度示例**：

```python
import torch
import torch.nn.functional as F

def softmax_with_temperature(logits, temperature=1.0):
    """
    带温度的 Softmax
    
    参数:
        logits: 模型输出 (未经过 softmax)
        temperature: 温度参数
    """
    return F.softmax(logits / temperature, dim=-1)


# 示例
logits = torch.tensor([2.0, 5.0, 1.0, 0.5])

print("T=1.0 (正常):", softmax_with_temperature(logits, T=1.0))
# [0.0149, 0.8679, 0.1091, 0.0081]

print("T=5.0 (平滑):", softmax_with_temperature(logits, T=5.0))
# [0.2487, 0.2869, 0.2365, 0.2279] ← 分布更均匀
```

#### B.3.3 蒸馏损失函数

**完整蒸馏损失**：

```
L_distill = α * L_soft(1-α) * L_hard
```

- `L_soft`：软标签损失（KL 散度）
- `L_hard`：硬标签损失（交叉熵）
- `α`：蒸馏权重（通常 α=0.5 ~ 0.7）

**完整实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    
    参数:
        temperature: 温度参数
        alpha: 蒸馏权重
    """
    
    def __init__(self, temperature=5.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        参数:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        """
        # 1. 软标签损失（KL 散度）
        # 教师 log_softmax
        teacher_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        # 学生 softmax
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # 2. 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 3. 加权组合
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


def train_with_distillation(teacher, student, train_loader, num_epochs=10):
    """
    知识蒸馏训练
    
    参数:
        teacher: 教师模型（已训练，冻结参数）
        student: 学生模型（待训练）
        train_loader: 训练数据
        num_epochs: 训练轮数
    """
    # 冻结教师模型
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # 训练学生模型
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    criterion = DistillationLoss(temperature=5.0, alpha=0.7)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = teacher.to(device)
    student = student.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            # 前向传播
            with torch.no_grad():
                teacher_output = teacher(data)
            student_output = student(data)
            
            # 计算蒸馏损失
            loss = criterion(student_output, teacher_output, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")
    
    return student
```

**蒸馏效果示例**：

```
教师模型（ResNet-50）:
  参数量: 25.6M
  Top-1 准确率: 76.1%

学生模型（ResNet-18）:
  无蒸馏: 69.8%
  有蒸馏: 73.5%  ← 提升 3.7%！

参数量对比: 11.7M vs 25.6M  ← 减少 54%
```

### B.4 模型剪枝（Pruning）

**核心思想**：移除模型中不重要的权重或神经元，减少模型规模，同时保持性能。

#### B.4.1 非结构化剪枝（Unstructured Pruning）

**原理**：剪掉权重值接近零的连接。

**完整实现**：

```python
import torch
import torch.nn.utils.prune as prune

def unstructured_prune_model(model, pruning_ratio=0.2):
    """
    非结构化剪枝
    
    参数:
        model: 待剪枝模型
        pruning_ratio: 剪枝比例（0.2 表示剪掉 20% 的权重）
    """
    # 1. 对所有 Linear 和 Conv2d 层应用 L1 非结构化剪枝
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
    
    # 2. 永久移除剪枝掩码（实际删除权重）
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.remove(module, 'weight')
    
    # 3. 统计稀疏度
    sparsity = calculate_sparsity(model)
    print(f"模型稀疏度: {sparsity:.2%}")
    
    return model


def calculate_sparsity(model):
    """计算模型稀疏度"""
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            total_params += module.weight.nelement()
            zero_params += (module.weight == 0).sum().item()
    
    return zero_params / total_params
```

**效果示例**：

```
原始模型:
  参数量: 10M
  准确率: 92.3%

剪枝后（sparsity=20%）:
  参数量: 10M（但 20% 为零）
  有效参数: 8M
  准确率: 91.8%  ← 几乎无损失！
```

#### B.4.2 结构化剪枝（Structured Pruning）

**问题**：非结构化剪枝虽然效果好，但硬件难以加速（稀疏矩阵计算不友好）。

**结构化剪枝**：剪掉整个通道或滤波器，硬件友好。

**完整实现**：

```python
def structured_prune_model(model, pruning_ratio=0.2):
    """
    结构化剪枝（剪枝整个通道）
    
    参数:
        model: 待剪枝模型
        pruning_ratio: 剪枝比例
    """
    # 1. 对 Conv2d 层应用 Ln 结构化剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 剪枝整个输出通道
            prune.ln_structured(
                module, 
                name='weight', 
                amount=pruning_ratio, 
                n=2,  # L2 范数
                dim=0  # 沿通道维度
            )
    
    # 2. 永久移除剪枝掩码
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')
    
    return model
```

**结构化 vs 非结构化对比**：

| 类型 | 精度损失 | 加速效果 | 硬件友好性 |
|------|---------|---------|-----------|
| **非结构化** | 小 | 无（稀疏矩阵） | 差 |
| **结构化** | 中 | 显著 | 好 |

### B.5 神经网络架构搜索（NAS）

**核心思想**：自动化搜索最优网络架构，代替人工设计。

#### B.5.1 搜索空间定义

```python
import torch
import torch.nn as nn

class SearchSpace(nn.Module):
    """
    搜索空间 — 定义可搜索的架构
    
    搜索选项:
        - 卷积核大小: 3x3, 5x5, 7x7
        - 激活函数: ReLU, LeakyReLU, Swish
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 定义候选操作
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            )
        ])
        
        # 可学习的架构参数（α）
        self.alphas = nn.Parameter(torch.zeros(len(self.ops)))
    
    def forward(self, x):
        """
        前向传播 — 加权融合所有操作
        """
        # 计算每个操作的权重
        weights = torch.softmax(self.alphas, dim=0)
        
        # 加权融合
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        
        return output
    
    def get_best_architecture(self):
        """
        获取最佳架构（权重最大的操作）
        """
        best_idx = torch.argmax(self.alphas).item()
        return self.ops[best_idx]
```

#### B.5.2 简化的 NAS 训练

```python
def train_search_space(model, train_loader, num_epochs=10):
    """
    训练搜索空间（DARTS 风格）
    
    同时优化:
        - 模型权重（θ）
        - 架构参数（α）
    """
    optimizer_weights = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'alphas' not in n],
        lr=0.001
    )
    optimizer_alphas = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'alphas' in n],
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # 1. 更新模型权重
        model.train()
        for data, labels in train_loader:
            output = model(data)
            loss = criterion(output, labels)
            
            optimizer_weights.zero_grad()
            loss.backward()
            optimizer_weights.step()
        
        # 2. 更新架构参数
        for data, labels in train_loader:
            output = model(data)
            loss = criterion(output, labels)
            
            optimizer_alphas.zero_grad()
            loss.backward()
            optimizer_alphas.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
    
    return model
```

### B.6 综合案例：完整模型压缩流程

```python
def complete_model_compression_pipeline(model, train_loader, val_loader):
    """
    完整模型压缩流程
    
    步骤:
        1. 知识蒸馏（大 → 中）
        2. 结构化剪枝（中 → 小）
        3. 量化感知训练（小 → 量化小）
    """
    
    print("=" * 50)
    print("开始模型压缩流程")
    print("=" * 50)
    
    # ===== 第一步：知识蒸馏 =====
    print("\n[第一步] 知识蒸馏")
    print(f"教师模型参数量: {count_parameters(model):,}")
    
    # 创建学生模型（更小的架构）
    student_model = create_student_model()
    print(f"学生模型参数量: {count_parameters(student_model):,}")
    
    # 蒸馏训练
    student_model = train_with_distillation(
        teacher=model,
        student=student_model,
        train_loader=train_loader,
        num_epochs=10
    )
    
    # ===== 第二步：结构化剪枝 =====
    print("\n[第二步] 结构化剪枝")
    sparsity_before = calculate_sparsity(student_model)
    print(f"剪枝前稀疏度: {sparsity_before:.2%}")
    
    student_model = structured_prune_model(student_model, pruning_ratio=0.2)
    
    # 剪枝后微调
    student_model = fine_tune_model(student_model, train_loader, num_epochs=5)
    
    # ===== 第三步：量化感知训练 =====
    print("\n[第三步] 量化感知训练")
    quantized_model = quantization_aware_training(
        student_model,
        train_loader,
        val_loader,
        num_epochs=5
    )
    
    # ===== 总结 =====
    print("\n" + "=" * 50)
    print("压缩流程完成！")
    print("=" * 50)
    print(f"原始模型: {count_parameters(model):,} 参数")
    print(f"压缩模型: {count_parameters(quantized_model):,} 参数")
    print(f"参数减少: {(1 - count_parameters(quantized_model) / count_parameters(model)):.1%}")
    
    # 性能对比
    acc_original = evaluate(model, val_loader)
    acc_compressed = evaluate(quantized_model, val_loader)
    print(f"\n准确率:")
    print(f"  原始模型: {acc_original:.4f}")
    print(f"  压缩模型: {acc_compressed:.4f}")
    print(f"  精度损失: {(acc_original - acc_compressed):.4f}")
    
    return quantized_model
```

### B.7 修炼心得总结

**压缩技术对比表**：

| 技术 | 压缩比 | 精度损失 | 训练成本 | 适用场景 |
|------|-------|---------|---------|---------|
| **PTQ 量化** | 4x | 1-3% | 无 | 快速部署 |
| **QAT 量化** | 4x | < 1% | 中 | 高精度要求 |
| **知识蒸馏** | 2-4x | 0-2% | 高 | 大→小模型 |
| **非结构化剪枝** | 2-10x | < 1% | 中 | 训练加速 |
| **结构化剪枝** | 2-4x | 1-3% | 中 | 推理加速 |
| **NAS** | 变化大 | 可控 | 极高 | 自动化设计 |

**修炼建议**：

1. **优先级顺序**：知识蒸馏 → 剪枝 → 量化
2. **组合使用**：可以同时应用多种技术（如蒸馏 + 量化）
3. **性能监控**：每步都要监控精度，避免过度压缩
4. **硬件考量**：结构化剪枝和量化对硬件更友好

---

## 第二章：时间感知 — 循环神经网络 (RNN/LSTM)

> *"万物皆有因果，过去决定现在，现在影响未来。*
> *能感知时间之流者，可预知敌手下一招。"*

### 2.1 序列的本质

火眼金睛（CNN）让你能感知**空间**中的模式。但这个世界不只有图像——还有
**随时间流逝的序列数据**：

- 文字是字符/词语的序列
- 语音是音频帧的序列
- 股价是时间点的序列
- 音乐是音符的序列

这些数据有一个共同特征：**顺序至关重要**。"我爱你"和"你爱我"包含完全相同的字，
但意思截然不同。

全连接网络和 CNN 都**没有时间概念**。它们处理每个输入都是独立的，不记得
之前看过什么。要处理序列，我们需要一种能**记住过去**的斗技。

### 2.2 RNN：时间感知的基本功法

**循环神经网络（RNN）** 的核心思想出奇简单：**让网络拥有记忆**。

```
                   ┌──────────┐
                   │          │
            ┌─────→│  隐状态 h │──────┐
            │      │ (记忆)   │      │
            │      └──────────┘      │
            │           ↑            │
            │           │            ↓
时间步 t-1  │     ┌─────┴─────┐     时间步 t+1
            └─────│  RNN 单元  │─────→
                  └─────┬─────┘
                        ↑
                        │
                    输入 x_t
```

在每个时间步 t：
1. 接收当前输入 x_t
2. 接收上一时间步的隐状态 h_{t-1}（累积的记忆）
3. 计算新的隐状态 h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
4. 输出 y_t = W_hy * h_t + b_y

**隐状态 h 就是网络的"记忆"**——它是网络对"到目前为止所见一切"的压缩总结。

用修炼的比喻：你在读一本古籍（序列数据）。每读一个字（时间步），
你脑中对这本书的理解（隐状态）就会更新一次。你不会记住每一个字的细节，
而是维护一个持续更新的**领悟**。

### 2.3 梯度消失：漫长经脉中的斗气损耗

RNN 理论上可以记住任意长的序列——但实际上做不到。

当序列很长时（比如一段 1000 字的文章），在反向传播时，梯度需要从最后一个时间步
一路传回第一个时间步。每经过一个时间步，梯度都要乘以一个权重矩阵 W_hh。

如果这个矩阵的**特征值 < 1**，梯度会**指数衰减**——这就是**梯度消失**。
如果**特征值 > 1**，梯度会**指数爆炸**——这就是**梯度爆炸**。

```
梯度消失示意：
时间步:    1      2      3      4      ...     100
梯度:    0.001  0.01   0.1    0.5    ...      1.0
         ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
         梯度在回传过程中指数衰减，远处的信息几乎无法影响参数更新
```

用修炼比喻：**斗气在漫长的经脉中流转，每经过一个穴位都会损耗一部分。
经脉越长，到达远端的斗气越微弱，直到完全消散。** 修炼者无法感知太久以前
发生的事情，形成了"记忆断层"。

这就是为什么基础 RNN 在实践中几乎不使用——它的"记忆"太短暂了。

### 2.4 LSTM：斗气门控之术

**长短期记忆网络（LSTM）** 是 Hochreiter 和 Schmidhuber 在 1997 年提出的解决方案。

LSTM 的核心思想：引入**门控机制**（gate），让网络学会**选择性地记忆和遗忘**。

LSTM 有三道门和一个细胞状态（cell state）：

```
                    ┌──────────── 细胞状态 C（长期记忆通道）──────────────┐
                    │                                                   │
                    │    ┌───┐        ┌───┐          ┌───┐            │
                    ├──→ │ × │←─f_t   │ + │←─i_t*C̃   │   │            │
                    │    └─┬─┘        └─┬─┘          │   │            │
                    │      │            │             │   │            │
                    │      ↓            ↓             │   │            ↓
                 C_{t-1}──→ 遗忘 ──→ 更新 ──→ C_t ──→│ × │←─o_t ──→ h_t
                                                      └───┘
                    ↑──────── h_{t-1} ────────↑
                                              ↑
                                            x_t
```

**三道斗气之门**：

#### 遗忘门（Forget Gate）—— 吐故

决定**丢弃**哪些旧记忆。就像修炼者定期清理脑海中无用的信息，
为新的领悟腾出空间。

```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
```

sigmoid 输出 0~1：0 表示"完全遗忘"，1 表示"完全保留"。

#### 输入门（Input Gate）—— 纳新

决定**存入**哪些新信息。就像修炼者判断当前接收到的信息中，
哪些值得记入长期记忆。

```
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)      # 哪些信息要存入
C̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)         # 候选新记忆
```

#### 细胞状态更新 —— 吐故纳新

```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

这一步至关重要：**旧记忆乘以遗忘门 + 新记忆乘以输入门**。
细胞状态 C 就像一条**高速公路**，信息可以在上面畅通无阻地流动，
不受太多非线性变换的干扰——这就是 LSTM 能保持长期记忆的关键。

#### 输出门（Output Gate）—— 有选择地表达

决定当前时间步**输出**什么。不是所有记忆都需要在此刻表达出来。

```
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

### 2.5 GRU：精简版门控

**门控循环单元（GRU）** 是 LSTM 的简化版本，由 Cho 等人在 2014 年提出。

GRU 将遗忘门和输入门合并为一个**更新门（Update Gate）**，
并将细胞状态和隐状态合并。参数更少，训练更快，效果通常与 LSTM 相当。

```
z_t = sigmoid(W_z * [h_{t-1}, x_t])          # 更新门
r_t = sigmoid(W_r * [h_{t-1}, x_t])          # 重置门
h̃_t = tanh(W * [r_t * h_{t-1}, x_t])         # 候选隐状态
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t       # 最终隐状态
```

就像一个经验丰富的修炼者开发出了**精简版门控功法**：
虽然少了一些精细控制，但施展更快捷，在大多数场景下威力相当。

### 2.6 双向 RNN：过去与未来双向感知

普通 RNN 只能从左到右（从过去到现在）处理序列。但有些任务需要
同时考虑过去和未来的上下文。

**双向 RNN** 同时运行两个 RNN：一个从左到右，一个从右到左，
然后将两个方向的隐状态拼接起来。

```
前向：  h1→ → h2→ → h3→ → h4→
                                    拼接：[h_i→; h_i←]
后向：  h1← ← h2← ← h3← ← h4←
```

就像修炼者同时感知因果之流的两个方向——既能追溯过去，又能预感未来。

**双向 LSTM 完整实现**：

```python
class BiLSTM(nn.Module):
    """
    双向 LSTM —— 同时感知过去与未来
    
    应用场景：
    - 序列标注：词性标注、命名实体识别
    - 机器翻译：理解上下文
    - 语音识别：利用前后文信息
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 双向
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层（双向的 hidden 需要拼接，所以输入是 2×hidden_dim）
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x):
        """
        x: (B, seq_len)
        输出: (B, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(x)  # (B, seq_len, embed_dim)
        
        # 双向 LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (B, seq_len, 2×hidden_dim) —— 双向隐状态已拼接
        
        # 分类
        output = self.fc(lstm_out)  # (B, seq_len, vocab_size)
        
        return output
```

**双向 vs 单向 LSTM 对比**：

| 特性 | 单向 LSTM | 双向 LSTM |
|------|----------|----------|
| 上下文信息 | 只能看到过去 | 同时看到过去和未来 |
| 推理延迟 | 可以流式处理 | 必须等待完整序列 |
| 适用场景 | 实时生成、预测 | 序列标注、分类 |

### 2.7 实战：LSTM 文本生成

```python
"""
=============================================================
  《焚诀》第二卷 - 时间感知修炼实录
  丹方：LSTM 字符级文本生成
  药材：自定义文本
  目标：学习序列中的模式，生成新的文本
=============================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============ 药材准备 ============
text = """斗之力三段斗之气化为斗技能外放斗气者方为斗师
斗师可凝斗气为铠甲斗气可化为千形万态
万般斗气皆为我用天地灵气尽入我怀
修炼之道在于持之以恒一日不练功力倒退三分"""

# 构建字符映射（灵文编码表）
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"灵文总数: {len(text)}, 独立灵文数: {vocab_size}")

# 将文本转化为数字序列
data = [char_to_idx[ch] for ch in text]

# 构建训练样本：输入序列 → 目标（下一个字符）
seq_length = 10  # 每次感知 10 个时间步
X, Y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    Y.append(data[i+seq_length])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)


# ============ 构建斗技 ============
class TemporalPerceptionLSTM(nn.Module):
    """时间感知 LSTM —— 序列模式领悟功法"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 灵文嵌入层：将离散字符映射为连续向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM 核心：时间感知经脉
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # 输出层：从隐状态预测下一个灵文
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embed = self.embedding(x)              # → (batch, seq_len, embed_dim)
        lstm_out, hidden = self.lstm(embed, hidden)  # → (batch, seq_len, hidden_dim)
        # 只取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])   # → (batch, vocab_size)
        return output, hidden


# ============ 炼丹 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TemporalPerceptionLSTM(
    vocab_size=vocab_size,
    embed_dim=32,
    hidden_dim=128,
    num_layers=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 训练
num_epochs = 200
X, Y = X.to(device), Y.to(device)

for epoch in range(num_epochs):
    model.train()
    output, _ = model(X)
    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"第 {epoch+1} 轮 | 天罚值: {loss.item():.4f}")


# ============ 生成文本 ============
def generate_text(model, start_str, length=50):
    """运用时间感知之力，预测未来的灵文序列"""
    model.eval()
    chars_generated = list(start_str)
    input_seq = [char_to_idx.get(ch, 0) for ch in start_str[-seq_length:]]

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([input_seq[-seq_length:]], dtype=torch.long).to(device)
            output, _ = model(x)
            # 使用温度采样增加多样性
            probs = torch.softmax(output / 0.8, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            chars_generated.append(idx_to_char[next_idx])
            input_seq.append(next_idx)

    return ''.join(chars_generated)


print("\n--- 时间感知生成的灵文 ---")
print(generate_text(model, "斗之力", length=30))
```

> **修炼笔记**：这个例子使用了极小的数据集，仅用于演示 LSTM 的工作原理。
> 在真实修炼中，你需要大量的文本药材才能炼出有意义的语言模型丹药。
> 后续卷章我们将使用更大规模的数据集。

---

## 第三章：天道法则 — Transformer 架构深度拆解

> *"当自注意力机制被领悟的那一天，整个修炼界的格局被彻底改写。*
> *从此，无论是语言、视觉、音频，万般道法殊途同归。*
> *这就是——天道法则。"*
> *——《焚诀》最高卷 批注*

### 3.1 为何需要 Transformer？

LSTM 能处理序列，但有两个致命弱点：

**弱点一：无法并行。**
LSTM 必须**逐步**处理序列——先计算 h_1，才能计算 h_2，再计算 h_3...
就像一个修炼者必须逐字逐句地阅读一本古籍，不能跳读或一目十行。
这导致训练速度极慢，无法充分利用 GPU 的并行算力（异火的真正威力无法释放）。

**弱点二：长距离依赖仍然困难。**
虽然 LSTM 比 RNN 好得多，但当序列很长时（比如数千个 token），
它仍然难以建立起首尾之间的直接联系。信息必须经过漫长的传递链，难免有所损耗。

2017 年，Google 的研究者发表了一篇名为 **"Attention Is All You Need"** 的论文，
提出了 **Transformer** 架构。这篇论文的标题就是它的核心宣言：**注意力就是你所需要的一切。**

Transformer 完全抛弃了循环结构，用**纯注意力机制**处理序列。
它不需要逐步处理，而是让序列中的**每个位置都能直接关注到任何其他位置**——
无论距离多远。

这就像修炼者不再需要通过漫长的经脉传递信息，而是开辟了一种
**超越空间限制的神识联通**——任何两个穴位之间都可以直接交流。

### 3.2 自注意力机制（Self-Attention）：天道法则的核心

**自注意力是 Transformer 的灵魂。理解了它，你就理解了当今 AI 最核心的原理。**

#### 直觉理解

想象一间密室中坐着一群修炼者（tokens），他们要讨论一个问题。
每个修炼者都需要从其他人那里获取信息来完善自己的理解。

自注意力的过程就是这场讨论：

1. 每个修炼者提出自己的**问题（Query）**："我需要什么信息？"
2. 每个修炼者展示自己的**标签（Key）**："我拥有什么信息？"
3. 每个修炼者准备好自己的**内容（Value）**："如果你关注我，这是我能提供的。"
4. 每个人根据自己的 Query 和所有人的 Key 计算**关注度**
5. 根据关注度，加权汇总所有人的 Value，得到自己的最终理解

更具体的例子。考虑这个句子：

> "那只猫坐在垫子上，因为**它**很累。"

当模型处理到"它"这个词时：
- "它"的 Query 在问："我指代的是什么？"
- "猫"的 Key 回答："我是一个可以被指代的实体。"
- "猫"和"它"的 Key-Query 匹配度最高
- 所以"它"会重点关注"猫"的 Value，从而理解"它"就是"猫"

#### 数学推导

**步骤 1：线性投影——生成 Q、K、V**

给定输入序列 X，形状为 (seq_len, d_model)：

```
Q = X @ W_Q    # 形状: (seq_len, d_k)   —— 生成每个 token 的"问题"
K = X @ W_K    # 形状: (seq_len, d_k)   —— 生成每个 token 的"标签"
V = X @ W_V    # 形状: (seq_len, d_v)   —— 生成每个 token 的"内容"
```

其中 W_Q, W_K 的形状为 (d_model, d_k)，W_V 的形状为 (d_model, d_v)。

**步骤 2：计算注意力分数**

```
Attention_Scores = Q @ K^T    # 形状: (seq_len, seq_len)
```

这是一个 **seq_len x seq_len** 的矩阵！第 i 行第 j 列的值表示
token i 对 token j 的"原始关注度"。每个 token 都与所有其他 token
（包括自己）计算了相关性。

**步骤 3：缩放——防止反噬**

```
Scaled_Scores = Attention_Scores / sqrt(d_k)
```

为什么要除以 sqrt(d_k)？

当 d_k 很大时，Q 和 K 的点积结果也会很大，导致 softmax 的输入值过大。
softmax 对大数值非常敏感——输入值过大会导致输出趋近于 one-hot 向量
（一个接近 1，其余接近 0），梯度几乎为零，训练停滞。

这就是一种**反噬**：能量过于集中导致系统失衡。
除以 sqrt(d_k) 是一种**斗气调节术**，让能量保持在合理范围。

**步骤 4：Softmax 归一化**

```
Attention_Weights = softmax(Scaled_Scores, dim=-1)  # 每行和为 1
```

现在每一行都是一个概率分布，表示该 token 对各个位置的关注权重。

**步骤 5：加权求和——信息汇聚**

```
Output = Attention_Weights @ V    # 形状: (seq_len, d_v)
```

每个 token 的输出是所有 token 的 Value 的加权和，权重由注意力决定。

**完整公式**：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

这就是天道法则的**核心心法**。简洁，优美，威力无穷。

#### 维度追踪（极其重要！）

假设：batch_size=B, seq_len=S, d_model=D, d_k=d_v=D_k

```
输入 X:                (B, S, D)
W_Q, W_K:              (D, D_k)
W_V:                   (D, D_k)

Q = X @ W_Q:           (B, S, D_k)      每个 token 的查询向量
K = X @ W_K:           (B, S, D_k)      每个 token 的键向量
V = X @ W_V:           (B, S, D_k)      每个 token 的值向量

Q @ K^T:               (B, S, D_k) @ (B, D_k, S) = (B, S, S)    注意力分数矩阵
/ sqrt(D_k):           (B, S, S)         缩放后的分数
softmax:               (B, S, S)         注意力权重（每行和为 1）
@ V:                   (B, S, S) @ (B, S, D_k) = (B, S, D_k)     输出
```

**关键洞察：注意力操作不改变序列长度，只可能改变特征维度。**

### 3.3 多头注意力（Multi-Head Attention）：多重神识并行感知

单一注意力头就像用一种感官去感知世界——你可能只关注到某一个方面。

**多头注意力让你同时用多种感官并行感知**，每个头关注信息的不同方面：

- Head 1 可能关注**语法结构**（"主语在哪？"）
- Head 2 可能关注**语义关系**（"谁做了什么？"）
- Head 3 可能关注**位置关系**（"相邻的词"）
- Head 4 可能关注**指代关系**（"它指的是谁？"）

```
                     输入 X
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
      ┌───────┐   ┌───────┐   ┌───────┐
      │Head 1 │   │Head 2 │   │Head 3 │    ... h 个头
      │Q1K1V1 │   │Q2K2V2 │   │Q3K3V3 │
      └───┬───┘   └───┬───┘   └───┬───┘
          │            │            │
          └────────────┼────────────┘
                       │ 拼接 (Concatenate)
                       ↓
                  ┌──────────┐
                  │ 线性投影 W_O │
                  └─────┬────┘
                        ↓
                      输出
```

数学表达：

```
head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)

MultiHead(X) = Concat(head_1, head_2, ..., head_h) @ W_O
```

#### 维度追踪

假设 d_model=512, h=8 个头：
- 每个头的维度 d_k = d_model / h = 512 / 8 = 64
- 每个头输出: (B, S, 64)
- 拼接后: (B, S, 512)   —— 恢复到 d_model
- 经过 W_O: (B, S, 512)  —— 维度不变

**多头注意力不改变输入输出的维度！** 输入 (B, S, d_model) → 输出 (B, S, d_model)。
这是一个非常优雅的设计。

### 3.4 位置编码（Positional Encoding）：时空感知法阵

自注意力有一个"盲点"：**它完全不知道 token 的顺序。**

对自注意力来说，"猫追狗"和"狗追猫"看起来完全一样（只是 token 的集合）。
因为注意力计算只涉及 token 之间的内积，与位置无关。

这就像一个拥有天眼通的修炼者——能看到万物，却不知道万物的空间位置。

**位置编码（Positional Encoding）** 就是一个"时空感知法阵"，
给每个位置注入独特的位置信息。

原始 Transformer 使用**正弦/余弦位置编码**：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中 pos 是位置索引，i 是维度索引。

**为什么用正弦/余弦？**

- 不同频率的正弦波在每个位置产生唯一的组合，就像每个位置有独特的"灵力指纹"
- 相对位置可以通过线性变换得到：PE(pos+k) 可以表示为 PE(pos) 的线性函数
- 可以外推到训练时未见过的序列长度

位置编码与输入嵌入**相加**（而非拼接）：

```
Input_with_position = Token_Embedding + Positional_Encoding
```

这二者形状都是 (S, d_model)，相加后仍然是 (S, d_model)。

> **注意**：现代模型（如 GPT、LLaMA）通常使用**可学习的位置嵌入**
> 或**旋转位置编码（RoPE）**，而非固定的正弦编码。我们会在后续卷中讨论这些进阶功法。

### 3.5 前馈网络（Feed-Forward Network）：灵力放大与变换

每个 Transformer 层在注意力之后，还有一个**逐位置的前馈网络（FFN）**：

```
FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2
```

或用 GELU 激活（现代模型更常用）：

```
FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2
```

FFN 对序列中的**每个位置独立**应用相同的两层网络。

维度变化：
```
输入:  (B, S, d_model)                    例: (B, S, 512)
W_1:   (d_model, d_ff)                    例: (512, 2048)  ← 通常 d_ff = 4 * d_model
中间:  (B, S, d_ff)                       例: (B, S, 2048)  ← "灵力放大"
W_2:   (d_ff, d_model)                    例: (2048, 512)
输出:  (B, S, d_model)                    例: (B, S, 512)   ← 恢复原始维度
```

**为什么需要 FFN？**

自注意力负责"交流"（token 之间的信息汇聚），FFN 负责"思考"（对每个 token 的
信息进行非线性变换和处理）。

用修炼比喻：注意力是修炼者之间的**交流与感知**，FFN 是每个修炼者独自
**消化吸收并转化**所获信息的过程。先与他人交流，再独自领悟——这是
Transformer 的基本节奏。

研究表明，FFN 在大模型中充当了"知识存储"的角色——大量的事实知识
编码在 FFN 的参数中。

### 3.6 层归一化（Layer Normalization）：稳定灵力之法

深度网络中，每一层的输入分布会随着训练不断变化（内部协变量偏移）。
这就像修炼中斗气流动忽强忽弱，难以控制。

**层归一化**对每个样本的特征维度进行归一化：

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

其中 gamma 和 beta 是可学习的缩放和偏移参数。

在 Transformer 中，Layer Norm 的位置有两种流派：

- **Post-LN**（原始 Transformer）：先做子层运算，再 LayerNorm
  ```
  output = LayerNorm(x + SubLayer(x))
  ```

- **Pre-LN**（现代主流）：先 LayerNorm，再做子层运算
  ```
  output = x + SubLayer(LayerNorm(x))
  ```

Pre-LN 训练更稳定，尤其在深层模型中。就像在运功之前先稳定斗气，
比运功之后再调整更安全。

### 3.7 残差连接（Residual Connections）：斗气传输捷径

和 ResNet 中一样，Transformer 的每个子层（注意力、FFN）都使用残差连接：

```
output = x + SubLayer(x)
```

这确保了：
1. **梯度可以直接流过**——即使 SubLayer 的梯度很小，x 的梯度也能直接回传
2. **底层信息不会丢失**——每一层都有"直通道"保留原始信息
3. **使得极深的网络成为可能**——GPT-3 有 96 层，没有残差连接根本训练不动

### 3.8 完整 Transformer 块：组装天道法则

现在我们把所有组件组装成一个完整的 **Transformer Encoder Block**：

```
输入 x (B, S, d_model)
       │
       ├──────────────────┐
       ↓                  │ (残差连接)
  LayerNorm               │
       ↓                  │
  Multi-Head Attention     │
       ↓                  │
       + ←────────────────┘
       │
       ├──────────────────┐
       ↓                  │ (残差连接)
  LayerNorm               │
       ↓                  │
  Feed-Forward Network     │
       ↓                  │
       + ←────────────────┘
       │
       ↓
输出 (B, S, d_model)
```

**注意**：输入和输出的维度完全相同！这意味着我们可以把多个 Transformer block
像乐高积木一样堆叠起来，构成更深的模型。

**维度变化全程追踪**（Pre-LN 版本）：

```
输入 x:                      (B, S, d_model)     例: (32, 128, 512)

1. LayerNorm(x):             (B, S, d_model)     不变
2. MultiHeadAttention:       (B, S, d_model)     不变（内部有升降维但最终恢复）
3. x + attention_output:     (B, S, d_model)     残差相加，不变

4. LayerNorm:                (B, S, d_model)     不变
5. FFN:
   - Linear1:               (B, S, d_ff)        升维到 4*d_model = 2048
   - Activation:            (B, S, d_ff)        不变
   - Linear2:               (B, S, d_model)     降维回 d_model = 512
6. x + ffn_output:          (B, S, d_model)     残差相加，不变

最终输出:                     (B, S, d_model)     与输入完全一致！
```

### 3.9 Encoder vs. Decoder：感知与生成

完整的 Transformer 有 **Encoder** 和 **Decoder** 两部分。

#### Encoder（感知模块）— "全方位感知阵"

- 每个 token 可以关注序列中的**所有**其他 token（包括左右两侧）
- 适合**理解**任务：分类、命名实体识别、句子编码
- 代表模型：**BERT**

```
"今天天气真好" → Encoder → 每个字都理解了完整语境的表示
```

#### Decoder（生成模块）— "因果推演阵"

- 每个 token **只能关注它之前的** token（因果遮罩 / causal mask）
- 适合**生成**任务：文本生成、代码生成、对话
- 代表模型：**GPT 系列**

```
"今" → "天" → "天" → "气" → "真" → "好"
每个字只能看到它左边的内容，逐字生成
```

因果遮罩的实现：在注意力分数矩阵上叠加一个上三角为负无穷的 mask，
使得 softmax 后未来位置的权重为 0。

```
遮罩矩阵 (4x4 序列)：
┌──────────────────────┐
│  0    -inf  -inf  -inf │    token 1 只能看 token 1
│  0     0    -inf  -inf │    token 2 能看 token 1, 2
│  0     0     0    -inf │    token 3 能看 token 1, 2, 3
│  0     0     0     0   │    token 4 能看所有
└──────────────────────┘
```

#### Encoder-Decoder（全能模块）— "感知-推演双修"

- Encoder 处理输入序列，Decoder 生成输出序列
- Decoder 中有一个额外的**交叉注意力层**：Q 来自 Decoder，K、V 来自 Encoder
- 适合**序列到序列**任务：翻译、摘要
- 代表模型：**T5, BART, 原始 Transformer**

```
Encoder: "I love you" → 编码表示
                              ↓ (Cross-Attention: K, V)
Decoder: "<start>" → "我" → "爱" → "你"
```

### 3.10 为何 Transformer 如此强大？

1. **并行计算**：所有 token 同时处理，充分利用 GPU 并行能力（异火全功率输出）
2. **直接长距离关联**：任意两个 token 之间只隔一步注意力，无需经过漫长传递链
3. **灵活的注意力模式**：模型自动学习应该关注什么，而非人工设计
4. **统一架构**：同一种架构适用于文本、图像、音频、视频...万法归一

这就是为什么 Transformer 被称为**天道法则**——它不是某一种特定的斗技，
而是统御万法的**根本法则**。

---

## 第四章：手搓斗技 — 从零实现 Transformer

> *"纸上得来终觉浅，绝知此事要躬行。"*
> *"唯有亲手锻造过一柄剑的人，才真正懂得剑的灵魂。"*

### 4.1 完整代码实现

下面是一个完整的 Transformer 实现。每一行代码都附有详细注释，
帮助你彻底理解每个组件。

对应的完整可运行脚本在 `notebooks/02_手搓transformer.py`。

```python
"""
=============================================================
  《焚诀》第二卷 - 天道法则·手搓实录
  从零实现完整 Transformer
=============================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
# 第一组件：位置编码 —— 时空感知法阵
# ================================================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码：为每个位置生成唯一的灵力指纹。

    使用不同频率的正弦和余弦函数，让模型能够感知序列中的位置关系。
    低维度使用高频波动（感知局部位置差异），
    高维度使用低频波动（感知全局位置结构）。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算所有位置的编码（不参与训练）
        pe = torch.zeros(max_len, d_model)            # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        # 分母项：10000^(2i/d_model)，用 exp-log 计算避免数值溢出
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) —— 添加 batch 维度

        self.register_buffer('pe', pe)  # 注册为 buffer，不参与梯度计算

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        输出: (batch, seq_len, d_model) —— 注入了位置信息
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ================================================================
# 第二组件：缩放点积注意力 —— 天道法则核心心法
# ================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力 —— Transformer 的灵魂。

    参数:
        Q: (batch, heads, seq_len, d_k)  —— 查询（我需要什么？）
        K: (batch, heads, seq_len, d_k)  —— 键（我有什么？）
        V: (batch, heads, seq_len, d_k)  —— 值（我能提供什么？）
        mask: 可选遮罩，用于因果推演阵（Decoder）

    返回:
        output: (batch, heads, seq_len, d_k)  —— 注意力输出
        weights: (batch, heads, seq_len, seq_len)  —— 注意力权重
    """
    d_k = Q.size(-1)

    # 步骤 1: 计算注意力分数 = Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (B, h, S, S)

    # 步骤 2: 缩放 —— 防止斗气过于集中导致反噬
    scores = scores / math.sqrt(d_k)

    # 步骤 3: 应用遮罩（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 步骤 4: Softmax 归一化 —— 将分数转化为概率分布
    weights = F.softmax(scores, dim=-1)              # (B, h, S, S)

    # 步骤 5: 加权求和 —— 信息汇聚
    output = torch.matmul(weights, V)                # (B, h, S, d_k)

    return output, weights


# ================================================================
# 第三组件：多头注意力 —— 多重神识并行感知
# ================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力：同时用多个感知维度观察世界。

    将 d_model 维特征分成 num_heads 个头，每个头独立做注意力，
    最后拼接起来再做一次线性变换。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性投影（三把灵力转化之钥）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影（多重神识汇聚之阵）
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        参数形状: (batch, seq_len, d_model)
        输出形状: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_Q(query)  # (B, S, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # 2. 拆分成多头: (B, S, d_model) → (B, S, h, d_k) → (B, h, S, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 执行注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 合并多头: (B, h, S, d_k) → (B, S, h, d_k) → (B, S, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5. 输出投影
        output = self.W_O(attn_output)    # (B, S, d_model)
        return self.dropout(output)


# ================================================================
# 第四组件：前馈网络 —— 灵力放大与变换
# ================================================================
class FeedForward(nn.Module):
    """
    逐位置前馈网络：对每个位置的特征独立做非线性变换。
    先升维（灵力放大），再降维（灵力凝聚），中间经过激活函数。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 升维：灵力放大
        self.linear2 = nn.Linear(d_ff, d_model)    # 降维：灵力凝聚
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, S, d_model) → (B, S, d_model)
        中间: (B, S, d_ff)
        """
        x = F.gelu(self.linear1(x))   # GELU 激活：平滑的灵力激发
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ================================================================
# 第五组件：Transformer Encoder Block —— 天道法则·感知阵
# ================================================================
class TransformerEncoderBlock(nn.Module):
    """
    Transformer 编码器块（Pre-LN 版本）

    结构：
    x → LayerNorm → MultiHeadAttention → + (残差) →
      → LayerNorm → FeedForward → + (残差) → output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层 1：多头自注意力 + 残差连接
        normed = self.norm1(x)
        attn_output = self.attention(normed, normed, normed, mask)
        x = x + attn_output                  # 残差连接：斗气捷径

        # 子层 2：前馈网络 + 残差连接
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout(ffn_output)     # 残差连接：斗气捷径

        return x


# ================================================================
# 第六组件：Transformer Decoder Block —— 天道法则·推演阵
# ================================================================
class TransformerDecoderBlock(nn.Module):
    """
    Transformer 解码器块

    与编码器块的区别：
    1. 自注意力使用因果遮罩（只能看过去）
    2. 额外的交叉注意力层（Q 来自 Decoder，K/V 来自 Encoder）
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 子层 1：因果自注意力（只看过去）
        normed = self.norm1(x)
        self_attn_output = self.self_attention(normed, normed, normed, tgt_mask)
        x = x + self_attn_output

        # 子层 2：交叉注意力（从 Encoder 获取信息）
        normed = self.norm2(x)
        cross_attn_output = self.cross_attention(
            normed, encoder_output, encoder_output, src_mask
        )
        x = x + cross_attn_output

        # 子层 3：前馈网络
        normed = self.norm3(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout(ffn_output)

        return x


# ================================================================
# 完整 Transformer 模型
# ================================================================
class MiniTransformer(nn.Module):
    """
    迷你 Transformer —— 天道法则完整实现

    包含：
    - Token Embedding（灵文嵌入）
    - Positional Encoding（位置编码）
    - N 层 Encoder blocks
    - N 层 Decoder blocks
    - 输出线性层
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256,
                 num_heads=8, d_ff=1024, num_layers=4,
                 max_len=512, dropout=0.1):
        super().__init__()

        # 灵文嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 嵌入缩放因子
        self.scale = math.sqrt(d_model)

        # Encoder 堆叠
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder 堆叠
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # 输出投影
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        """编码器：感知输入序列"""
        x = self.pos_encoding(self.src_embedding(src) * self.scale)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码器：基于编码结果生成输出"""
        x = self.pos_encoding(self.tgt_embedding(tgt) * self.scale)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        完整前向传播：编码 → 解码 → 投影

        src: (B, S_src) —— 源序列
        tgt: (B, S_tgt) —— 目标序列
        输出: (B, S_tgt, tgt_vocab_size) —— 每个位置对词表的预测分布
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

    @staticmethod
    def generate_causal_mask(size):
        """生成因果遮罩：下三角为 1，上三角为 0"""
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)
```

### 4.2 验证实现：序列复制任务

我们用一个简单的任务来验证 Transformer 实现的正确性——**序列复制**：
模型的目标是学会将输入序列原样复制到输出。

```python
# ============ 验证：序列复制任务 ============
def train_copy_task():
    """
    序列复制任务：输入 [1, 5, 3, 7, 2] → 输出 [1, 5, 3, 7, 2]

    这是验证 Transformer 实现是否正确的经典方法。
    如果模型能学会完美复制，说明各组件工作正常。
    """
    vocab_size = 20        # 灵文种类：20 种
    seq_len = 10           # 序列长度：10
    d_model = 64           # 斗气维度
    num_heads = 4          # 感知头数
    d_ff = 256             # 灵力放大倍数
    num_layers = 2         # 阵法层数
    batch_size = 64        # 批量大小
    num_epochs = 100       # 修炼轮数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = MiniTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=seq_len + 10,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("开始序列复制修炼...")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        model.train()

        # 生成随机训练数据：值在 [1, vocab_size-1]（0 保留为 padding）
        src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        # 目标序列 = 源序列（复制任务）
        tgt_input = src[:, :-1]     # Decoder 输入：去掉最后一个 token
        tgt_output = src[:, 1:]     # 预测目标：去掉第一个 token

        # 生成因果遮罩
        tgt_mask = MiniTransformer.generate_causal_mask(tgt_input.size(1)).to(device)

        # 前向传播
        logits = model(src, tgt_input, tgt_mask=tgt_mask)

        # 计算损失
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            # 测试：检查模型是否学会了复制
            model.eval()
            with torch.no_grad():
                test_src = torch.randint(1, vocab_size, (1, seq_len)).to(device)
                test_tgt = test_src[:, :-1]
                test_mask = MiniTransformer.generate_causal_mask(
                    test_tgt.size(1)
                ).to(device)
                pred = model(test_src, test_tgt, tgt_mask=test_mask)
                pred_tokens = pred.argmax(dim=-1)

                print(f"第 {epoch+1} 轮 | 天罚值: {loss.item():.4f}")
                print(f"  源序列:   {test_src[0].tolist()}")
                print(f"  预测序列: {[test_src[0, 0].item()] + pred_tokens[0].tolist()}")

    print("\n天道法则修炼完成！")


# 运行
if __name__ == "__main__":
    train_copy_task()
```

### 4.3 维度变化全程追踪

让我们用一个具体的例子追踪张量在 Transformer 中的完整流转：

```
配置: batch=2, src_len=10, tgt_len=8, d_model=64, heads=4, d_ff=256, vocab=20

═══════════════════ ENCODER ═══════════════════

源序列 src:                          (2, 10)              整数 token IDs
src_embedding(src):                  (2, 10, 64)          灵文嵌入
× sqrt(d_model):                     (2, 10, 64)          缩放
+ positional_encoding:               (2, 10, 64)          注入位置信息

--- Encoder Block (重复 N 次) ---
LayerNorm:                           (2, 10, 64)          稳定灵力
  Q = x @ W_Q:                      (2, 10, 64)
  拆分多头: Q:                        (2, 4, 10, 16)       4 头, 每头 16 维
  K 同理:                            (2, 4, 10, 16)
  V 同理:                            (2, 4, 10, 16)
  Q @ K^T:                          (2, 4, 10, 10)       注意力分数
  / sqrt(16):                        (2, 4, 10, 10)       缩放
  softmax:                           (2, 4, 10, 10)       注意力权重
  @ V:                               (2, 4, 10, 16)       加权求和
  合并多头:                           (2, 10, 64)          拼接恢复
  W_O:                               (2, 10, 64)          输出投影
+ 残差 (x):                          (2, 10, 64)          斗气捷径

LayerNorm:                           (2, 10, 64)
  Linear1 (64→256):                  (2, 10, 256)         灵力放大
  GELU:                              (2, 10, 256)         灵力激发
  Linear2 (256→64):                  (2, 10, 64)          灵力凝聚
+ 残差 (x):                          (2, 10, 64)          斗气捷径

Encoder 最终输出:                     (2, 10, 64)

═══════════════════ DECODER ═══════════════════

目标序列 tgt:                         (2, 8)
tgt_embedding(tgt):                  (2, 8, 64)
× sqrt(d_model) + pos_encoding:      (2, 8, 64)

--- Decoder Block (重复 N 次) ---

[因果自注意力]
  Q, K, V from tgt:                  (2, 4, 8, 16)       各 4 头
  Q @ K^T:                          (2, 4, 8, 8)         注意力分数
  + 因果遮罩:                        (2, 4, 8, 8)         屏蔽未来
  softmax → @ V:                    (2, 4, 8, 16)
  合并 → W_O:                       (2, 8, 64)
  + 残差:                           (2, 8, 64)

[交叉注意力]
  Q from decoder:                    (2, 4, 8, 16)        Decoder 提问
  K, V from encoder:                 (2, 4, 10, 16)       Encoder 回答
  Q @ K^T:                          (2, 4, 8, 10)        ← 注意尺寸！
  softmax → @ V:                    (2, 4, 8, 16)
  合并 → W_O:                       (2, 8, 64)
  + 残差:                           (2, 8, 64)

[前馈网络]
  同 Encoder

Decoder 最终输出:                     (2, 8, 64)

═══════════════════ OUTPUT ═══════════════════

output_projection (64 → 20):         (2, 8, 20)           每个位置对词表的预测
```

**关键发现**：
- Encoder 输出长度 = 源序列长度 (10)
- Decoder 输出长度 = 目标序列长度 (8)
- 交叉注意力中 Q 和 K 的序列长度可以不同！
- 每个 block 的输入输出维度完全一致，可以任意堆叠

---

## 第四章（进阶篇）：天道法则的进化之路 — 前沿技术融合

> *"天道法则（Transformer）并非一成不变的功法，它在不断进化。*
> *从最初的朴素自注意力，到 Flash Attention 的内存优化，*
> *从固定的正弦编码，到 RoPE 的相对位置感知，*
> *每一次进化，都让这座法阵变得更加精妙、更加强大。"*

### 4.1 Flash Attention — 闪电般的灵力操控

#### 4.1.1 传统注意力的致命弱点

回顾我们之前实现的缩放点积注意力：

```python
# 传统实现（朴素）
def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))   # 注意力分数矩阵
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)              # 注意力权重矩阵
    output = torch.matmul(weights, V)                # 加权求和
    return output, weights
```

这个实现看起来完美无缺，但隐藏着一个致命的**内存瓶颈**：

**问题分析**：

假设我们处理一个长度为 4096 的序列（这在大模型中很常见）：

```
序列长度 S = 4096
注意力矩阵 S x S = 4096 x 4096 = 16,777,216 个浮点数
```

这个矩阵需要占用多少显存？

```
16,777,216 个 FP32 数值 × 4 字节/数值 = 64 MB
```

**64 MB 看起来不多？别急——这只是这一个矩阵！**

在完整的注意力计算中，我们需要：

1. **Q @ K^T** → 注意力分数矩阵 (S × S)
2. **softmax 后的权重矩阵** (S × S)
3. 这些中间结果需要保存在内存中，直到最后一步

对于 batch_size=1、seq_len=4096、d_model=4096 的场景：

- 注意力矩阵: 64 MB
- 注意力权重矩阵: 64 MB
- **单个 batch 的注意力层就需要 128 MB**

如果 batch_size=32（常见的训练批次），需要的显存就是：

```
128 MB × 32 = 4,096 MB = 4 GB
```

**这还只是一层注意力！** 对于一个 32 层的模型，光是注意力层就需要 **128 GB 的显存**——远超绝大多数 GPU 的容量。

用修炼比喻：这就像你在施展天道法则时，必须在经脉中同时维持**数千个中间灵力漩涡**，每个漩涡都要占用你的丹田空间。经脉容量有限，你无法支持长序列的修炼。

#### 4.1.2 Flash Attention 的核心洞察

Flash Attention (2022) 来自斯坦福大学的 Tri Dao 等人，它用一个天才般的洞察解决了这个问题：

**关键发现：我们不需要显式地计算并存储完整的注意力矩阵！**

重新审视注意力公式：

```
Output = softmax(Q @ K^T / sqrt(d_k)) @ V
```

如果将其展开：

```
Output_i = Σ_j softmax((Q_i @ K_j^T) / sqrt(d_k)) × V_j
```

**每个输出位置 i，只需要知道它与所有 j 的 softmax 加权结果，不需要知道所有 j 之间的相互关系。**

Flash Attention 的核心思想：

1. **分块计算（Tiling）**：将序列分成小块，逐块计算
2. **融合核（Fused Kernel）**：将 softmax 和 matmul 融合在一个 CUDA 核中
3. **在线 softmax（Online Softmax）**：逐步累积 softmax 结果，不需要存储完整矩阵
4. **重计算（Recomputation）**：前向传播不保存中间结果，反向传播时重新计算（用算力换内存）

#### 4.1.3 数学原理：在线 Softmax

传统 softmax：

```python
def softmax(x):
    # x: (seq_len,) —— 注意力分数向量
    exp_x = torch.exp(x - x.max())        # 数值稳定性
    return exp_x / exp_x.sum()
```

这需要存储整个 `exp_x` 向量。

**在线 Softmax** 逐步累积：

```python
class OnlineSoftmax:
    """
    在线 Softmax —— 逐步累积，无需存储完整向量
    
    数学原理：
    softmax(x) = exp(x) / Σ exp(x) 可以通过递归计算
    
    对于序列 x_1, x_2, ..., x_n：
    - 第 1 步：m_1 = x_1, d_1 = 1, softmax_1 = 1
    - 第 2 步：m_2 = max(m_1, x_2), d_2 = d_1 + exp(x_2 - m_2)
              softmax_2 = exp(x_i - m_2) / d_2
    - ...
    - 第 n 步：softmax_n = exp(x_i - m_n) / d_n
    
    其中 m 是当前最大值，d 是归一化分母
    """
    def __init__(self):
        self.max_val = float('-inf')
        self.sum_exp = 0.0
    
    def update(self, x):
        """处理一个新的元素"""
        if x > self.max_val:
            # 新的最大值，需要调整所有之前的 exp
            self.sum_exp = self.sum_exp * torch.exp(self.max_val - x) + 1.0
            self.max_val = x
        else:
            # 旧的最大值保持不变
            self.sum_exp += torch.exp(x - self.max_val)
    
    def get_result(self, x):
        """获取当前元素的 softmax 值"""
        return torch.exp(x - self.max_val) / self.sum_exp
```

关键洞察：**每一步只需要保存两个标量（max_val 和 sum_exp），而不是整个向量！**

#### 4.1.4 Flash Attention 算法流程

```python
def flash_attention_forward(Q, K, V, block_size=128):
    """
    Flash Attention 前向传播（简化版）
    
    参数:
        Q, K, V: (B, heads, S, d_k)
        block_size: 块大小（通常 128 或 256）
    
    核心思想：
    1. 将序列分成 blocks
    2. 对每个输出块，逐步遍历所有输入块
    3. 在遍历过程中，逐步累积 softmax 结果
    4. 只需存储最终输出，不存储中间注意力矩阵
    """
    B, H, S, D_k = Q.shape
    O = torch.zeros_like(Q)  # 输出矩阵
    
    # 将序列分成块
    num_blocks = (S + block_size - 1) // block_size
    
    for i in range(num_blocks):  # 输出块
        # 当前输出块的索引范围
        start_i, end_i = i * block_size, min((i + 1) * block_size, S)
        Q_block = Q[:, :, start_i:end_i, :]
        
        # 在线 softmax 的累积器
        O_block = torch.zeros_like(Q_block)
        l = torch.zeros(B, H, end_i - start_i, 1)  # softmax 分母累积
        m = torch.full((B, H, end_i - start_i, 1), float('-inf'))  # 最大值累积
        
        for j in range(num_blocks):  # 输入块
            start_j, end_j = j * block_size, min((j + 1) * block_size, S)
            K_block = K[:, :, start_j:end_j, :]
            V_block = V[:, :, start_j:end_j, :]
            
            # 计算当前块之间的注意力分数
            S_ij = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(D_k)
            
            # 更新最大值和归一化分母（在线 softmax 第一步）
            new_m = torch.maximum(m, S_ij.max(dim=-1, keepdim=True)[0])
            l = l * torch.exp(m - new_m) + torch.exp(S_ij - new_m).sum(dim=-1, keepdim=True)
            m = new_m
            
            # 计算加权和（在线 softmax 第二步）
            P_ij = torch.exp(S_ij - m)  # 局部 softmax（未归一化）
            O_block = O_block + torch.matmul(P_ij, V_block)
        
        # 最终归一化
        O[:, :, start_i:end_i, :] = O_block / l
    
    return O
```

**内存复杂度对比**：

| 方法 | 内存复杂度 (单层) | seq_len=4096, batch=32 |
|------|------------------|------------------------|
| 传统注意力 | O(S²) + O(S × d_k) | ~4 GB |
| Flash Attention | O(S × d_k) | ~200 MB |

**显存节省：20 倍！**

#### 4.1.5 实际应用：使用 Flash Attention 2

PyTorch 2.0+ 已内置 Flash Attention 2 支持：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashMultiHeadAttention(nn.Module):
    """
    Flash Attention 多头注意力 —— 内存高效的天道法则
    
    使用 PyTorch 内置的 scaled_dot_product_attention
    (Flash Attention 2 实现)
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        使用 Flash Attention 2 计算
        
        参数:
            x: (B, S, d_model)
            mask: 可选，用于因果遮罩
        """
        B, S, D = x.shape
        
        # 1. 线性投影
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (B, H, S, d_k)
        
        # 2. Flash Attention 2（PyTorch 内置）
        # 这会自动选择最优的 CUDA kernel（包括 Flash Attention）
        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,      # 可选遮罩
            dropout_p=0.0,        # 训练时可用
            is_causal=False        # 因果模式（用于 GPT）
        )
        # output: (B, H, S, d_k)
        
        # 3. 合并多头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_O(output)
        
        return output
```

**关键优势**：

1. **内存效率**：可处理 32k+ 长度的序列
2. **速度提升**：通常比传统实现快 2-4 倍
3. **数值稳定性**：内置的数值稳定性优化

#### 4.1.6 修炼心得：从朴素到精妙

| 维度 | 传统注意力 | Flash Attention |
|------|-----------|----------------|
| 内存占用 | O(S²) - 显存爆炸 | O(S×d_k) - 线性增长 |
| 计算复杂度 | O(S²×d_k) | O(S²×d_k) - 相同 |
| 实际速度 | 中等（受限于显存带宽） | 快 2-4x（融合核优化） |
| 适用场景 | 短序列（< 2048） | 长序列（> 2048） |

Flash Attention 不是改变了注意力的数学，而是改变了**计算方式**。这就像修炼者学会了更精妙的斗气操控技巧——同样的功力（数学公式），却能释放出更强大的威力。

---

### 4.2 RoPE（旋转位置编码） — 相对位置的感知

#### 4.2.1 位置编码的进化史

回顾第三章我们介绍的位置编码：

**原始 Transformer（2017）：正弦位置编码**

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 绝对位置编码：与 token embedding 相加
        return x + self.pe[:x.size(1), :]
```

**问题**：这是**绝对位置编码**。

- "猫" 在位置 3 有固定的编码向量
- 不管 "猫" 与 "狗" 的距离是 1 还是 100，它们的编码都不变

**直觉告诉我们：相对位置更重要。**

考虑这个句子：

> "小明昨天买了**苹果**，今天买了**香蕉**。"

模型需要理解：
- "苹果"（位置 5）和 "昨天"（位置 3）的关系
- "香蕉"（位置 9）和 "今天"（位置 7）的关系

**两个关系是完全相同的**（都是"名词 + 时间词，距离=2），但绝对位置编码会给出完全不同的表示！

#### 4.2.2 RoPE 的核心思想

**旋转位置编码（Rotary Positional Embedding, RoPE）** 由 Google 提出（2021），核心思想：

**用复数旋转来编码相对位置信息。**

**数学基础：二维旋转**

将向量看作复数：`v = [x, y]` → `v = x + iy`

旋转 θ 角度：

```
v_rotated = v × e^{iθ}
          = (x + iy) × (cosθ + i·sinθ)
          = x·cosθ - y·sinθ + i(x·sinθ + y·cosθ)
          = [x·cosθ - y·sinθ, x·sinθ + y·cosθ]
```

用矩阵表示：

```
[cosθ  -sinθ]
[sinθ   cosθ]
```

**关键洞察**：旋转具有群性质：

```
Rot(θ₁) × Rot(θ₂) = Rot(θ₁ + θ₂)
```

这意味着：
- 位置 3 的编码 = 旋转 3θ
- 位置 5 的编码 = 旋转 5θ
- **相对距离 = 2** → 相对编码 = 旋转 (5θ - 3θ) = 旋转 2θ

**相对位置通过简单的减法就能得到！**

#### 4.2.3 高维 RoPE

对于 d_model 维向量，我们将其分成 d_model/2 个二维对：

```
[x₁, x₂, x₃, x₄, ...] → [(x₁, x₂), (x₃, x₄), ...]
```

每个二维对独立旋转：

```
频率 θ_i = 1 / 10000^(2i/d_model)

对于第 i 个二维对（对应维度 2i 和 2i+1）：
  [x_{2i}, x_{2i+1}] → [x_{2i}·cos(p·θ_i) - x_{2i+1}·sin(p·θ_i),
                           x_{2i}·sin(p·θ_i) + x_{2i+1}·cos(p·θ_i)]
```

其中 p 是位置索引。

#### 4.2.4 RoPE 完整实现

```python
def precompute_freqs_cis(d_model, max_len, theta=10000.0):
    """
    预计算旋转频率
    
    参数:
        d_model: 模型维度
        max_len: 最大序列长度
        theta: 基础频率（默认 10000）
    
    返回:
        freqs_cis: (max_len, d_model//2) —— 复数形式的旋转频率
    """
    # 计算频率 θ_i
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[:(d_model//2)].float() / d_model))
    
    # 生成位置索引
    t = torch.arange(max_len)
    
    # 计算旋转角度：p × θ_i
    freqs = t[:, None] * freqs[None, :]  # (max_len, d_model//2)
    
    # 转换为复数形式（用于加速计算）
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # polar(mag, angle) 创建复数：mag × e^(i×angle)
    
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    应用 RoPE 到 Q 和 K
    
    参数:
        xq, xk: (B, H, S, d_k) —— Q 和 K
        freqs_cis: (S, d_k//2) —— 预计算的旋转频率
    
    返回:
        旋转后的 Q 和 K
    """
    seq_len = xq.shape[2]
    
    # 取当前序列长度的频率
    freqs_cis = freqs_cis[:seq_len]
    
    # 将 Q, K 重塑为复数形式
    xq_complex = torch.view_as_complex(xq.float())
    xk_complex = torch.view_as_complex(xk.float())
    # 形状: (B, H, S, d_k//2)
    
    # 应用旋转：乘以复数频率
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_k//2)
    xq_rotated = torch.view_as_real(xq_complex * freqs_cis)
    xk_rotated = torch.view_as_real(xk_complex * freqs_cis)
    
    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)


class RoPEMultiHeadAttention(nn.Module):
    """
    使用 RoPE 的多头注意力
    """
    
    def __init__(self, d_model, num_heads, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # 预计算旋转频率
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(d_model, max_len)
        )
    
    def forward(self, x):
        """
        使用 RoPE 的自注意力
        
        参数:
            x: (B, S, d_model)
        """
        B, S, D = x.shape
        
        # 1. 线性投影
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用 RoPE
        Q, K = apply_rotary_emb(Q, K, self.freqs_cis)
        
        # 3. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        
        # 4. 合并多头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_O(output)
        
        return output
```

#### 4.2.5 RoPE 的独特优势

| 特性 | 绝对位置编码 | RoPE |
|------|-------------|------|
| 相对位置 | 需要额外学习 | 天然支持（旋转减法） |
| 外推性 | 差（训练外长度不稳定） | 好（外推能力强） |
| 参数量 | 固定（可学习时需参数） | 零（完全解析式） |
| 计算复杂度 | O(1) 加法 | O(d_k) 旋转 |

**外推性**：RoPE 最强大的能力之一。

假设我们在 2048 长度上训练，但在 4096 长度上推理：

- 绝对编码：没有见过位置 3000+ 的编码，性能急剧下降
- RoPE：旋转角度是连续的，位置 3000 的编码可以通过插值很好地估计

这也是为什么 LLaMA、Mistral 等现代模型都采用 RoPE。

#### 4.2.6 修炼心得：从固定到流动

RoPE 让位置信息变得"流动"：

- 绝对编码：每个位置是固定的点
- RoPE：位置信息嵌入在相对旋转中

用修炼比喻：
- 绝对编码：记住"北边第 3 个穴位"
- RoPE：记住"从这个穴位逆时针旋转 120 度"

后者更具泛化能力——无论你在身体的哪个位置开始，旋转关系都成立。

---

### 4.3 GQA（分组查询注意力） — 计算效率的极致

#### 4.3.1 MHA 与 MQA 的权衡

回顾标准的多头注意力（MHA）：

```
32 个注意力头（num_heads=32）:
  - 32 个独立的 Q 头
  - 32 个独立的 K 头
  - 32 个独立的 V 头
  - 每个头: d_k = d_model / 32 = 4096 / 32 = 128
```

**问题**：每个头都要计算自己的 K 和 V，这导致：

1. **计算量**：需要计算 32 套 K 和 V 的投影
2. **内存占用**：需要存储 32 套 K 和 V 的缓存（用于推理时的 KV Cache）

**多头查询注意力（Multi-Query Attention, MQA）** 的解决方案：

```
32 个 Q 头，但只有 1 个 K 和 V：
  - 32 个独立的 Q 头
  - 1 个共享的 K（广播到所有 Q）
  - 1 个共享的 V（广播到所有 Q）
```

优势：
- KV Cache 内存减少 32 倍！
- 计算量显著降低（只计算一次 K, V 投影）

劣势：
- **性能损失**：共享 K/V 限制了表达能力

#### 4.3.2 GQA：平衡的艺术

**分组查询注意力（Grouped-Query Attention, GQA）**（2023）在 MHA 和 MQA 之间找到了平衡点：

```
32 个 Q 头，分成 8 组，每组 4 个 Q 共享一个 K/V：
  - 32 个 Q 头（num_heads=32）
  - 8 个 K 头（num_kv_heads=8）
  - 8 个 V 头（num_kv_heads=8）
  - 每组：4 个 Q → 1 个 K/V
```

用公式表达：

```
groups = num_heads / num_kv_heads = 32 / 8 = 4

每个 Q 头 i 对应的 K/V 头：
  kv_head_index = i // groups = i // 4
```

**权衡表**：

| 架构 | Q 头数 | K/V 头数 | KV Cache 大小 | 表达能力 |
|------|--------|-----------|---------------|----------|
| MHA | 32 | 32 | 32× | 完整（最强） |
| GQA (8 groups) | 32 | 8 | 0.25× | 良好（95% 性能） |
| MQA | 32 | 1 | 0.03× | 较弱（90% 性能） |

**GQA 在几乎不损失性能的前提下，将 KV Cache 内存减少 4 倍！**

#### 4.3.3 GQA 完整实现

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    
    参数:
        d_model: 模型维度
        num_heads: Q 的头数
        num_kv_heads: K/V 的头数（必须是 num_heads 的约数）
    """
    
    def __init__(self, d_model, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        
        assert num_heads % num_kv_heads == 0, \
            "num_heads 必须能被 num_kv_heads 整除"
        
        self.groups = num_heads // num_kv_heads  # 每组 Q 的数量
        
        # Q, K, V 的线性投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        GQA 前向传播
        
        参数:
            x: (B, S, d_model)
        """
        B, S, D = x.shape
        
        # 1. 线性投影
        Q = self.W_Q(x)  # (B, S, d_model) = (B, S, num_heads * d_k)
        K = self.W_K(x)  # (B, S, num_kv_heads * d_k)
        V = self.W_V(x)  # (B, S, num_kv_heads * d_k)
        
        # 2. 重塑并转置
        Q = Q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, S, d_k)
        K = K.view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        # (B, num_kv_heads, S, d_k)
        V = V.view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        # (B, num_kv_heads, S, d_k)
        
        # 3. 将 K 和 V 广播到 Q 的头数
        # 每个 K/V 头要服务 self.groups 个 Q 头
        K = K.repeat_interleave(self.groups, dim=1)
        # (B, num_heads, S, d_k)
        V = V.repeat_interleave(self.groups, dim=1)
        # (B, num_heads, S, d_k)
        
        # 4. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (B, num_heads, S, S)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        # (B, num_heads, S, d_k)
        
        # 5. 合并多头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_O(output)
        
        return output
```

**repeat_interleave 的作用**：

假设 num_heads=4, num_kv_heads=2, groups=2：

```
K 头索引: [0, 1]
广播后:    [0, 0, 1, 1]

每个 K 头（索引 i）广播到 Q 头：
  - Q 头 0, 1 使用 K 头 0
  - Q 头 2, 3 使用 K 头 1
```

#### 4.3.4 实际应用：KV Cache 优化

在自回归生成（如 GPT）中，KV Cache 的内存占用是主要瓶颈：

**传统 MHA**（seq_len=4096, batch=1）：

```
每层的 KV Cache:
  - K: (1, num_heads=32, 4096, d_k=128) = 16,777,216 FP16 = 32 MB
  - V: (1, num_heads=32, 4096, d_k=128) = 16,777,216 FP16 = 32 MB
  - 总计: 64 MB/层

32 层模型: 64 MB × 32 = 2,048 MB = 2 GB
```

**GQA**（num_kv_heads=8）：

```
每层的 KV Cache:
  - K: (1, num_kv_heads=8, 4096, d_k=128) = 4,194,304 FP16 = 8 MB
  - V: (1, num_kv_heads=8, 4096, d_k=128) = 4,194,304 FP16 = 8 MB
  - 总计: 16 MB/层

32 层模型: 16 MB × 32 = 512 MB
```

**内存节省：75%！**

这使得在消费级 GPU（如 RTX 4090 的 24GB 显存）上推理长序列（4096+）成为可能。

#### 4.3.5 修炼心得：共享与表达的平衡

GQA 的核心思想：

**不是所有 Q 头都需要独立的 K 和 V。**

用修炼比喻：
- MHA：每个弟子都有独立的灵力源
- MQA：所有弟子共享一个灵力源（资源节省，但能力受限）
- GQA：弟子分组，每 4 个弟子共享一个灵力源（平衡）

现代大模型（LLaMA 2/3、Mistral、Qwen）都采用 GQA，因为它在不损失性能的前提下，显著提升了推理效率。

---

### 4.6 Transformer 训练技巧与稳定性优化

训练大模型需要精妙的技巧和策略，这就像修炼者在突破境界时需要运用各种秘术来稳固根基、加速修炼。

#### 4.6.1 学习率 Warmup

**问题**：在训练初期，如果直接使用较大的学习率，可能会导致：
- 模型参数剧烈震荡
- 梯度爆炸，训练不稳定
- 损失值变成 NaN

**Warmup 策略**：在训练的前几个 epoch 或 steps，线性地从很小的学习率增加到目标学习率。

```python
def get_lr_schedule(optimizer, warmup_steps, total_steps, max_lr):
    """
    带有 warmup 的余弦退火学习率调度
    
    参数:
        optimizer: 优化器
        warmup_steps: warmup 步数
        total_steps: 总训练步数
        max_lr: 最大学习率
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup 阶段：线性增长
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦退火阶段
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**修炼比喻**：Warmup 就像修炼者在初学功法时，先慢运行气，逐渐加大灵力流量，避免经脉受损。

#### 4.6.2 梯度累积（Gradient Accumulation）

**问题**：GPU 显存有限，无法使用大的 batch size。

**梯度累积**：模拟更大的 batch size 的效果。

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, criterion, 
                                 accumulation_steps=4, device='cuda'):
    """
    梯度累积训练
    
    参数:
        accumulation_steps: 累积步数（相当于 batch_size 扩大 4 倍）
    """
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps  # 除以累积步数
        
        # 反向传播（梯度累积）
        loss.backward()
        
        # 每 accumulation_steps 步更新一次
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

**修炼心得**：就像修炼者虽然每次只能吸收少量的天地灵气，但通过多次累积，最终也能达到一次性吸收大量灵气的效果。

#### 4.6.3 混合精度训练（Mixed Precision Training）

**问题**：FP32 训练虽然稳定，但：
- 显存占用大（4 字节/参数）
- 计算速度慢

**混合精度训练**：同时使用 FP16 和 FP32：
- 前向传播：FP16（节省显存，加速计算）
- 梯度缩放：防止 FP16 下溢
- 优化器状态：FP32（保持稳定性）

```python
from torch.cuda.amp import GradScaler, autocast

def train_mixed_precision(model, dataloader, optimizer, criterion):
    """
    混合精度训练
    
    优势：
    - 显存节省约 50%
    - 训练速度提升 2-3 倍
    """
    scaler = GradScaler()  # 梯度缩放器
    
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 自动混合精度上下文
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 反向传播（自动缩放梯度）
        scaler.scale(loss).backward()
        
        # 梯度裁剪（防止梯度爆炸）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数（自动反缩放）
        scaler.step(optimizer)
        scaler.update()
```

**性能对比**：

| 方法 | 显存占用 | 训练速度 | 数值稳定性 |
|------|----------|----------|------------|
| FP32 | 100% | 1x | 最稳定 |
| FP16 | 50% | 2-3x | 不稳定（下溢） |
| 混合精度 | 50% | 2-3x | 稳定（通过梯度缩放） |

#### 4.6.4 长上下文注意力变体

对于超长序列（如 32k+），标准 Transformer 的 O(S²) 复杂度仍然太高。

**稀疏注意力（Sparse Attention）**：只计算部分位置的注意力。

```python
class SparseAttention(nn.Module):
    """
    稀疏注意力 —— 只关注局部和全局关键位置
    
    策略：
    1. 局部窗口注意力：每个 token 只关注周围 w 个 token
    2. 全局 token：随机选择或学习一些全局 token
    """
    
    def __init__(self, d_model, num_heads, window_size=128, num_global_tokens=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        x: (B, S, d_model)
        """
        B, S, D = x.shape
        
        # 线性投影
        Q = self.W_Q(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        # 局部窗口注意力
        # 实现略...
        
        # 全局 token 注意力
        # 实现略...
        
        # 合并局部和全局注意力
        # 实现略...
        
        return output
```

**线性注意力（Linear Attention）**：将注意力复杂度从 O(S²) 降低到 O(S)。

```python
class LinearAttention(nn.Module):
    """
    线性注意力 —— O(S) 复杂度
    
    核心思想：使用 Kernel 技巧避免显式计算 S×S 矩阵
    
    公式：
        Attention(Q, K, V) ≈ softmax(Q @ K^T / sqrt(d_k)) @ V
                               ≈ φ(Q) @ φ(K)^T @ V / φ(K)^T @ 1
        
        其中 φ(x) = ELU(x) + 1
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        x: (B, S, d_model)
        """
        B, S, D = x.shape
        
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        
        # Kernel 特征映射
        Q_hat = F.elu(Q) + 1  # (B, H, S, d_k)
        K_hat = F.elu(K) + 1  # (B, H, S, d_k)
        
        # 线性注意力
        KV = torch.einsum('bhse,bhsd->bhed', K_hat, V)  # (B, H, d_k, d_k)
        Z = torch.einsum('bhse,bhed->bhsd', Q_hat, KV)  # (B, H, S, d_k)
        
        # 归一化
        K_sum = K_hat.sum(dim=2, keepdim=True)  # (B, H, 1, d_k)
        output = Z / (K_sum + 1e-6)  # (B, H, S, d_k)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.W_O(output)
        
        return output
```

**复杂度对比**：

| 方法 | 复杂度 | 适用场景 |
|------|--------|----------|
| 标准注意力 | O(S²) | S < 2048 |
| 稀疏注意力 | O(S × √S) | 2048 < S < 32k |
| 线性注意力 | O(S × d_k) | S > 32k |

### 4.7 LoRA（低秩自适应）— 轻量级微调术

#### 4.4.1 全量微调的沉重负担

假设你想在 LLaMA-7B 上微调一个特定任务：

**全量微调（Full Fine-tuning）**：

```
总参数量: 7,000,000,000
需要微调: 7,000,000,000 (100%)
存储占用: 7B FP16 = 14 GB
显存需求: ~40 GB (包含梯度和优化器状态)
```

**问题**：
- 需要巨大的计算资源
- 每个任务都需要保存一个完整的模型副本
- 无法在消费级 GPU 上运行

#### 4.4.2 LoRA 的核心洞察

**低秩自适应（Low-Rank Adaptation, LoRA）**（2021）的洞察：

**预训练模型的权重已经编码了大量通用知识。**
**微调时，我们只需要学习任务特定的增量。**

数学上：

```
原始权重: W ∈ R^(d_in × d_out)
微调权重: W' = W + ΔW

关键假设：ΔW 是低秩的
即: ΔW = B @ A，其中 B ∈ R^(d_in × r), A ∈ R^(r × d_out)
```

其中 `r` 是低秩维度（通常 r << d_in, d_out）：

```
典型配置:
  - d_in = 4096
  - d_out = 4096
  - r = 8 或 16
  - 参数量: 4096 × 8 + 8 × 4096 = 65,536
  - 相比全量: 65,536 / (4096 × 4096) = 0.0039 (0.39%)
```

**参数量减少：250 倍！**

#### 4.4.3 LoRA 完整实现

```python
class LoRALinear(nn.Module):
    """
    LoRA 线性层
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        rank: 低秩维度 r（通常 8-64）
        alpha: 缩放因子（通常等于 rank）
        dropout: dropout 概率
    """
    
    def __init__(self, in_features, out_features, rank=8, alpha=8, dropout=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # 冻结预训练权重
        
        # LoRA 低秩分解
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化：A 使用高斯分布，B 使用零
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scaling = self.alpha / self.rank
    
    def forward(self, x):
        """
        前向传播
        
        原始输出 + LoRA 增量
        """
        # 原始输出（冻结）
        base_output = self.linear(x)
        
        # LoRA 增量（可训练）
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        lora_output = lora_output * self.scaling
        
        # 合并
        return base_output + lora_output


def inject_lora_to_model(model, rank=8, alpha=8, target_modules=["q_proj", "v_proj"]):
    """
    将 LoRA 注入到现有模型中
    
    参数:
        model: 预训练模型（如 LLaMA）
        rank: 低秩维度
        alpha: 缩放因子
        target_modules: 需要注入 LoRA 的模块名称
    
    返回:
        可训练参数总数
    """
    trainable_params = 0
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 获取原始层配置
                in_features = module.in_features
                out_features = module.out_features
                
                # 创建 LoRA 层
                lora_layer = LoRALinear(
                    in_features, out_features, rank, alpha
                )
                
                # 复制原始权重
                lora_layer.linear.weight.data = module.weight.data.clone()
                
                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part) if part else parent
                
                setattr(parent, child_name, lora_layer)
                
                # 统计可训练参数
                trainable_params += (
                    lora_layer.lora_A.weight.numel() +
                    lora_layer.lora_B.weight.numel()
                )
    
    print(f"注入 LoRA 完成！可训练参数: {trainable_params:,}")
    return trainable_params
```

**使用示例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 注入 LoRA
trainable_params = inject_lora_to_model(
    model,
    rank=16,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 3. 冻结非 LoRA 参数
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

# 4. 训练（只训练 LoRA 参数）
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# ... 正常训练流程 ...
```

#### 4.4.4 QLoRA：结合量化

**QLoRA（Quantized LoRA）**（2023）将 LoRA 与 4-bit 量化结合：

```
1. 将预训练模型量化到 4-bit（FP32 → INT4）
   - 模型大小: 7B FP16 = 14 GB → 7B INT4 = 3.5 GB
   - 内存需求: 40 GB → 8 GB
   
2. 在量化模型上添加 LoRA
   - 可训练参数: ~100M (rank=16)
   - 额外显存: ~200 MB

3. 训练和推理
   - 反向传播时，将 4-bit 权重解量化为 16-bit
   - 前向传播使用量化权重加速
```

**显存需求对比**（LLaMA-7B，seq_len=4096, batch=1）：

| 方法 | 模型大小 | 训练显存 | 推理显存 |
|------|---------|----------|---------|
| 全量 FP16 | 14 GB | 40 GB | 16 GB |
| LoRA FP16 | 14 GB | 24 GB | 16 GB |
| QLoRA | 3.5 GB | **12 GB** | **6 GB** |

**在 RTX 3090（24GB）上，只能用 QLoRA！**

#### 4.4.5 LoRA 的独特优势

| 特性 | 全量微调 | LoRA |
|------|----------|------|
| 可训练参数 | 100% | 0.1-1% |
| 存储占用 | 每任务 14 GB | 每任务 100 MB |
| 训练显存 | 40 GB | 12 GB (QLoRA) |
| 任务切换 | 需要重新加载 | 快速切换 Adapter |
| 知识遗忘 | 风险高 | 风险低（保留预训练知识） |

**应用场景**：

1. **多任务适配**：为每个任务训练独立的 LoRA adapter
2. **个性化**：为每个用户训练个性化 adapter
3. **领域适配**：快速适配到医疗、法律等专业领域
4. **低资源环境**：在消费级 GPU 上微调大模型

#### 4.4.6 修炼心得：增量与共享

LoRA 的核心哲学：

**不要重新创造，而是在现有基础上添加轻量级的增量。**

用修炼比喻：
- 全量微调：推翻原有功法，从头修炼
- LoRA：在原有功法上添加轻量级招式

增量式学习（Incremental Learning）——这正是现代 AI 修炼的核心智慧。

---

### 4.5 三大前沿技术对比总结

| 技术 | 核心问题 | 解决方案 | 性能影响 | 计算影响 |
|------|---------|---------|----------|----------|
| Flash Attention | 内存瓶颈 O(S²) | 分块 + 融合核 + 在线 softmax | 无损失 | 快 2-4x |
| RoPE | 绝对位置编码的外推性差 | 旋转相对位置编码 | 无损失（更好） | 增加少许开销 |
| GQA | KV Cache 内存瓶颈 | 多 Q 共享 K/V | 损失 5-10% | 快 2-3x |
| LoRA | 微调成本高 | 低秩自适应 | 损失 1-5% | 快 10x+ |

**组合使用**：

现代大模型（LLaMA 3、Mistral、Qwen）通常组合使用：

```
LLaMA-3-8B:
  - Flash Attention 2: 内存高效
  - RoPE: 相对位置感知
  - GQA (groups=8): 推理加速
  - 训练时可添加 LoRA 适配特定任务
```

---

## 第五章：道法评估 — 模型调试与修炼进阶

> *"能施展出斗技不难，难的是知道斗技为何有效，又为何失效。*
> *修炼者不仅要会'用'道法，更要会'观'道法，'悟'道法。*
> *评估与调试，就是修炼中的内观与自省。*"*
> *——《焚诀》第二卷 进阶篇*

### 5.1 模型评估指标体系

#### 5.1.1 分类问题评估

**混淆矩阵（Confusion Matrix）**是所有分类指标的基础：

```
                    预测
                  正类    负类
实际  正类     TP (真正)  FN (假负)
      负类     FP (假正)  TN (真负)
```

**核心指标**：

```python
def compute_metrics(y_true, y_pred, average='macro'):
    """
    计算分类指标
    
    参数:
        y_true: (N,) —— 真实标签
        y_pred: (N,) —— 预测标签
        average: 'macro'（宏平均）或 'micro'（微平均）
    
    返回:
        metrics: 包含 accuracy, precision, recall, f1 的字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics
```

**Precision vs Recall 的权衡**：

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def precision_recall_analysis(y_true, y_scores):
    """
    Precision-Recall 曲线分析
    
    适用于二分类或多分类（需要 one-vs-all）
    
    返回:
        precisions: 不同阈值下的精确率
        recalls: 不同阈值下的召回率
        thresholds: 对应的阈值
        f1_scores: 不同阈值下的 F1 分数
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # 计算 F1 分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # 找到最佳 F1 分数对应的阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return precisions, recalls, thresholds, f1_scores, best_threshold, best_f1
```

**ROC 曲线和 AUC**：

```python
from sklearn.metrics import roc_curve, auc

def roc_analysis(y_true, y_scores):
    """
    ROC 曲线和 AUC 分析
    
    ROC 曲线展示了不同阈值下真正率（TPR）与假正率（FPR）的关系
    
    参数:
        y_true: (N,) —— 真实标签（0 或 1）
        y_scores: (N,) —— 预测概率
    
    返回:
        fpr: 假正率
        tpr: 真正率
        thresholds: 阈值
        roc_auc: AUC 值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc
```

**AUC 的意义**：
- AUC = 1：完美分类
- AUC = 0.5：随机猜测
- AUC < 0.5：比随机猜测还差（可能是标签搞反了）

#### 5.1.2 回归问题评估

```python
def regression_metrics(y_true, y_pred):
    """
    回归问题评估指标
    
    参数:
        y_true: (N,) —— 真实值
        y_pred: (N,) —— 预测值
    
    返回:
        metrics: 包含 mae, mse, rmse, r2 的字典
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics
```

**指标解读**：
- **MAE（平均绝对误差）**：误差的平均值，单位与目标变量相同
- **RMSE（均方根误差）**：对大误差更敏感
- **R²（决定系数）**：模型解释的方差比例（0-1，越接近 1 越好）

### 5.2 学习曲线分析

**学习曲线**是诊断模型是否过拟合或欠拟合的重要工具：

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X_train, y_train, X_val, y_val, 
                      metric='loss', figsize=(10, 6)):
    """
    绘制学习曲线
    
    参数:
        model: 模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        metric: 监控的指标 ('loss', 'accuracy', 'f1' 等)
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    train_history = []
    val_history = []
    
    # 训练模型并记录指标
    num_epochs = model.num_epochs
    for epoch in range(num_epochs):
        model.train()
        # 训练一个 epoch
        train_loss = train_one_epoch(model, X_train, y_train)
        
        # 评估
        model.eval()
        train_metric = evaluate(model, X_train, y_train, metric)
        val_metric = evaluate(model, X_val, y_val, metric)
        
        train_history.append(train_metric)
        val_history.append(val_metric)
    
    # 绘制曲线
    epochs = range(1, num_epochs + 1)
    ax.plot(epochs, train_history, 'b-', label=f'Train {metric}', linewidth=2)
    ax.plot(epochs, val_history, 'r-', label=f'Validation {metric}', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax
```

**学习曲线诊断**：

```
情况 1: 欠拟合（High Bias）
训练集: 误差高
验证集: 误差高
→ 解决：增加模型复杂度、减少正则化、增加特征

情况 2: 过拟合（High Variance）
训练集: 误差低
验证集: 误差高
→ 解决：增加数据、正则化、减少模型复杂度、早停

情况 3: 合适
训练集: 误差适中
验证集: 误差与训练集接近
→ 继续训练，可能需要微调
```

### 5.3 错误分析（Error Analysis）

错误分析是改进模型最有效的方法之一：

```python
def error_analysis(model, X_test, y_test, num_samples=20):
    """
    错误分析 —— 找出模型预测错误的样本
    
    参数:
        model: 模型
        X_test: 测试数据
        y_test: 测试标签
        num_samples: 展示的错误样本数量
    
    返回:
        error_samples: 错误样本列表
    """
    model.eval()
    
    # 获取预测
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = y_pred.argmax(dim=-1)
    
    # 找出预测错误的样本
    mask = y_pred_classes != y_test
    error_indices = torch.where(mask)[0]
    
    # 获取错误样本
    error_samples = []
    for idx in error_indices[:num_samples]:
        error_samples.append({
            'index': idx.item(),
            'true_label': y_test[idx].item(),
            'predicted_label': y_pred_classes[idx].item(),
            'confidence': torch.max(torch.softmax(y_pred[idx], dim=-1)).item(),
            'input': X_test[idx]  # 根据需要修改
        })
    
    return error_samples
```

**错误分析步骤**：

1. **收集错误样本**：找出模型预测错误的样本
2. **分类错误类型**：
   - 硬错误：完全错误的预测
   - 软错误：正确但置信度低
3. **寻找共同特征**：
   - 错误样本是否有共同特点？（如特定类别、噪声）
   - 是否与训练数据分布不同？
4. **提出改进方案**：
   - 数据增强
   - 特征工程
   - 模型架构调整

### 5.4 模型调试技巧

#### 5.4.1 梯度检查（Gradient Checking）

当实现自定义层或复杂的反向传播时，梯度检查是验证实现是否正确的金标准：

```python
def gradient_check(model, X, y, epsilon=1e-7):
    """
    数值梯度检查
    
    通过数值计算近似梯度，并与自动计算梯度对比
    
    参数:
        model: 模型
        X: (B, ...) 输入数据
        y: (B, ...) 目标数据
        epsilon: 扰动量
    
    返回:
        relative_error: 相对误差（应该 < 1e-7）
    """
    model.eval()
    
    # 计算自动计算的梯度
    loss = model.compute_loss(X, y)
    loss.backward()
    
    # 获取第一个参数的梯度
    param = next(model.parameters())
    grad_auto = param.grad.clone().detach()
    
    # 数值梯度
    grad_numeric = torch.zeros_like(param)
    
    # 对每个参数施加微小扰动
    it = np.nditer(param.numpy(), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        
        # f(x + epsilon)
        param[idx] += epsilon
        loss_plus = model.compute_loss(X, y)
        
        # f(x - epsilon)
        param[idx] -= 2 * epsilon
        loss_minus = model.compute_loss(X, y)
        
        # 数值梯度
        grad_numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # 恢复参数
        param[idx] += epsilon
        
        it.iternext()
    
    # 计算相对误差
    numerator = torch.norm(grad_auto - grad_numeric)
    denominator = torch.norm(grad_auto) + torch.norm(grad_numeric)
    relative_error = numerator / denominator
    
    return relative_error.item()
```

**相对误差标准**：
- < 1e-7：正确
- 1e-5 ~ 1e-7：可以接受
- > 1e-3：可能有问题

#### 5.4.2 模型可视化

对于 CNN，可视化卷积核和激活图有助于理解模型学到了什么：

```python
def visualize_filters(model, layer_name, num_filters=16):
    """
    可视化卷积核
    
    参数:
        model: 模型
        layer_name: 层名称
        num_filters: 可视化的卷积核数量
    """
    # 获取指定层的权重
    for name, param in model.named_parameters():
        if name == layer_name:
            weights = param.data.cpu().numpy()
            break
    
    # 归一化到 [0, 1]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # 绘制
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_filters, weights.shape[0])):
        ax = axes[i]
        if weights.shape[1] == 3:  # RGB
            ax.imshow(weights[i].transpose(1, 2, 0))
        else:  # 灰度
            ax.imshow(weights[i][0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i}')
    
    plt.tight_layout()
    return fig


def visualize_activations(model, image, layer_name):
    """
    可视化特征图（激活）
    
    参数:
        model: 模型
        image: 输入图像
        layer_name: 层名称
    """
    model.eval()
    
    # 注册 hook 来获取中间层输出
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册 hook
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(name))
    
    # 前向传播
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    # 获取激活
    act = activation[layer_name][0].cpu().numpy()  # (C, H, W)
    
    # 绘制
    num_channels = min(16, act.shape[0])
    cols = 4
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(act[i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Channel {i}')
    
    # 隐藏多余的子图
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
```

### 5.5 修炼总结与境界突破条件（更新）

在进入第三卷之前，确保你能做到以下几点：

**基础斗技**：
- [ ] **能画出 CNN 的完整架构图**，标注每一层的输出维度
- [ ] **能解释 LSTM 三个门的作用**，以及为什么它比 RNN 更擅长长期记忆
- [ ] **能手写 Self-Attention 的完整公式**，包括 Q/K/V 投影、缩放、softmax、加权求和
- [ ] **能追踪张量在 Transformer block 中的完整维度变化**
- [ ] **能从零实现一个可运行的 Transformer block**（不看任何参考代码）
- [ ] **能解释 Encoder 和 Decoder 的区别**，以及因果遮罩的作用
- [ ] **能说出 Transformer 相比 RNN 的三个核心优势**

**进阶斗技**：
- [ ] **能解释传统注意力的内存瓶颈**，并计算 seq_len=4096 时的显存占用
- [ ] **能手写在线 softmax 的简化版**，理解其数学原理
- [ ] **能说明 Flash Attention 的三个核心技术**（分块、融合核、在线 softmax）
- [ ] **能手写 RoPE 的旋转公式**，解释复数旋转如何编码相对位置
- [ ] **能对比绝对位置编码和 RoPE** 的外推能力差异
- [ ] **能手写 GQA 的广播逻辑**（repeat_interleave）
- [ ] **能计算 MHA、GQA、MQA 的 KV Cache 内存占用**差异
- [ ] **能手写 LoRA 线性层的完整实现**，包括初始化策略
- [ ] **能计算 LoRA 相比全量微调的参数量占比**（d_in=4096, d_out=4096, r=16）
- [ ] **能解释 LoRA 为何能有效微调**（ΔW 低秩假设）

**评估与调试**（新增）：
- [ ] **能手动计算混淆矩阵、Precision、Recall、F1**
- [ ] **能绘制并解释学习曲线**，诊断过拟合/欠拟合
- [ ] **能实现梯度检查函数**，验证自定义层的反向传播
- [ ] **能进行错误分析**，找出模型预测错误的共同特征
- [ ] **能实现 BatchNorm 的前向和反向传播**（包括移动平均）
- [ ] **能实现 Beam Search 算法**，解释为何优于贪婪解码
- [ ] **能实现深度可分离卷积**，并计算其相对于标准卷积的参数量节省
- [ ] **能解释混合精度训练的原理**，包括梯度缩放的必要性

如果以上你都能做到，恭喜你，你已经突破到**斗师**之境，并且掌握了天道法则的进阶形态！

---

## 实战案例：完整图像分类项目 — 从零到部署

> *"纸上得来终觉浅，绝知此事要躬行。*
> *修炼者只有在实战中才能真正领悟斗技的精髓。"*
> *——《焚诀》实战篇*

### 案例目标

在本案例中，我们将**从零开始**完成一个完整的图像分类项目，涵盖以下全流程：

1. **数据准备**：自定义 Dataset、数据增强、Dataloader
2. **模型构建**：ResNet-18 架构实现
3. **训练循环**：完整的训练流程（带验证、早停、学习率调度）
4. **模型保存与加载**：Checkpointing
5. **推理与可视化**：预测、混淆矩阵、学习曲线
6. **部署准备**：模型导出、ONNX 转换

### 6.1 项目结构

```
image_classification_project/
├── data/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── val/
│       ├── cat/
│       └── dog/
├── models/
│   ├── __init__.py
│   └── resnet.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── predict.py
└── config.yaml
```

### 6.2 配置文件（config.yaml）

```yaml
# 训练配置
data:
  data_dir: "./data"
  image_size: 224
  batch_size: 32
  num_workers: 4
  pin_memory: true

model:
  num_classes: 2
  dropout: 0.5

training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 5
  patience: 10  # 早停耐心

augmentation:
  horizontal_flip_prob: 0.5
  rotation_degrees: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

checkpoints:
  save_dir: "./checkpoints"
  save_freq: 5  # 每 5 个 epoch 保存一次
```

### 6.3 数据工具（utils/data_utils.py）

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import yaml

class CustomDataset(Dataset):
    """
    自定义数据集
    
    参数:
        root_dir: 数据根目录
        transform: 数据增强/预处理
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 收集所有图像路径和标签
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(config):
    """
    创建训练和验证 DataLoader
    
    返回:
        train_loader, val_loader, class_names
    """
    # 读取配置
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg['data']['image_size']),
        transforms.RandomHorizontalFlip(p=cfg['augmentation']['horizontal_flip_prob']),
        transforms.RandomRotation(cfg['augmentation']['rotation_degrees']),
        transforms.ColorJitter(**cfg['augmentation']['color_jitter']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证数据预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CustomDataset(
        os.path.join(cfg['data']['data_dir'], 'train'),
        transform=train_transform
    )
    
    val_dataset = CustomDataset(
        os.path.join(cfg['data']['data_dir'], 'val'),
        transform=val_transform
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory']
    )
    
    return train_loader, val_loader, train_dataset.classes
```

### 6.4 ResNet 模型（models/resnet.py）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet 基础块
    
    结构：
        conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+) -> relu
              |_______________________________|
                   Skip Connection
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: 如果维度或步长变化，需要 1x1 卷积匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 架构
    
    结构：
        conv1 -> bn1 -> relu -> maxpool
        -> layer1 (2 blocks)
        -> layer2 (2 blocks, stride=2)
        -> layer3 (2 blocks, stride=2)
        -> layer4 (2 blocks, stride=2)
        -> avgpool -> fc
    """
    
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        创建一个 ResNet 层（包含多个 BasicBlock）
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
```

### 6.5 训练脚本（train.py）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
import os
from datetime import datetime

from utils.data_utils import get_data_loaders
from models.resnet import ResNet18
from utils.metrics import compute_metrics
from utils.checkpointing import save_checkpoint, load_checkpoint

# 早停类（前面已定义）
class EarlyStopping:
    # ... (略，见前面)
    pass


class AverageMeter:
    """
    计算和存储平均值和当前值
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch
    
    返回:
        average_loss, average_accuracy
    """
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / targets.size(0)
        
        # 更新统计
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(accuracy, inputs.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, dataloader, criterion, device):
    """
    验证模型
    
    返回:
        average_loss, average_accuracy, metrics_dict
    """
    model.eval()
    
    losses = AverageMeter()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 记录预测
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 更新统计
            losses.update(loss.item(), inputs.size(0))
    
    # 计算指标
    metrics = compute_metrics(all_targets, all_predictions)
    
    return losses.avg, metrics['accuracy'], metrics


def train(config_path='config.yaml'):
    """
    完整训练流程
    """
    # 读取配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据加载
    train_loader, val_loader, class_names = get_data_loaders(config_path)
    print(f'Classes: {class_names}')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # 模型
    model = ResNet18(
        num_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # 学习率调度器（带 warmup）
    scheduler = get_lr_schedule(
        optimizer,
        warmup_epochs=cfg['training']['warmup_epochs'],
        total_epochs=cfg['training']['num_epochs'],
        max_lr=cfg['training']['learning_rate']
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=cfg['training']['patience'],
        verbose=True,
        mode='max'
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 训练循环
    best_val_acc = 0.
    for epoch in range(cfg['training']['num_epochs']):
        print(f'\nEpoch [{epoch+1}/{cfg["training"]["num_epochs"]}]')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f} | '
              f'Val Recall: {val_metrics["recall"]:.4f} | '
              f'Val F1: {val_metrics["f1"]:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': cfg
            }, os.path.join(cfg['checkpoints']['save_dir'], 'best_model.pth'))
            print(f'✓ Saved best model (val_acc: {val_acc:.4f})')
        
        # 定期保存 checkpoint
        if (epoch + 1) % cfg['checkpoints']['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': cfg,
                'history': history
            }, os.path.join(cfg['checkpoints']['save_dir'], 
                         f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 早停检查
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print('Early stopping triggered!')
            # 加载最佳模型
            model = early_stopping.load_best_model(model)
            break
    
    print(f'\nTraining completed! Best val acc: {best_val_acc:.4f}')
    
    return model, history


def get_lr_schedule(optimizer, warmup_epochs, total_epochs, max_lr):
    """
    带有 warmup 的余弦退火学习率调度
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == '__main__':
    import numpy as np
    
    # 训练
    model, history = train()
    
    # 绘制学习曲线
    from utils.visualization import plot_learning_curves
    plot_learning_curves(history)
```

### 6.6 Checkpoint 工具（utils/checkpointing.py）

```python
import torch
import os


def save_checkpoint(state, filename):
    """
    保存 checkpoint
    
    参数:
        state: 包含模型状态、优化器状态、epoch 等的字典
        filename: 保存路径
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f'Checkpoint saved to {filename}')


def load_checkpoint(filename, model, optimizer=None):
    """
    加载 checkpoint
    
    参数:
        filename: checkpoint 文件路径
        model: 模型实例
        optimizer: 优化器实例（可选）
    
    返回:
        start_epoch: 开始的 epoch
        best_val_acc: 最佳验证准确率
    """
    if not os.path.isfile(filename):
        print(f'No checkpoint found at {filename}')
        return 0, 0.
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_acc = checkpoint.get('val_acc', 0.)
    
    print(f'Checkpoint loaded from {filename}')
    print(f'Starting from epoch {start_epoch}, best val acc: {best_val_acc:.4f}')
    
    return start_epoch, best_val_acc
```

### 6.7 可视化工具（utils/visualization.py）

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch


def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    绘制学习曲线
    
    参数:
        history: 包含训练历史的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Learning curves saved to {save_path}')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
    plt.close()
```

### 6.8 预测脚本（predict.py）

```python
import torch
from torchvision import transforms
from PIL import Image
import yaml
import argparse

from models.resnet import ResNet18
from utils.checkpointing import load_checkpoint


def predict_image(image_path, checkpoint_path, config_path='config.yaml'):
    """
    对单张图像进行预测
    
    参数:
        image_path: 图像路径
        checkpoint_path: 模型 checkpoint 路径
        config_path: 配置文件路径
    
    返回:
        predicted_class: 预测类别
        confidence: 置信度
    """
    # 读取配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载模型
    model = ResNet18(
        num_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 返回结果
    class_names = ['cat', 'dog']  # 根据实际类别修改
    predicted_class = class_names[predicted.item()]
    
    return predicted_class, confidence.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    args = parser.parse_args()
    
    predicted_class, confidence = predict_image(args.image, args.checkpoint)
    print(f'Predicted: {predicted_class} (confidence: {confidence:.4f})')
```

### 6.9 运行项目

```bash
# 1. 准备数据
mkdir -p data/train/cat data/train/dog data/val/cat data/val/dog
# 将图像放到对应目录

# 2. 训练
python train.py --config config.yaml

# 3. 预测
python predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

### 6.10 修炼心得

通过这个完整的实战案例，你应该掌握：

1. **项目结构设计**：清晰的代码组织
2. **配置管理**：使用 YAML 管理超参数
3. **数据加载**：自定义 Dataset 和 DataLoader
4. **模型构建**：从零实现 ResNet-18
5. **训练流程**：完整的训练循环、验证、早停、学习率调度
6. **Checkpointing**：模型保存与加载
7. **可视化**：学习曲线、混淆矩阵
8. **推理部署**：单张图像预测

这是你从**理论到实践**的第一次完整突破！

---

## 第六章（续篇）：实战宝典 — 模型部署与工程实践

> *"炼出丹药只是第一步，如何将丹药炼制成丹丸、保存、运输、服用，才是真正的考验。*
> *训练模型只是开始，部署模型、监控性能、持续优化，才是修炼者必须掌握的实战宝典。"*
> *——《焚诀》实战篇*

### 6.11 模型导出与转换

**为什么需要导出？**

- PyTorch 模型只能用 PyTorch 推理
- 不同平台需要不同格式（ONNX、TensorRT、TFLite）
- 优化推理速度和内存占用

#### 6.11.1 ONNX 模型导出

**ONNX（Open Neural Network Exchange）**：统一的神经网络交换格式，支持多种框架。

**完整实现**：

```python
import torch
import torch.onnx

def export_to_onnx(model, example_input, onnx_path='model.onnx'):
    """
    导出 PyTorch 模型为 ONNX 格式
    
    参数:
        model: PyTorch 模型
        example_input: 示例输入（用于追踪）
        onnx_path: 输出路径
    """
    model.eval()
    
    # 导出 ONNX
    torch.onnx.export(
        model,                          # 模型
        example_input,                  # 示例输入
        onnx_path,                     # 输出路径
        export_params=True,             # 导出参数
        opset_version=14,               # ONNX 版本
        do_constant_folding=True,       # 常量折叠优化
        input_names=['input'],          # 输入名称
        output_names=['output'],        # 输出名称
        dynamic_axes={                  # 动态维度
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX 模型已导出到: {onnx_path}")


# 使用示例
model = ResNet18(num_classes=2)
example_input = torch.randn(1, 3, 224, 224)

export_to_onnx(model, example_input, 'resnet18.onnx')
```

**验证 ONNX 模型**：

```python
import onnx

def verify_onnx_model(onnx_path):
    """
    验证 ONNX 模型是否有效
    """
    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    
    # 检查模型
    onnx.checker.check_model(onnx_model)
    
    print("✓ ONNX 模型验证通过")
    
    # 打印模型信息
    print("\n=== 模型输入 ===")
    for input_info in onnx_model.graph.input:
        print(f"名称: {input_info.name}")
        print(f"形状: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
    
    print("\n=== 模型输出 ===")
    for output_info in onnx_model.graph.output:
        print(f"名称: {output_info.name}")
        print(f"形状: {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")


verify_onnx_model('resnet18.onnx')
```

#### 6.11.2 ONNX Runtime 推理

**完整实现**：

```python
import onnxruntime as ort
import numpy as np

class ONNXInference:
    """
    ONNX Runtime 推理引擎
    
    优势:
        - 跨平台（Windows/Linux/macOS）
        - 支持多种硬件（CPU/GPU/NPU）
        - 自动优化（图优化、算子融合）
    """
    
    def __init__(self, onnx_path, use_gpu=False):
        """
        参数:
            onnx_path: ONNX 模型路径
            use_gpu: 是否使用 GPU
        """
        # 选择执行提供者
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # 创建会话
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX Runtime 已初始化")
        print(f"执行提供者: {self.session.get_providers()}")
    
    def predict(self, x):
        """
        推理
        
        参数:
            x: 输入数据 (numpy array 或 torch tensor)
        
        返回:
            预测结果
        """
        # 转换为 numpy
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: x})
        
        return outputs[0]


# 使用示例
onnx_engine = ONNXInference('resnet18.onnx', use_gpu=True)

# 推理
output = onnx_engine.predict(torch.randn(1, 3, 224, 224))
print(f"输出形状: {output.shape}")
```

#### 6.11.3 ONNX 优化

**使用 onnx-simplifier 优化**：

```bash
# 安装
pip install onnx-sim

# 优化
onnxsim input.onnx optimized.onnx
```

**Python 中优化**：

```python
import onnxsim

def optimize_onnx_model(input_path, output_path):
    """
    优化 ONNX 模型
    
    优化内容:
        - 常量折叠
        - 死代码消除
        - 算子融合
    """
    # 加载原始模型
    onnx_model = onnx.load(input_path)
    
    # 优化
    model_opt, check = onnxsim.simplify(onnx_model)
    
    # 保存优化后的模型
    onnx.save(model_opt, output_path)
    
    print(f"原始模型大小: {get_file_size(input_path):.2f} MB")
    print(f"优化后模型大小: {get_file_size(output_path):.2f} MB")


def get_file_size(file_path):
    """获取文件大小（MB）"""
    import os
    return os.path.getsize(file_path) / 1024 / 1024


optimize_onnx_model('resnet18.onnx', 'resnet18_optimized.onnx')
```

### 6.12 TensorRT 加速推理

**TensorRT**：NVIDIA 推理引擎，专为 GPU 优化，速度提升 2-5 倍。

#### 6.12.1 ONNX 转 TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def convert_onnx_to_tensorrt(onnx_path, engine_path='model.engine', fp16=False):
    """
    将 ONNX 模型转换为 TensorRT 引擎
    
    参数:
        onnx_path: ONNX 模型路径
        engine_path: TensorRT 引擎输出路径
        fp16: 是否使用 FP16 精度
    """
    # 创建 TensorRT 日志记录器
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 创建构建器
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析 ONNX 模型
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    
    # 配置构建器
    config = builder.create_builder_config()
    
    # 设置最大工作空间
    config.max_workspace_size = 1 << 30  # 1GB
    
    # 启用 FP16
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT 引擎已保存到: {engine_path}")


# 使用示例
convert_onnx_to_tensorrt('resnet18.onnx', 'resnet18.engine', fp16=True)
```

#### 6.12.2 TensorRT 推理

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TensorRTInference:
    """
    TensorRT 推理引擎
    
    优势:
        - GPU 加速（2-5 倍提速）
        - 自动内核调优
        - FP16/INT8 量化支持
    """
    
    def __init__(self, engine_path):
        """
        参数:
            engine_path: TensorRT 引擎路径
        """
        # 加载引擎
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            # 分配内存
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def predict(self, input_data):
        """
        推理
        
        参数:
            input_data: 输入数据 (numpy array)
        
        返回:
            预测结果
        """
        # 拷贝输入数据到 GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(stream_handle=self.stream.handle)
        
        # 拷贝输出数据到 CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # 返回输出
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])


# 使用示例
trt_engine = TensorRTInference('resnet18.engine')

# 推理
output = trt_engine.predict(np.random.randn(1, 3, 224, 224).astype(np.float32))
print(f"输出形状: {output.shape}")
```

### 6.13 移动端部署

#### 6.13.1 TFLite 转换（Android）

```python
import torch
import tensorflow as tf

def convert_to_tflite(model, tflite_path='model.tflite'):
    """
    转换 PyTorch 模型为 TFLite 格式
    
    路径: PyTorch → ONNX → TensorFlow → TFLite
    """
    # 1. PyTorch → ONNX
    onnx_path = 'temp_model.onnx'
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_path)
    
    # 2. ONNX → TensorFlow
    # 需要安装: pip install onnx-tf
    import onnx
    from onnx_tf.backend import prepare
    
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('temp_tf_model')
    
    # 3. TensorFlow → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
    
    # 量化（INT8）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # 生成量化校准数据
    def representative_dataset():
        for _ in range(100):
            data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    
    # 转换
    tflite_model = converter.convert()
    
    # 保存
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite 模型已保存到: {tflite_path}")


# 使用示例
model = ResNet18(num_classes=2)
convert_to_tflite(model, 'resnet18.tflite')
```

**Android 中使用 TFLite**（Kotlin）：

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelPath)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

// 加载模型
val tfliteModel = loadModelFile(this, "resnet18.tflite")
val interpreter = Interpreter(tfliteModel)

// 推理
val inputBuffer = ByteBuffer.allocateDirect(3 * 224 * 224 * 4).order(ByteOrder.nativeOrder())
// ... 填充 inputBuffer ...

val output = Array(1) { FloatArray(2) }  // 2 类
interpreter.run(inputBuffer, output)

val predictedClass = output[0].indices.maxByOrNull { output[0][it] }
```

#### 6.13.2 CoreML 转换（iOS）

```python
import coremltools as ct

def convert_to_coreml(model, coreml_path='model.mlmodel'):
    """
    转换 PyTorch 模型为 CoreML 格式
    """
    # 跟踪模型
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    
    # 转换为 CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS14
    )
    
    # 设置元数据
    mlmodel.short_description = "图像分类模型"
    mlmodel.author = "Your Name"
    
    # 保存
    mlmodel.save(coreml_path)
    
    print(f"CoreML 模型已保存到: {coreml_path}")


# 使用示例
model = ResNet18(num_classes=2)
convert_to_coreml(model, 'resnet18.mlmodel')
```

**iOS 中使用 CoreML**（Swift）：

```swift
import CoreML
import Vision

// 加载模型
guard let model = try? Resnet18(configuration: MLModelConfiguration()) else {
    fatalError("无法加载模型")
}

// 创建 Vision 请求
let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model.model)) { request, error in
    guard let results = request.results as? [VNClassificationObservation],
          let topResult = results.first else {
        return
    }
    
    print("预测类别: \(topResult.identifier)")
    print("置信度: \(topResult.confidence)")
}

// 准备图像
guard let image = UIImage(named: "test.jpg"),
      let ciImage = CIImage(image: image) else {
    return
}

// 执行推理
let handler = VNImageRequestHandler(ciImage: ciImage)
try? handler.perform([request])
```

### 6.14 模型服务化

#### 6.14.1 FastAPI 服务

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms

app = FastAPI(title="图像分类 API")

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_classes=2)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    图像分类 API
    
    输入: 图像文件
    输出: 预测类别和置信度
    """
    # 读取图像
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # 预处理
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
    
    # 返回结果
    return JSONResponse({
        "class_id": predicted.item(),
        "class_name": ["cat", "dog"][predicted.item()],
        "confidence": float(confidence.item())
    })


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**测试 API**：

```bash
# 启动服务
python app.py

# 测试（使用 curl）
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict

# 测试（使用 Python）
import requests

with open('test.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    print(response.json())
```

#### 6.14.2 批量推理优化

```python
from queue import Queue
import threading

class BatchInferenceEngine:
    """
    批量推理引擎
    
    优势:
        - 提高 GPU 利用率
        - 降低延迟（小 batch）
        - 吞吐量提升 2-5 倍
    """
    
    def __init__(self, model, max_batch_size=32, timeout=0.1):
        """
        参数:
            model: 模型
            max_batch_size: 最大 batch size
            timeout: 批量超时（秒）
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        
        # 请求队列
        self.queue = Queue()
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        
        # 启动推理线程
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def predict(self, input_data):
        """
        异步推理
        
        参数:
            input_data: 输入数据
        
        返回:
            未来对象（可通过 result() 获取结果）
        """
        future = Future()
        self.queue.put((input_data, future))
        return future
    
    def _inference_loop(self):
        """推理循环"""
        batch = []
        futures = []
        
        while True:
            # 获取请求
            try:
                input_data, future = self.queue.get(timeout=self.timeout)
                batch.append(input_data)
                futures.append(future)
            except:
                pass
            
            # 批量推理
            if len(batch) > 0 and (len(batch) >= self.max_batch_size or not self.queue.empty()):
                # 合并 batch
                batch_tensor = torch.cat(batch, dim=0)
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                
                # 返回结果
                for future, output in zip(futures, outputs):
                    future.set_result(output)
                
                # 清空 batch
                batch.clear()
                futures.clear()


class Future:
    """简单的 Future 对象"""
    def __init__(self):
        self.result_ = None
        self.event = threading.Event()
    
    def set_result(self, result):
        self.result_ = result
        self.event.set()
    
    def result(self):
        self.event.wait()
        return self.result_


# 使用示例
engine = BatchInferenceEngine(model, max_batch_size=32)

# 异步推理
futures = []
for i in range(100):
    input_data = torch.randn(1, 3, 224, 224)
    future = engine.predict(input_data)
    futures.append(future)

# 获取结果
for future in futures:
    output = future.result()
    print(f"输出形状: {output.shape}")
```

### 6.15 模型监控与 A/B 测试

#### 6.15.1 在线性能监控

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义指标
prediction_counter = Counter('predictions_total', 'Total number of predictions', ['model_version'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')

class MonitoredModel:
    """
    带监控的模型包装器
    
    功能:
        - 记录预测次数
        - 记录预测延迟
        - 记录准确率
        - 暴露 Prometheus 指标
    """
    
    def __init__(self, model, model_version='v1.0'):
        self.model = model
        self.model_version = model_version
        self.correct = 0
        self.total = 0
    
    @prediction_latency.time()
    def predict(self, x):
        """推理（带监控）"""
        # 预测
        output = self.model(x)
        
        # 记录次数
        prediction_counter.labels(model_version=self.model_version).inc()
        
        return output
    
    def update_accuracy(self, predictions, labels):
        """更新准确率"""
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        self.correct += correct
        self.total += total
        
        # 更新指标
        accuracy = self.correct / self.total
        model_accuracy.set(accuracy)
    
    def get_metrics(self):
        """获取监控指标"""
        return {
            "predictions_total": prediction_counter.labels(model_version=self.model_version)._value._value,
            "accuracy": self.correct / self.total if self.total > 0 else 0,
            "latency_avg": prediction_latency.observe()
        }


# 使用示例
model = ResNet18(num_classes=2)
monitored_model = MonitoredModel(model, model_version='v1.0')

# 启动 Prometheus 服务器（端口 8000）
start_http_server(8000)

# 推理（自动记录指标）
for x, y in test_loader:
    predictions = monitored_model.predict(x).argmax(dim=1)
    monitored_model.update_accuracy(predictions, y)
```

#### 6.15.2 A/B 测试框架

```python
import random
from typing import Dict, List

class ABTestFramework:
    """
    A/B 测试框架
    
    功能:
        - 随机分配用户到不同模型
        - 收集性能指标
        - 统计显著性检验
    """
    
    def __init__(self, models: Dict[str, 'Model'], ratios: Dict[str, float]):
        """
        参数:
            models: 模型字典 {"model_a": model_a, "model_b": model_b}
            ratios: 分配比例 {"model_a": 0.7, "model_b": 0.3}
        """
        self.models = models
        self.ratios = ratios
        self.results = {name: [] for name in models.keys()}
    
    def assign_model(self, user_id: str) -> str:
        """
        为用户分配模型
        
        参数:
            user_id: 用户 ID
        
        返回:
            模型名称
        """
        # 基于 user_id 哈希，确保同一用户始终分配到同一模型
        hash_value = hash(user_id) % 100
        cumulative = 0
        
        for model_name, ratio in self.ratios.items():
            cumulative += ratio * 100
            if hash_value < cumulative:
                return model_name
        
        return list(self.ratios.keys())[0]
    
    def predict(self, user_id: str, input_data) -> Dict:
        """
        推理（自动分配模型）
        
        返回:
            {"model_name": str, "prediction": ..., "latency": float}
        """
        # 分配模型
        model_name = self.assign_model(user_id)
        model = self.models[model_name]
        
        # 推理
        start_time = time.time()
        prediction = model.predict(input_data)
        latency = time.time() - start_time
        
        # 记录结果
        self.results[model_name].append({
            "prediction": prediction,
            "latency": latency
        })
        
        return {
            "model_name": model_name,
            "prediction": prediction,
            "latency": latency
        }
    
    def evaluate(self, labels: Dict[str, List]) -> Dict:
        """
        评估模型性能
        
        参数:
            labels: 真实标签 {"model_a": [label1, label2, ...], ...}
        
        返回:
            性能对比
        """
        metrics = {}
        
        for model_name, predictions in self.results.items():
            model_labels = labels[model_name]
            
            # 计算准确率
            correct = sum(1 for p, l in zip(predictions, model_labels) if p["prediction"] == l)
            accuracy = correct / len(predictions) if predictions else 0
            
            # 计算平均延迟
            avg_latency = sum(p["latency"] for p in predictions) / len(predictions) if predictions else 0
            
            metrics[model_name] = {
                "accuracy": accuracy,
                "avg_latency": avg_latency,
                "num_predictions": len(predictions)
            }
        
        return metrics
    
    def print_comparison(self):
        """打印对比结果"""
        print("=" * 60)
        print("A/B 测试结果对比")
        print("=" * 60)
        
        metrics = self.evaluate(labels={})  # 需要提供真实标签
        
        for model_name, metric in metrics.items():
            print(f"\n模型: {model_name}")
            print(f"  预测次数: {metric['num_predictions']}")
            print(f"  准确率: {metric['accuracy']:.4f}")
            print(f"  平均延迟: {metric['avg_latency']:.4f}s")


# 使用示例
model_a = ResNet18(num_classes=2)
model_b = ResNet18(num_classes=2)

ab_test = ABTestFramework(
    models={"model_a": model_a, "model_b": model_b},
    ratios={"model_a": 0.7, "model_b": 0.3}
)

# 为不同用户分配模型
for user_id in ["user1", "user2", "user3", "user4", "user5"]:
    input_data = torch.randn(1, 3, 224, 224)
    result = ab_test.predict(user_id, input_data)
    print(f"用户 {user_id} 分配到 {result['model_name']}")
```

### 6.16 灰度发布与回滚

```python
class CanaryDeployment:
    """
    灰度发布（金丝雀发布）
    
    策略:
        - 初始: 5% 流量到新模型
        - 稳定后: 50% 流量到新模型
        - 无问题: 100% 流量到新模型
        - 有问题: 立即回滚
    """
    
    def __init__(self, old_model, new_model):
        self.old_model = old_model
        self.new_model = new_model
        self.traffic_ratio = 0.05  # 初始 5% 流量到新模型
        
        # 监控指标
        self.error_rate_new = 0
        self.error_rate_old = 0
        self.requests_new = 0
        self.requests_old = 0
        self.errors_new = 0
        self.errors_old = 0
    
    def predict(self, input_data):
        """推理（根据流量比例分配）"""
        # 随机选择模型
        if random.random() < self.traffic_ratio:
            # 新模型
            try:
                output = self.new_model.predict(input_data)
                self.requests_new += 1
                return output
            except Exception as e:
                self.errors_new += 1
                # 回退到旧模型
                print(f"新模型出错，回退到旧模型: {e}")
                output = self.old_model.predict(input_data)
                self.requests_old += 1
                return output
        else:
            # 旧模型
            try:
                output = self.old_model.predict(input_data)
                self.requests_old += 1
                return output
            except Exception as e:
                self.errors_old += 1
                raise e
    
    def update_metrics(self):
        """更新监控指标"""
        if self.requests_new > 0:
            self.error_rate_new = self.errors_new / self.requests_new
        if self.requests_old > 0:
            self.error_rate_old = self.errors_old / self.requests_old
    
    def should_increase_traffic(self) -> bool:
        """判断是否应该增加新模型流量"""
        self.update_metrics()
        
        # 条件：新模型错误率 <= 旧模型错误率 * 1.1
        return self.error_rate_new <= self.error_rate_old * 1.1
    
    def increase_traffic(self):
        """增加新模型流量"""
        if self.should_increase_traffic():
            self.traffic_ratio = min(self.traffic_ratio * 2, 1.0)
            print(f"增加新模型流量到: {self.traffic_ratio * 100:.1f}%")
        else:
            print(f"新模型错误率过高，保持当前流量: {self.error_rate_new:.4f}")
    
    def rollback(self):
        """回滚到旧模型"""
        self.traffic_ratio = 0
        print("⚠️  回滚到旧模型")


# 使用示例
canary = CanaryDeployment(old_model, new_model)

# 逐步增加流量
for _ in range(10):
    for _ in range(100):
        input_data = torch.randn(1, 3, 224, 224)
        canary.predict(input_data)
    
    canary.increase_traffic()

# 最终回滚（如果有问题）
if canary.error_rate_new > 0.01:
    canary.rollback()
```

### 6.17 修炼心得总结

**部署技术对比表**：

| 技术 | 优势 | 局限 | 适用场景 |
|------|------|------|---------|
| **ONNX** | 跨平台、通用 | 优化有限 | 通用部署 |
| **TensorRT** | GPU 极速 | 仅 NVIDIA GPU | 高性能推理 |
| **TFLite** | 移动端优化 | TensorFlow 生态 | Android |
| **CoreML** | iOS 原生 | 仅 Apple 设备 | iOS |
| **FastAPI** | 快速开发 | 单机限制 | 快速原型 |
| **TorchServe** | 生产级 | 配置复杂 | 大规模部署 |

**部署流程总结**：

```
1. 模型导出
   ├── ONNX（通用格式）
   ├── TensorRT（GPU 加速）
   ├── TFLite（Android）
   └── CoreML（iOS）

2. 服务化
   ├── FastAPI（快速开发）
   ├── TorchServe（生产级）
   └── 自定义服务（灵活控制）

3. 监控与优化
   ├── Prometheus 监控
   ├── A/B 测试
   └── 灰度发布

4. 持续优化
   ├── 性能调优
   ├── 模型更新
   └── 自动回滚
```

**修炼建议**：

1. **性能优先**：TensorRT > ONNX Runtime > PyTorch
2. **移动端优先**：TFLite（Android）> CoreML（iOS）
3. **监控先行**：部署前建立监控体系
4. **灰度发布**：逐步增加流量，避免全量失败
5. **自动回滚**：发现问题立即回滚

---

## 附录 C：神识合道 — 分布式训练技术

> *"修炼者之力，终有极限。欲成就不朽，唯有合道。*
> *将多个修炼者的力量合而为一，方能突破天地桎梏。*
> *分布式训练，就是让多个 GPU 合力，成就超大规模模型。"*
> *——《焚诀》合道篇*

### C.1 为何需要分布式训练？

**问题背景**：

随着模型规模爆炸式增长，单卡训练面临三大瓶颈：

| 瓶颈 | 具体表现 | 影响 |
|------|---------|------|
| **显存不足** | 单卡 24GB 显存无法训练 7B 模型 | 无法训练大模型 |
| **训练太慢** | 单卡训练 GPT-3 需要数年 | 无法实际应用 |
| **数据太大** | 大规模数据集单卡处理太慢 | 训练周期长 |

**分布式训练目标**：

- **扩展显存**：多卡显存叠加（4 卡 → 96GB）
- **加速训练**：线性加速（4 卡 → 4 倍速度）
- **支持大模型**：训练 10B+ 参数模型

### C.2 数据并行（DataParallel）

**核心思想**：每个 GPU 拥有一份完整的模型副本，处理不同的数据，然后同步梯度。

**工作流程**：

```
输入数据 → 分配到不同 GPU
          ↓
    ┌─────┴─────┐
    ↓           ↓
  GPU 0       GPU 1
  副本 1       副本 2
    ↓           ↓
  前向传播     前向传播
  计算梯度     计算梯度
    ↓           ↓
    └─────┬─────┘
          ↓
      同步梯度
          ↓
      更新参数
```

#### C.2.1 PyTorch DataParallel

**简单实现**：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 定义模型
model = ResNet18(num_classes=10)

# 包装为 DataParallel（多卡训练）
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU")
    model = nn.DataParallel(model)

model = model.cuda()

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for data, labels in train_loader:
        data, labels = data.cuda(), labels.cuda()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**DataParallel 的局限性**：

| 局限 | 说明 |
|------|------|
| **单进程**：所有 GPU 在一个进程中 | CPU 成为瓶颈 |
| **负载不均**：主 GPU（GPU 0）负载更重 | 效率降低 |
| **不支持复杂操作**：RNN、某些自定义层 | 功能受限 |

### C.3 分布式数据并行（DistributedDataParallel）

**核心思想**：每个进程对应一个 GPU，独立计算梯度，然后通过 All-Reduce 同步。

**工作原理**：

```
进程 0 (GPU 0)          进程 1 (GPU 1)
    ↓                       ↓
  模型副本 0               模型副本 1
    ↓                       ↓
  前向传播 0              前向传播 1
  梯度 0                  梯度 1
    ↓                       ↓
    └─────────┬─────────────┘
              ↓
          All-Reduce
        (梯度同步)
              ↓
    ┌─────────┴─────────────┐
    ↓                       ↓
  更新参数 0               更新参数 1
```

#### C.3.1 DDP 完整实现

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """
    初始化分布式环境
    
    参数:
        rank: 当前进程的 rank
        world_size: 总进程数
    """
    # 设置通信后端
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPU 通信
        init_method='env://',  # 从环境变量读取
        world_size=world_size,
        rank=rank
    )


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_ddp(rank, world_size):
    """
    DDP 训练函数
    
    参数:
        rank: 当前进程的 rank
        world_size: 总进程数
    """
    # 初始化
    setup(rank, world_size)
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    
    # 创建模型
    model = ResNet18(num_classes=10).cuda(rank)
    
    # 包装为 DDP
    model = DDP(model, device_ids=[rank])
    
    # 创建数据加载器（使用 DistributedSampler）
    train_dataset = ...
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4
    )
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(num_epochs):
        # 设置 epoch（确保每个进程的数据随机性一致）
        train_sampler.set_epoch(epoch)
        
        model.train()
        for data, labels in train_loader:
            data, labels = data.cuda(rank), labels.cuda(rank)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 只有 rank 0 打印日志
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
    
    # 清理
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

#### C.3.2 启动 DDP 训练

**方法 1：使用 torchrun（推荐）**

```bash
# 4 卡训练
torchrun --nproc_per_node=4 train_ddp.py

# 多机训练（节点 1）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 train_ddp.py

# 多机训练（节点 2）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=29500 train_ddp.py
```

**方法 2：使用 python -m**

```bash
# 单机多卡
python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py

# 多机
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train_ddp.py
```

#### C.3.3 DDP vs DP 对比

| 特性 | DataParallel | DistributedDataParallel |
|------|--------------|------------------------|
| **进程数** | 单进程 | 多进程（每个 GPU 一个） |
| **效率** | 较低（CPU 瓶颈） | 高（各进程独立） |
| **负载均衡** | 不均（GPU 0 更重） | 均衡 |
| **支持模型** | 简单模型 | 复杂模型（RNN、自定义层） |
| **速度** | 较慢 | 快 |
| **推荐** | 快速原型 | 生产环境 |

### C.4 混合精度训练（Mixed Precision Training）

**核心思想**：使用 FP16 进行前向/反向传播，FP32 进行参数更新。

**为什么有效**：

| 精度 | 显存占用 | 计算速度 | 数值范围 |
|------|---------|---------|---------|
| **FP32** | 100% | 1x | ±3.4e38 |
| **FP16** | 50% | 2-3x | ±6.5e4 |
| **BF16** | 50% | 2-3x | ±3.4e38 |

#### C.4.1 AMP（自动混合精度）训练

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_amp(model, train_loader, num_epochs=10):
    """
    混合精度训练（AMP）
    
    优势:
        - 显存占用减少 50%
        - 训练速度提升 2-3 倍
        - 自动处理精度转换
    """
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建 GradScaler（用于 Loss Scaling）
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data, labels in train_loader:
            data, labels = data.cuda(), labels.cuda()
            
            # 使用 autocast 自动转换精度
            with autocast():
                # 前向传播（自动使用 FP16）
                output = model(data)
                loss = criterion(output, labels)
            
            # 反向传播（自动使用 FP16 + Loss Scaling）
            scaler.scale(loss).backward()
            
            # 更新参数（自动使用 FP32）
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")
```

#### C.4.2 Loss Scaling 原理

**问题**：FP16 数值范围小，梯度容易下溢（变为 0）。

**解决方案**：将 loss 乘以一个大数（如 2^16），反向传播后再除回来。

```python
# 手动 Loss Scaling（不推荐，使用自动版本）
loss = criterion(output, labels)
loss = loss * 65536  # 乘以 2^16
loss.backward()

# 梯度除回来
for param in model.parameters():
    if param.grad is not None:
        param.grad = param.grad / 65536
```

**自动 Loss Scaling**：

```python
scaler = GradScaler(
    init_scale=2.0 ** 16,      # 初始缩放因子
    growth_factor=2.0,         # 无梯度溢出时，每次翻倍
    backoff_factor=0.5,       # 梯度溢出时，每次减半
    growth_interval=2000,     # 每 2000 次迭代增长一次
)

# 自动检测梯度溢出
scaler.scale(loss).backward()
scaler.step(optimizer)  # 如果溢出，自动回滚
scaler.update()
```

### C.5 梯度累积（Gradient Accumulation）

**核心思想**：模拟大 batch size，使用小 batch size 进行多次前向/反向传播后统一更新参数。

**为什么需要**：

| 场景 | 需求 | 现实 |
|------|------|------|
| **训练稳定** | batch_size=256 | 单卡显存仅支持 32 |
| **大模型训练** | batch_size=64 | 单卡仅支持 4 |

#### C.5.1 梯度累积实现

```python
def train_with_gradient_accumulation(
    model, 
    train_loader, 
    target_batch_size=256, 
    actual_batch_size=32,
    num_epochs=10
):
    """
    梯度累积训练
    
    参数:
        model: 模型
        train_loader: 数据加载器（batch_size=actual_batch_size）
        target_batch_size: 目标 batch size
        actual_batch_size: 实际 batch size
    """
    # 计算累积步数
    accumulation_steps = target_batch_size // actual_batch_size
    
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # 每个 epoch 开始时清零
        
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, labels)
            
            # 归一化 loss（除以累积步数）
            loss = loss / accumulation_steps
            
            # 反向传播（累积梯度）
            loss.backward()
            
            # 每累积 accumulation_steps 次后更新参数
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
            
            # 最后一次迭代（可能不满 accumulation_steps）
            if (i + 1) == len(train_loader) and (i + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
```

**效果对比**：

```
无梯度累积:
  batch_size=32
  单卡显存: 8GB
  准确率: 88.5%

有梯度累积:
  batch_size=32 (累积 8 次 = 256)
  单卡显存: 8GB
  准确率: 90.2%  ← 提升 1.7%！
```

### C.6 FSDP（完全分片数据并行）

**核心思想**：不仅数据并行，还将模型参数、梯度、优化器状态分片到不同 GPU。

**为什么需要**：

| 方法 | 参数存储 | 梯度存储 | 优化器存储 | 总显存占用 |
|------|---------|---------|-----------|-----------|
| **DDP** | 完整 | 完整 | 完整 | 3x 参数量 |
| **FSDP** | 分片 | 分片 | 分片 | 参数量 + 梯度 + 少量优化器 |

**适用场景**：训练超大模型（10B+ 参数）。

#### C.6.1 FSDP 完整实现

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

def train_fsdp(rank, world_size):
    """
    FSDP 训练
    
    优势:
        - 显存占用显著降低（3x → 1.5x）
        - 支持训练超大模型
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 创建模型
    model = LargeModel(...).cuda(rank)
    
    # FSDP 配置
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    # 包装为 FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
        mixed_precision=mixed_precision_policy,
        device_id=rank
    )
    
    # 训练（与 DDP 类似）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.cuda(rank), labels.cuda(rank)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)
```

### C.7 DeepSpeed

**核心思想**：Microsoft 推出的分布式训练框架，提供 ZeRO（Zero Redundancy Optimizer）优化策略。

**ZeRO 策略**：

| 阶段 | 分片内容 | 显存占用 | 通信开销 |
|------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | 减少 4 倍 | 低 |
| **ZeRO-2** | 优化器状态 + 梯度 | 减少 8 倍 | 中 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | 减少 N 倍（N=GPU数） | 高 |

#### C.7.1 DeepSpeed 完整实现

```python
import deepspeed

def train_deepspeed():
    """
    DeepSpeed 训练
    
    优势:
        - ZeRO-3 可以训练万亿参数模型
        - 自动梯度检查点
        - 混合精度训练
    """
    # 创建模型
    model = LargeModel(...)
    
    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 32,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 3,  # ZeRO-3
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100
    }
    
    # 初始化 DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 训练
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.cuda(), labels.cuda()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, labels)
            
            # 反向传播
            model.backward(loss)
            model.step()
```

**启动 DeepSpeed**：

```bash
deepspeed --num_gpus=4 train_deepspeed.py --deepspeed ds_config.json
```

### C.8 综合案例：完整分布式训练流程

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

def distributed_training(rank, world_size):
    """
    完整分布式训练流程
    
    技术:
        1. DDP（分布式数据并行）
        2. AMP（混合精度训练）
        3. 梯度累积
        4. 梯度裁剪
    """
    # 初始化
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 创建模型
    model = ResNet18(num_classes=10).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 梯度累积
    accumulation_steps = 8  # 模拟 batch_size=256
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(rank), labels.cuda(rank)
            
            # 混合精度前向传播
            with autocast():
                output = model(data)
                loss = criterion(output, labels) / accumulation_steps
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积更新
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # 验证（只在 rank 0）
        if rank == 0:
            val_acc = validate(model, val_loader)
            print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
    
    # 保存模型（只在 rank 0）
    if rank == 0:
        torch.save(model.module.state_dict(), 'model.pth')
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(distributed_training, args=(world_size,), nprocs=world_size, join=True)
```

### C.9 修炼心得总结

**分布式训练技术对比表**：

| 技术 | 适用场景 | 显存节省 | 速度 | 复杂度 |
|------|---------|---------|------|--------|
| **DataParallel** | 快速原型 | 无 | 慢 | 低 |
| **DDP** | 单机多卡 | 无 | 快 | 中 |
| **AMP** | 所有场景 | 50% | 2-3x | 低 |
| **梯度累积** | 大 batch 需求 | 无 | 慢 | 低 |
| **FSDP** | 超大模型 | 50-80% | 快 | 高 |
| **DeepSpeed** | 超大规模 | 80-95% | 快 | 高 |

**修炼建议**：

1. **小模型（< 1B）**：DDP + AMP
2. **中等模型（1-10B）**：DDP + AMP + 梯度累积
3. **大模型（10-100B）**：FSDP + AMP
4. **超大模型（> 100B）**：DeepSpeed ZeRO-3

---

## 附录 D：实战修炼 — 从理论到实战

> *"修炼者不能只知理论，必须在实战中检验所学。*
> *以下是三大经典修炼场景，让你将斗师境界的神术付诸实践。"*
> *——《焚诀》实战篇*

### D.1 文本分类修炼 — 情感分析

**场景描述**：给定用户评论，判断其情感倾向（正面/负面）

**修炼目标**：
- 掌握文本数据的预处理流程
- 实现基于 Transformer 的文本分类器
- 学习评估指标的选取与计算
- 完整的训练-验证-测试流程

#### D.1.1 数据准备

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 创建模拟数据集（实际项目中应使用 IMDB、Yelp 等真实数据）
def create_sample_dataset():
    """
    创建情感分析数据集
    """
    data = {
        'text': [
            "This movie is absolutely fantastic! I loved every minute of it.",
            "Terrible experience. The worst movie I've ever seen.",
            "Great acting and amazing plot. Highly recommended!",
            "Boring and predictable. Don't waste your time.",
            "Outstanding performance by the lead actor.",
            "I fell asleep halfway through. Very disappointing.",
            "The cinematography is beautiful and the story is compelling.",
            "Poor script and awful direction. A total mess.",
            "One of the best films of the year!",
            "I regret watching this. Complete waste of money."
        ] * 100,  # 复制 100 次以增加数据量
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100
    }
    return pd.DataFrame(data)

class SentimentDataset(Dataset):
    """
    情感分析数据集
    """
    
    def __init__(self, texts, labels, vocab, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        tokens = self.tokenizer(text.lower())
        
        # 转换为索引
        token_indices = [self.vocab[token] for token in tokens if token in self.vocab]
        
        # 截断/填充
        if len(token_indices) > self.max_length:
            token_indices = token_indices[:self.max_length]
        else:
            token_indices += [self.vocab['<pad>']] * (self.max_length - len(token_indices))
        
        return {
            'input_ids': torch.tensor(token_indices, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1] * min(len(token_indices), self.max_length) + 
                [0] * (self.max_length - min(len(token_indices), self.max_length)),
                dtype=torch.long
            ),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 数据预处理
def preprocess_data():
    """
    数据预处理流程
    """
    # 加载数据
    df = create_sample_dataset()
    
    # 分词器
    tokenizer = get_tokenizer('basic_english')
    
    # 构建词表
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text.lower())
    
    vocab = build_vocab_from_iterator(
        yield_tokens(df['text']),
        specials=['<pad>', '<unk>'],
        min_freq=2
    )
    vocab.set_default_index(vocab['<unk>'])
    
    # 划分数据集
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"词表大小: {len(vocab)}")
    
    return train_df, val_df, test_df, vocab, tokenizer
```

#### D.1.2 模型定义

```python
class TransformerSentimentClassifier(nn.Module):
    """
    基于 Transformer 的情感分类器
    """
    
    def __init__(
        self, 
        vocab_size, 
        embed_dim=256, 
        num_heads=8, 
        num_layers=4, 
        dim_feedforward=512,
        max_length=128,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(embed_dim, max_length)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.embedding(input_ids)  # (B, L, D)
        
        # 位置编码
        x = self.position_encoding(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # (B, L, D)
        
        # 聚合（取第一个 token 或平均池化）
        if attention_mask is not None:
            # 使用 attention_mask 进行加权平均
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
        else:
            x = x.mean(dim=1)  # 平均池化
        
        # 分类
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        
        # 创建位置矩阵
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 计算 PE
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]

import math
```

#### D.1.3 训练与评估

```python
def evaluate_model(model, data_loader, criterion, device):
    """
    评估模型
    
    返回：
        - loss
        - accuracy
        - precision
        - recall
        - f1
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_sentiment_classifier(
    train_df, 
    val_df, 
    vocab, 
    tokenizer,
    max_length=128,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    device='cuda'
):
    """
    训练情感分类器
    """
    # 创建数据集
    train_dataset = SentimentDataset(train_df['text'].values, train_df['label'].values, vocab, tokenizer, max_length)
    val_dataset = SentimentDataset(val_df['text'].values, val_df['label'].values, vocab, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    model = TransformerSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_length=max_length,
        num_classes=2,
        dropout=0.1
    ).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练循环
    best_val_accuracy = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# 运行训练
def run_sentiment_classification():
    """
    运行完整的情感分类流程
    """
    # 预处理数据
    train_df, val_df, test_df, vocab, tokenizer = preprocess_data()
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model, train_losses, val_losses = train_sentiment_classifier(
        train_df, val_df, vocab, tokenizer,
        max_length=128,
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        device=device
    )
    
    # 测试集评估
    test_dataset = SentimentDataset(test_df['text'].values, test_df['label'].values, vocab, tokenizer, 128)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*50)
    print("测试集评估结果")
    print("="*50)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    
    # 详细分类报告
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())
    
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
    
    return model, vocab, tokenizer

# 实际推理示例
def predict_sentiment(model, vocab, tokenizer, text, device='cuda'):
    """
    预测单条文本的情感
    """
    model.eval()
    
    # 分词
    tokens = tokenizer(text.lower())
    token_indices = [vocab[token] for token in tokens if token in vocab]
    
    # 截断/填充
    max_length = 128
    if len(token_indices) > max_length:
        token_indices = token_indices[:max_length]
    else:
        token_indices += [vocab['<pad>']] * (max_length - len(token_indices))
    
    # 转换为张量
    input_ids = torch.tensor([token_indices], dtype=torch.long).to(device)
    attention_mask = torch.tensor(
        [[1] * min(len(token_indices), max_length) + [0] * (max_length - min(len(token_indices), max_length))],
        dtype=torch.long
    ).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=-1)
        pred = outputs.argmax(dim=-1).item()
    
    return {
        'prediction': pred,
        'label': 'Positive' if pred == 1 else 'Negative',
        'confidence': probs[0][pred].item(),
        'probabilities': probs[0].cpu().numpy()
    }
```

### D.2 图像分割修炼 — 语义分割

**场景描述**：给定图像，为每个像素分配类别标签

**修炼目标**：
- 理解语义分割的基本概念
- 实现基于 U-Net 的分割模型
- 学习分割任务的特殊处理（转置卷积、跳跃连接）
- 掌握 Dice Loss 等分割专用损失函数

#### D.2.1 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    双重卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样模块：MaxPool -> DoubleConv
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    上采样模块：Upsample -> DoubleConv（带跳跃连接）
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 转置卷积上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 卷积
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: 来自上层的特征
        x2: 来自编码器的跳跃连接特征
        """
        x1 = self.up(x1)
        
        # 处理可能的尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net 模型
    
    架构：
        - 编码器：逐步下采样，提取特征
        - 解码器：逐步上采样，恢复空间分辨率
        - 跳跃连接：将编码器的特征传到解码器
    """
    
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # 编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        return logits
```

#### D.2.2 损失函数

```python
class DiceLoss(nn.Module):
    """
    Dice Loss（适用于分割任务）
    
    公式：
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        Loss = 1 - Dice
    
    优点：
        - 对类别不平衡不敏感
        - 直接优化 IoU
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) logits
        target: (B, H, W) 类别索引
        """
        # softmax
        pred = F.softmax(pred, dim=1)
        
        # 转换为 one-hot
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # 计算交集和并集
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        # Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 平均
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    组合损失：CrossEntropy + Dice
    """
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice
```

#### D.2.3 评估指标

```python
def calculate_iou(pred, target, num_classes):
    """
    计算 IoU（交并比）
    
    公式：
        IoU = |预测 ∩ 真实| / |预测 ∪ 真实|
    """
    ious = []
    
    pred = pred.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)

def calculate_pixel_accuracy(pred, target):
    """
    计算像素准确率
    """
    pred = pred.argmax(dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def evaluate_segmentation(model, data_loader, criterion, num_classes, device):
    """
    评估分割模型
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # 计算指标
            ious = calculate_iou(outputs, masks, num_classes)
            pixel_acc = calculate_pixel_accuracy(outputs, masks)
            
            # 平均 IoU（忽略 NaN）
            valid_ious = ious[~np.isnan(ious)]
            if len(valid_ious) > 0:
                total_iou += valid_ious.mean()
            
            total_pixel_acc += pixel_acc
    
    num_batches = len(data_loader)
    return {
        'loss': total_loss / num_batches,
        'miou': total_iou / num_batches,
        'pixel_accuracy': total_pixel_acc / num_batches
    }
```

#### D.2.4 数据集类

```python
class SegmentationDataset(Dataset):
    """
    语义分割数据集
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return {
            'image': image,
            'mask': mask
        }

# 数据增强
def get_segmentation_augmentation():
    """
    分割数据增强
    """
    import torchvision.transforms as transforms
    
    # 图像增强
    image_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 标签增强（只做几何变换，不归一化）
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform
```

### D.3 序列生成修炼 — 简易语言模型

**场景描述**：给定前文，生成后续文本

**修炼目标**：
- 理解自回归生成的基本原理
- 实现基于 LSTM 的语言模型
- 学习文本生成的解码策略（贪婪、束搜索、采样）
- 掌握困惑度（Perplexity）等评估指标

#### D.3.1 模型定义

```python
class LSTMLanguageModel(nn.Module):
    """
    LSTM 语言模型
    
    输入：文本序列
    输出：下一个词的概率分布
    """
    
    def __init__(
        self, 
        vocab_size, 
        embed_dim=256, 
        hidden_dim=512, 
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, input_ids, hidden=None):
        """
        input_ids: (B, L) 词索引
        hidden: LSTM 的隐藏状态
        """
        # 词嵌入
        embed = self.dropout(self.embedding(input_ids))  # (B, L, D)
        
        # LSTM
        output, hidden = self.lstm(embed, hidden)  # (B, L, H)
        
        # Dropout
        output = self.dropout(output)
        
        # 输出 logits
        logits = self.fc(output)  # (B, L, V)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
```

#### D.3.2 文本生成

```python
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=0.9):
    """
    采样下一个 token
    
    策略：
        1. Temperature: 控制随机性
           - 温度高 → 分布更均匀 → 更多样性
           - 温度低 → 分布更尖锐 → 更确定性
        
        2. Top-k: 只从概率最高的 k 个候选中采样
        
        3. Top-p (Nucleus): 从累积概率达到 p 的候选中采样
    """
    # 应用温度
    logits = logits / temperature
    
    # Top-k 采样
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, indices, values)
    
    # Top-p 采样
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过 p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # 采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

def generate_text(
    model, 
    vocab, 
    prompt, 
    tokenizer, 
    max_length=100,
    temperature=1.0,
    top_k=None,
    top_p=0.9,
    device='cuda'
):
    """
    生成文本
    
    参数：
        - prompt: 提示文本
        - max_length: 最大生成长度
        - temperature: 采样温度
        - top_k: Top-k 采样
        - top_p: Top-p 采样
    """
    model.eval()
    
    # 分词
    tokens = tokenizer(prompt.lower())
    token_ids = [vocab[token] for token in tokens if token in vocab]
    
    # 转换为张量
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # 初始化隐藏状态
    hidden = model.init_hidden(1, device)
    
    generated_ids = token_ids.copy()
    
    with torch.no_grad():
        # 编码提示词
        for _ in range(len(token_ids) - 1):
            _, hidden = model(input_ids[:, :1], hidden)
            input_ids = input_ids[:, 1:]
        
        # 自回归生成
        for _ in range(max_length):
            # 前向传播
            logits, hidden = model(input_ids[:, -1:], hidden)
            
            # 采样下一个 token
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            
            # 添加到生成序列
            generated_ids.append(next_token.item())
            
            # 更新输入
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 遇到结束符则停止
            if next_token.item() == vocab.get('<eos>', -1):
                break
    
    # 解码
    itos = vocab.get_itos()
    generated_text = ' '.join([itos[idx] for idx in generated_ids if idx < len(itos)])
    
    return generated_text

def beam_search_decode(
    model, 
    vocab, 
    prompt, 
    tokenizer, 
    beam_width=5, 
    max_length=100,
    device='cuda'
):
    """
    束搜索解码
    
    思想：
        - 同时维护 top-k 个候选序列
        - 每一步扩展所有候选
        - 保留分数最高的 k 个序列
    
    优点：
        - 比贪婪搜索更好
        - 比纯随机搜索更稳定
    """
    model.eval()
    
    # 分词
    tokens = tokenizer(prompt.lower())
    token_ids = [vocab[token] for token in tokens if token in vocab]
    
    # 初始化束
    beams = [(token_ids, 0.0)]  # (序列, 分数)
    
    for _ in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq[-1] == vocab.get('<eos>', -1):
                new_beams.append((seq, score))
                continue
            
            # 前向传播
            input_ids = torch.tensor([seq], dtype=torch.long).to(device)
            with torch.no_grad():
                logits, _ = model(input_ids)
            
            # 获取 top-k 下一个 token
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
            
            # 扩展束
            for i in range(beam_width):
                next_token = top_k_indices[i].item()
                next_score = top_k_probs[i].item()
                
                new_seq = seq + [next_token]
                new_score = score + next_score
                
                new_beams.append((new_seq, new_score))
        
        # 保留 top-k
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # 检查是否所有束都结束
        if all(seq[-1] == vocab.get('<eos>', -1) for seq, _ in beams):
            break
    
    # 返回最佳序列
    best_seq = beams[0][0]
    itos = vocab.get_itos()
    generated_text = ' '.join([itos[idx] for idx in best_seq if idx < len(itos)])
    
    return generated_text
```

#### D.3.3 评估指标

```python
def calculate_perplexity(model, data_loader, criterion, device):
    """
    计算困惑度（Perplexity）
    
    公式：
        PPL = exp(average_loss)
    
    含义：
        - 模型对测试集的"困惑程度"
        - 越低越好
        - PPL=10 表示模型在 10 个等概率的选项中犹豫
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target'].to(device)
            
            # 前向传播
            logits, _ = model(input_ids)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def compute_bleu(references, hypotheses, max_n=4):
    """
    计算 BLEU 分数
    
    参数：
        references: 参考文本列表
        hypotheses: 生成文本列表
        max_n: 最大的 n-gram
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    # 计算每个 n-gram 的精确率
    precisions = []
    
    for n in range(1, max_n + 1):
        correct = 0
        total = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))
            
            for ngram, count in hyp_ngrams.items():
                correct += min(count, ref_ngrams.get(ngram, 0))
                total += count
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(correct / total)
    
    # 几何平均
    if all(p > 0 for p in precisions):
        score = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        score = 0.0
    
    # 短句惩罚
    ref_lengths = [len(ref.split()) for ref in references]
    hyp_lengths = [len(hyp.split()) for hyp in hypotheses]
    
    avg_ref_len = sum(ref_lengths) / len(ref_lengths)
    avg_hyp_len = sum(hyp_lengths) / len(hyp_lengths)
    
    if avg_hyp_len < avg_ref_len:
        bp = math.exp(1 - avg_ref_len / avg_hyp_len)
    else:
        bp = 1.0
    
    return score * bp
```

### D.4 修炼心得总结

| 实战场景 | 核心技术 | 关键指标 | 适用任务 |
|---------|---------|---------|---------|
| **文本分类** | Transformer + 分类头 | Accuracy, F1 | 情感分析、垃圾邮件检测 |
| **图像分割** | U-Net + 跳跃连接 | IoU, Pixel Acc | 医学影像、自动驾驶 |
| **序列生成** | LSTM + 解码策略 | Perplexity, BLEU | 文本生成、机器翻译 |

**修炼建议**：

1. **从简单开始**：先用小数据集跑通流程，再逐步增加复杂度
2. **监控指标**：不要只看 loss，要关注业务相关的指标
3. **可视化结果**：分割、生成任务尤其需要可视化验证
4. **A/B 测试**：对比不同策略的效果
5. **记录实验**：保存超参数、结果，便于后续复现

---

## 第七章：造物之术 — 生成式模型基础

> *"斗师之境，不仅能感知天地灵气，更能以灵力造物。*
> *能无中生有者，方为造物主。*
> *生成式模型，就是让机器学会创造，而不是仅仅判断。"*
> *——《焚诀》造物篇*

### 7.1 为何需要生成式模型？

**从判别到生成**：

- **判别模型（Discriminative）**：输入 → 输出（判断）
  - 图像分类：这是猫还是狗？
  - 情感分析：这段话是正面还是负面？

- **生成模型（Generative）**：无/输入 → 输出（创造）
  - 文本生成：写一首诗
  - 图像生成：画一只猫
  - 代码生成：写一个排序算法

**生成式 AI 时代**：

- **GPT 系列**：文本生成、代码生成、对话
- **Stable Diffusion**：图像生成、图像编辑
- **Sora**：视频生成
- **MusicLM**：音乐生成

**生成模型的核心挑战**：

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| **多样性**：生成内容不能单调 | 避免总是生成相似内容 | Temperature、Top-k 采样 |
| **质量**：生成内容必须合理 | 避免胡言乱语/模糊图像 | 大规模预训练、评估指标 |
| **控制性**：用户希望控制生成 | 指定主题/风格 | Prompt Engineering、ControlNet |

### 7.2 自回归语言模型

**核心思想**：给定上下文，预测下一个 token。

```
P(今天 天气 很 好) 
= P(今天) × P(天气 | 今天) × P(很 | 今天 天气) × P(好 | 今天 天气 很)
```

#### 7.2.1 GPT 风格模型实现

```python
import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    """
    GPT 基础块
    
    结构:
        - LayerNorm
        - Multi-Head Attention（带因果掩码）
        - LayerNorm
        - Feed-Forward Network
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-Head Attention + 残差连接
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feed-Forward Network + 残差连接
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class GPT(nn.Module):
    """
    GPT（Generative Pre-trained Transformer）
    
    结构:
        - Token Embedding
        - Positional Encoding
        - 多个 GPT Block
        - Language Modeling Head
    """
    
    def __init__(
        self, 
        vocab_size, 
        d_model=768, 
        num_heads=12, 
        num_layers=12, 
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1
    ):
        super().__init__()
        
        # Token 和位置编码
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 语言建模头
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（可选）
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        """
        参数:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) 1 表示有效，0 表示填充
        """
        # Embedding
        x = self.token_embedding(input_ids)  # (B, S, D)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 创建因果掩码
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Transformer 层
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # 语言建模头
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)
        
        return logits


class MultiHeadAttention(nn.Module):
    """多头注意力（带因果掩码）"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)


class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """位置编码（正弦编码）"""
    
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def train_gpt(model, train_loader, num_epochs=10):
    """
    训练 GPT 模型
    
    参数:
        model: GPT 模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # 前向传播
            logits = model(input_ids)
            
            # 计算 loss（只计算最后一个 token）
            # GPT 的目标：预测下一个 token
            loss = criterion(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                target_ids[:, 1:].contiguous().view(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")
    
    return model
```

#### 7.2.2 文本生成（推理）

```python
def generate_text(
    model, 
    prompt, 
    max_length=100, 
    temperature=1.0, 
    top_k=50,
    top_p=0.9
):
    """
    生成文本
    
    参数:
        model: GPT 模型
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数（控制随机性）
        top_k: Top-k 采样
        top_p: Top-p（nucleus）采样
    
    返回:
        生成的文本
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]  # 取最后一个 token 的 logits
            
            # Temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Top-k 过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p（nucleus）过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Softmax 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 终止条件（可选）
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# 使用示例
model = GPT(vocab_size=50000)
generated = generate_text(
    model,
    prompt="从前有一座山，",
    max_length=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
print(generated)
```

**采样策略对比**：

| 策略 | 特点 | 适用场景 |
|------|------|---------|
| **Greedy**：选择概率最大的 token | 确定性，容易重复 | 短文本、事实性内容 |
| **Temperature**：缩放概率分布 | T=0.8 平衡，T=1.5 创意 | 平衡创造性与质量 |
| **Top-k**：只从 k 个最高概率中采样 | 避免极端错误 | 通用场景 |
| **Top-p（Nucleus）**：累积概率超过 p 的 token | 动态调整，更灵活 | 通用场景 |

### 7.3 扩散模型（Diffusion Models）

**核心思想**：通过逐步去噪生成高质量图像。

#### 7.3.1 前向扩散（加噪）

**过程**：从干净图像开始，逐步添加高斯噪声，直到变成纯噪声。

```
x_0 (干净图像) → x_1 → x_2 → ... → x_T (纯噪声)

每步加噪：
x_t = sqrt(1 - β_t) * x_{t-1} + sqrt(β_t) * ε
```

其中：
- `β_t`：加噪调度（schedule）
- `ε`：标准高斯噪声

#### 7.3.2 反向扩散（去噪）

**过程**：从纯噪声开始，逐步去噪，最终得到清晰图像。

```
x_T (纯噪声) → x_{T-1} → ... → x_1 → x_0 (干净图像)

每步去噪（由神经网络预测）:
x_{t-1} = (x_t - sqrt(1 - α_t) / sqrt(α_t) * ε_θ(x_t, t)) / sqrt(β_t)
```

其中：
- `α_t = 1 - β_t`
- `ε_θ(x_t, t)`：神经网络预测的噪声

#### 7.3.3 DDPM 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianDiffusion:
    """
    DDPM（Denoising Diffusion Probabilistic Models）
    
    前向扩散：逐步加噪
    反向扩散：逐步去噪（由神经网络预测）
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        # β 调度（加噪强度）
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # α = 1 - β
        self.alphas = 1 - self.betas
        
        # ᾱ = α_1 × α_2 × ... × α_t（累积）
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算一些常用量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：给图像加噪
        
        参数:
            x_start: 原始图像 (B, C, H, W)
            t: 时间步 (B,)
            noise: 噪声（可选）
        
        返回:
            加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def p_sample(self, model, x_t, t):
        """
        反向扩散：去噪一步
        
        参数:
            model: 去噪网络
            x_t: 当前图像 (B, C, H, W)
            t: 时间步 (B,)
        
        返回:
            去噪后的图像
        """
        # 预测噪声
        with torch.no_grad():
            pred_noise = model(x_t, t)
        
        # 计算 x_{t-1}
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # 公式
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        x_t_minus_1 = torch.sqrt(alpha_t) * x_0_pred + torch.sqrt(1 - alpha_t) * pred_noise
        
        return x_t_minus_1


class UNet(nn.Module):
    """
    UNet —— 扩散模型的去噪网络
    
    结构:
        - Encoder（下采样）
        - Bottleneck
        - Decoder（上采样）
        - Skip connections
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_block(in_channels, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels * 2)
        self.enc3 = self._make_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._make_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec4 = self._make_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._make_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._make_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._make_block(base_channels * 2 + base_channels, base_channels)
        
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_block(self, in_channels, out_channels):
        """卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x, t):
        """前向传播
        
        参数:
            x: 图像 (B, C, H, W)
            t: 时间步 (B,)
        """
        # 时间嵌入
        t_emb = self._get_time_embedding(t, x.shape[0])
        
        # Encoder
        e1 = self.enc1(x + t_emb)
        e2 = self.enc2(self.pool(e1) + t_emb)
        e3 = self.enc3(self.pool(e2) + t_emb)
        e4 = self.enc4(self.pool(e3) + t_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4) + t_emb)
        
        # Decoder（带 skip connections）
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1) + t_emb)
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1) + t_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1) + t_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1) + t_emb)
        
        # 输出
        output = self.out_conv(d1)
        return output
    
    def _get_time_embedding(self, t, batch_size):
        """时间嵌入"""
        # 简化版：将时间步映射到 embedding
        # 实际应该使用正弦编码
        t = t.view(batch_size, 1, 1, 1).float() / self.num_timesteps * 2 * math.pi
        t_emb = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        return t_emb.expand(batch_size, -1, x.shape[2], x.shape[3])


def train_ddpm(model, diffusion, train_loader, num_epochs=100):
    """
    训练 DDPM
    
    参数:
        model: UNet 模型
        diffusion: GaussianDiffusion
        train_loader: 训练数据
        num_epochs: 训练轮数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    diffusion = diffusion.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # 随机时间步
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # 前向扩散：加噪
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise)
            
            # 预测噪声
            pred_noise = model(x_t, t)
            
            # 计算 loss（MSE）
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")
    
    return model


def generate_image(model, diffusion, num_steps=1000):
    """
    生成图像（从纯噪声开始）
    
    参数:
        model: 训练好的 UNet
        diffusion: GaussianDiffusion
        num_steps: 去噪步数
    
    返回:
        生成的图像
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 从纯噪声开始
    x = torch.randn(1, 3, 64, 64).to(device)
    
    # 逐步去噪
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t] * x.shape[0], device=device)
            x = diffusion.p_sample(model, x, t_tensor)
    
    return x
```

**DDPM 生成效果**：

```
从纯噪声 (T=1000):
  ████████████████████████

去噪 500 步 (T=500):
  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░

去噪 100 步 (T=100):
  ▓▓▓▓▓░░░░░░░░░░░░░░

最终图像 (T=0):
  一只清晰的猫！
```

### 7.4 变分自编码器（VAE）

**核心思想**：学习数据的潜在空间（Latent Space），从潜在空间采样生成新样本。

#### 7.4.1 VAE 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    变分自编码器（VAE）
    
    结构:
        - Encoder: 编码器（x → μ, σ）
        - Latent Space: 潜在空间（采样 z ~ N(μ, σ²)）
        - Decoder: 解码器（z → x̂）
    
    Loss:
        L = Reconstruction Loss + KL Divergence
    """
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def encode(self, x):
        """编码：x → μ, log(σ²)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧（Reparameterization Trick）
        
        z = μ + σ × ε, ε ~ N(0, I)
        
        使得梯度可以通过采样反向传播
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码：z → x̂"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE 损失函数
    
    Loss = Reconstruction Loss + KL Divergence
    
    参数:
        recon_x: 重建的 x
        x: 原始 x
        mu: 均值 μ
        logvar: 对数方差 log(σ²)
    """
    # Reconstruction Loss（二值交叉熵）
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence
    # KL(N(μ, σ²) || N(0, I)) = -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def train_vae(model, train_loader, num_epochs=10):
    """
    训练 VAE
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)  # Flatten
            
            # 前向传播
            recon_batch, mu, logvar = model(data)
            
            # 计算损失
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum').item()
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
            total_recon_loss += recon_loss
            total_kld_loss += kld_loss
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Recon Loss: {total_recon_loss/len(train_loader):.4f}")
        print(f"  KLD Loss: {total_kld_loss/len(train_loader):.4f}")
    
    return model


def generate_from_vae(model, num_samples=10):
    """
    从潜在空间生成图像
    
    参数:
        model: 训练好的 VAE
        num_samples: 生成数量
    
    返回:
        生成的图像
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(num_samples, 20).to(device)
        
        # 解码
        samples = model.decode(z)
        
        # Reshape 为图像
        samples = samples.view(num_samples, 1, 28, 28)
    
    return samples
```

**VAE 潜在空间可视化**：

```
潜在空间（2D）:

    数字 0          数字 1
        ●
           ●  ●
        ●     ●     ●  数字 2
              ●
                 ●  数字 3
                    ●
                       ●  数字 4
                          ...
```

- 每个数字占据潜在空间的一个区域
- 从两个数字之间采样，会得到中间状态
- 可以通过插值实现平滑过渡

### 7.5 生成对抗网络（GAN）

**核心思想**：生成器和判别器博弈，最终达到纳什均衡。

#### 7.5.1 GAN 基础架构

```
生成器（Generator）:
  随机噪声 z → 生成假图像 x̂

判别器（Discriminator）:
  真图像 x → 判别是否为真
  假图像 x̂ → 判别是否为真

目标:
  生成器: 最大化 D(G(z))  （欺骗判别器）
  判别器: 最大化 D(x) - D(G(z))  （正确判断）
```

#### 7.5.2 DCGAN 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

class Generator(nn.Module):
    """
    DCGAN 生成器
    
    结构:
        - 转置卷积（上采样）
        - BatchNorm
        - ReLU
    """
    
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        参数:
            nz: 噪声向量维度
            ngf: 生成器特征数
            nc: 输出通道数（3=RGB）
        """
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (B, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态: (B, ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态: (B, ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态: (B, ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态: (B, ngf, 32, 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (B, nc, 64, 64)
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """
    DCGAN 判别器
    
    结构:
        - 标准卷积
        - BatchNorm
        - LeakyReLU
    """
    
    def __init__(self, nc=3, ndf=64):
        """
        参数:
            nc: 输入通道数（3=RGB）
            ndf: 判别器特征数
        """
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (B, nc, 64, 64)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (B, ndf, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (B, ndf*2, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (B, ndf*4, 8, 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (B, ndf*8, 4, 4)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: (B, 1, 1, 1) - 概率
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def train_gan(generator, discriminator, train_loader, num_epochs=100, nz=100):
    """
    训练 GAN
    
    参数:
        generator: 生成器
        discriminator: 判别器
        train_loader: 训练数据
        num_epochs: 训练轮数
        nz: 噪声维度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 损失函数（二元交叉熵）
    criterion = nn.BCELoss()
    
    # 固定噪声（用于可视化）
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ===== 训练判别器 =====
            optimizer_D.zero_grad()
            
            # 真图像的标签
            real_labels = torch.ones(batch_size, device=device)
            
            # 假图像的标签
            fake_labels = torch.zeros(batch_size, device=device)
            
            # 判别器对真图像的输出
            output_real = discriminator(real_images)
            loss_real = criterion(output_real, real_labels)
            
            # 生成假图像
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(noise)
            
            # 判别器对假图像的输出
            output_fake = discriminator(fake_images.detach())
            loss_fake = criterion(output_fake, fake_labels)
            
            # 判别器总损失
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            
            # ===== 训练生成器 =====
            optimizer_G.zero_grad()
            
            # 生成器希望判别器认为假图像是真的
            output_fake = discriminator(fake_images)
            loss_G = criterion(output_fake, real_labels)
            
            loss_G.backward()
            optimizer_G.step()
            
            # 打印进度
            if i % 100 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
        
        # 保存生成图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            vutils.save_image(fake, f'generated_{epoch}.png', normalize=True)
    
    return generator, discriminator
```

**GAN 训练稳定性技巧**：

| 技巧 | 说明 |
|------|------|
| **标签平滑**：使用 0.9/0.1 代替 1/0 | 避免判别器过于自信 |
| **WGAN-GP**：使用 Wasserstein 距离 | 梯度更稳定 |
| **谱归一化**：限制判别器的 Lipschitz 常数 | 防止梯度爆炸 |
| **Two-Time-Scale Update (TTUR)**：生成器和判别器使用不同学习率 | 平衡训练速度 |

### 7.6 提示工程（Prompt Engineering）

**核心思想**：通过精心设计提示词，引导模型生成高质量输出。

#### 7.6.1 基础提示技巧

**1. Few-shot Learning（少样本学习）**

```python
# Zero-shot（零样本）
prompt = "翻译成英语：你好"
# 输出可能不准确

# Few-shot（少样本）
prompt = """
翻译成英语：

例子：
你好 → Hello
世界 → World
今天天气很好 → The weather is nice today

任务：你好吗？
"""
# 输出更准确
```

**2. Chain-of-Thought（思维链）**

```python
# 直接提问
prompt = "15 + 27 * 3 = ?"
# 可能出错

# 思维链
prompt = """
请一步步计算：
15 + 27 * 3 = ?

步骤：
1. 先计算乘法：27 * 3 = 81
2. 再计算加法：15 + 81 = ?

答案：
"""
# 准确率大幅提升
```

**3. 角色设定**

```python
prompt = """
你是一位专业的 Python 程序员，擅长算法和数据结构。

任务：写一个快速排序算法。

代码：
"""
# 代码质量更高
```

#### 7.6.2 高级提示技巧

**1. Self-Consistency（自洽性）**

```python
# 多次采样，选择最一致的答案
def self_consistency_generation(prompt, num_samples=5):
    """自洽性生成"""
    answers = []
    for _ in range(num_samples):
        answer = generate_text(prompt, temperature=0.7)
        answers.append(answer)
    
    # 投票选择最频繁的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

**2. ReAct（Reasoning + Acting）**

```python
prompt = """
问题：Python 中如何读取 CSV 文件？

思考（Thought）：
需要使用 pandas 库的 read_csv 函数。

行动（Action）：
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

答案：
使用 pandas 的 read_csv 函数读取 CSV 文件。
"""
```

### 7.7 修炼心得总结

**生成模型对比表**：

| 模型 | 输入 | 输出 | 优势 | 劣势 |
|------|------|------|------|------|
| **GPT** | 文本 | 文本/代码 | 通用性强、效果好 | 计算量大 |
| **DDPM** | 噪声 | 图像 | 高质量、多样性 | 采样慢 |
| **VAE** | 潜在向量 | 图像/文本 | 快速、可解释 | 模糊 |
| **GAN** | 噪声 | 图像 | 速度快、清晰 | 训练不稳定 |

**生成任务总结**：

```
文本生成:
  - 自回归语言模型（GPT）
  - 采样策略：Greedy/Top-k/Top-p/温度采样
  
图像生成:
  - 扩散模型（DDPM、Stable Diffusion）
  - 逐步去噪，高质量
  
潜在空间学习:
  - VAE（变分自编码器）
  - 可控生成、插值
  
对抗生成:
  - GAN（生成对抗网络）
  - 快速生成、清晰图像
  
提示工程:
  - Few-shot Learning
  - Chain-of-Thought
  - 角色设定
```

---

## 修炼总结与境界突破条件

### 本卷回顾

在本卷中，你修炼了三大斗技，实力从**斗者**跃升至**斗师**：

| 斗技 | 核心能力 | 适用场景 | 局限性 |
|------|---------|---------|--------|
| 火眼金睛 (CNN) | 空间模式识别 | 图像分类、目标检测 | 对序列/全局关系弱 |
| 时间感知 (RNN/LSTM) | 序列模式记忆 | 文本生成、时序预测 | 无法并行、长依赖困难 |
| 天道法则 (Transformer) | 全局关联感知 | 几乎一切任务 | 计算量随序列长度平方增长 |
| 闪电神术 (Flash Attention) | 内存高效注意力 | 长序列处理 | 实现复杂 |
| 旋转感知 (RoPE) | 相对位置编码 | 长度外推 | 需要额外计算 |
| 分组灵动 (GQA) | KV Cache 优化 | 推理加速 | 轻微性能损失 |
| 低秩适配 (LoRA) | 轻量级微调 | 任务适配 | 需要基础模型 |

### 核心心法铭记

**CNN 心法**：
- 卷积 = 参数共享的局部感知
- 池化 = 降维 + 增大感受野
- 残差连接 = 解决深层训练的关键

**LSTM 心法**：
- 门控机制 = 选择性记忆与遗忘
- 细胞状态 = 信息高速公路
- 梯度裁剪 = 防止梯度爆炸

**Transformer 心法**（最重要）：
- `Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V` —— 这个公式必须梦中都能写出来
- 多头注意力 = 多角度并行感知
- 位置编码 = 注入序列顺序信息
- 残差 + LayerNorm = 深层训练的基石
- 输入输出维度不变 = 可堆叠的模块化设计

**Flash Attention 心法**：
- **内存瓶颈的根本原因**：需要显式存储 S×S 的注意力矩阵
- **核心解决方案**：分块计算 + 融合核 + 在线 softmax
- **数学精髓**：不需要存储完整的注意力矩阵，只需累积 softmax 结果
- **内存复杂度**：从 O(S²) 降低到 O(S×d_k)
- **修炼境界**：用算力换内存，用精巧的计算图换简单的实现

**RoPE 心法**：
- **绝对位置的问题**：无法表达相对位置关系
- **旋转的优雅**：通过复数旋转编码相对位置
- **数学之美**：Rot(θ₁) × Rot(θ₂) = Rot(θ₁ + θ₂) —— 群性质
- **外推能力**：旋转是连续的，可以自然外推到训练外长度
- **修炼境界**：从固定点到流动的相对关系

**GQA 心法**：
- **KV Cache 瓶颈**：推理时需要存储所有历史 K 和 V
- **共享的智慧**：多个 Q 头可以共享 K/V，损失很小
- **平衡艺术**：在 MHA（32x）和 MQA（1x）之间找到 GQA（8x）
- **推理加速**：KV Cache 减少 4-8 倍，推理速度提升 2-3 倍
- **修炼境界**：合理的共享，而非极致的孤立

**LoRA 心法**：
- **全量微调的沉重**：7B 参数全部更新，需要 40GB 显存
- **低秩的洞察**：ΔW 是低秩的，可以分解为 B @ A
- **增量式学习**：保留预训练知识，只学习任务特定增量
- **参数效率**：0.1-1% 的参数量，达到 95-99% 的性能
- **修炼境界**：在现有基础上轻量级扩展，而非推翻重来

### 境界突破条件（自检清单）

在进入第三卷之前，确保你能做到以下几点：

**基础斗技**：
- [ ] **能画出 CNN 的完整架构图**，标注每一层的输出维度
- [ ] **能解释 LSTM 三个门的作用**，以及为什么它比 RNN 更擅长长期记忆
- [ ] **能手写 Self-Attention 的完整公式**，包括 Q/K/V 投影、缩放、softmax、加权求和
- [ ] **能追踪张量在 Transformer block 中的完整维度变化**
- [ ] **能从零实现一个可运行的 Transformer block**（不看任何参考代码）
- [ ] **能解释 Encoder 和 Decoder 的区别**，以及因果遮罩的作用
- [ ] **能说出 Transformer 相比 RNN 的三个核心优势**

**进阶斗技（新增）**：
- [ ] **能解释传统注意力的内存瓶颈**，并计算 seq_len=4096 时的显存占用
- [ ] **能手写在线 softmax 的简化版**，理解其数学原理
- [ ] **能说明 Flash Attention 的三个核心技术**（分块、融合核、在线 softmax）
- [ ] **能手写 RoPE 的旋转公式**，解释复数旋转如何编码相对位置
- [ ] **能对比绝对位置编码和 RoPE** 的外推能力差异
- [ ] **能手写 GQA 的广播逻辑**（repeat_interleave）
- [ ] **能计算 MHA、GQA、MQA 的 KV Cache 内存占用**差异
- [ ] **能手写 LoRA 线性层的完整实现**，包括初始化策略
- [ ] **能计算 LoRA 相比全量微调的参数量占比**（d_in=4096, d_out=4096, r=16）
- [ ] **能解释 LoRA 为何能有效微调**（ΔW 低秩假设）

如果以上你都能做到，恭喜你，你已经突破到**斗师**之境，并且掌握了天道法则的进阶形态！

### 下一卷预告

> *第三卷：凝晶篇（大斗师）— 万法归一*
>
> 在第二卷中，你掌握了天道法则的三大进阶形态：
> - **Flash Attention**：闪电般的灵力操控，突破内存瓶颈
> - **RoPE**：旋转的相对位置感知，超越绝对位置的局限
> - **GQA**：分组查询的灵动，极致的推理效率
> - **LoRA**：低秩的增量修炼，轻量级的微调神术
>
> 现在，你将学习如何将这些精妙的技术组合，应用于具体的修炼流派：
> - **BERT**：双向感知大法——用 Masked Language Model 理解语言
> - **GPT**：因果推演大法——自回归语言生成，结合 Flash Attention + RoPE + GQA
> - **预训练与微调**：先吸收天地灵气（大规模预训练），再用 LoRA 针对性修炼（下游微调）
> - **分词器（Tokenizer）**：如何将万千灵文分解为模型能理解的基本元素
> - **MLOps 工程实践**：从炼丹房到真实战场——模型部署、监控、A/B 测试
>
> 从理论走向实践，从理解走向应用。真正的修炼，从这里开始。

---

> *"从朴素自注意力到 Flash Attention，从固定编码到 RoPE，从全量微调到 LoRA，*
> *你见证了天道法则的每一次进化。这不仅是技术，更是一种修炼的智慧——*
> *在保留核心思想的同时，不断突破边界，让古老的功法焕发新的生命力。"*
> *继续前进吧，斗师。前方还有更高的境界等待着你。"*
> *——《焚诀》第二卷 尾声*

---

## 本卷增强补全（2026）— 注意力统一视角 · 工程组件 · 稳定性清单

> 本节为《焚诀》深度研究版的**回填内容**，重点补齐：  
> 1) Self-Attention 的三层理解（几何/概率/系统）；2) 现代 LLM 常用组件（RoPE/RMSNorm/SwiGLU/GQA）；3) Transformer 训练稳定性与诊断。  
> 完整增强总览见：[焚诀-深度研究版-全卷增强补全（2026）](../焚诀-深度研究版-全卷增强补全.md)

### 1) Self-Attention 的三层理解（建议作为 Transformer 章节的“开场框架”）

**(1) 几何视角：相似度检索（Query-Key）**  
把 Self-Attention 看成一个“可微分检索器”：  
\\[
A=\\text{softmax}(QK^\\top/\\sqrt{d}),\\quad \\text{Attn}(Q,K,V)=AV
\\]  
直觉：每个 Query 在 Key 空间里找“最像”的位置，再把对应 Value 加权求和。

**(2) 概率视角：对 Value 的条件期望**  
注意力权重矩阵 \\(A\\) 可视为对位置的离散分布，输出是对 \\(V\\) 的加权期望（因此易做对齐/可解释分析）。

**(3) 系统视角：长上下文的瓶颈是 IO（读写/带宽），不是算术**  
即使 FLOPs 足够，\\(O(L^2)\\) 的内存访问和 KV cache 才是工程瓶颈；这也是 FlashAttention 系列价值所在。

### 2) 现代 LLM 的“常用组件层”补齐（建议补到卷末或附录）

- **RoPE（旋转位置编码）**：相对位置感知更强，外推与长上下文更常用  
- **RMSNorm**：现代 LLM 常见归一化选择（更轻量）  
- **SwiGLU/GeGLU**：FFN 激活主流方案（更强表达）  
- **GQA/MQA**：显著降低 KV cache 成本，是推理侧提效关键  

### 3) Transformer 训练稳定性与诊断清单（Best Practices）

- [ ] **学习率策略**：warmup + cosine decay（与 AdamW 搭配）  
- [ ] **梯度裁剪**：尤其在长序列/小数据场景  
- [ ] **监控注意力熵**：熵过低可能意味着注意力塌缩（过于尖锐）  
- [ ] **监控 grad norm 与激活分布**：用于定位爆炸/消失  
- [ ] **对齐到真实任务**：分类/生成/检索三类任务应分别设置验证集与指标
