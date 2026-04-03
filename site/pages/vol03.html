# 第三卷：凝晶篇（大斗师）— 斗气固化

> *"斗气由气态凝为液态，再由液态化为晶体，方为大斗师之境。"*
>
> *"Pre-training is the crystallization of knowledge — turning raw compute into reusable intelligence."*

---

## 开篇引言：从气态到晶体

修炼至此，你已掌握三大斗技——卷积神经之术（CNN）、循环记忆之术（RNN/LSTM）、以及天阶斗技 Transformer。

你的斗气已然充沛，能以 CNN 识图辨物，能以 RNN 解析序列，能以 Transformer 捕捉万物关联。然而，此刻你的斗气仍是**气态**的——每次战斗（每个任务），你都需要从零开始凝聚力量，从随机初始化的参数出发，在特定数据上重新训练。

这就如同一位武者，虽然知晓拳法套路，但每次上阵前都要重新修炼基本功。这不仅效率低下，更无法应对千变万化的实战。

**大斗师之境的本质，是将斗气从气态凝聚为晶体。**

在技术世界里，这意味着掌握**预训练（Pre-training）**的奥义：

- 在海量文本上训练模型，使其**内化语言的结构、语义和世界知识**
- 将这些知识**固化为模型参数**（斗气结晶），形成可复用的基座
- 此后面对任何下游任务，只需在结晶基础上**微调（Fine-tuning）**，便能快速适应

这便是从 "task-specific training" 到 "pre-train then fine-tune" 范式的跃迁——NLP 乃至整个 AI 领域最深刻的变革之一。

本卷你将修炼：

| 章节 | 修炼内容 | 对应技术 |
|------|---------|---------|
| 第一章 | 文字炼化 | Tokenizer 原理与构建 |
| 第二章 | 斗气空间 | Embedding 的本质 |
| 第三章 | 双面法则 | BERT 预训练原理 |
| 第四章 | 因果法则 | GPT 预训练原理 |
| 第五章 | 首次凝晶 | 从零训练文本模型 |

**修炼目标**：在小型数据集上，从零完成一个文本生成模型的训练，生成连贯文本。

**前置要求**：已完成第二卷（纳灵篇），熟练掌握 Transformer 架构。

---

## 第一章：文字炼化 — Tokenizer 的原理与构建

> *"药材入炉之前，必先研磨成标准大小的粉末，方能均匀受热，充分反应。"*

### 1.1 为什么需要 Tokenizer？

神经网络只能处理数字，而人类的语言是由文字组成的。Tokenizer（分词器）就是连接人类语言与模型世界的桥梁——它将文本切割成一个个 **token**（词元），并将每个 token 映射为一个整数 ID。

这就如同炼丹之前的**药材预处理**：

- 原始药材（原始文本）形状各异，大小不一
- 必须将其研磨、切割成**标准规格**（token）
- 每种标准规格的药材都有一个**编号**（token ID）
- 丹炉（模型）只接受编号后的标准药材

```
原始文本: "深度学习改变世界"
         ↓ Tokenizer
Token序列: ["深度", "学习", "改变", "世界"]
         ↓ 查表
ID序列:   [1024,  2048,  3072,  4096]
         ↓ 送入模型
模型处理数字序列
```

**一个好的 Tokenizer 需要平衡以下矛盾**：

| 粒度 | 优点 | 缺点 |
|------|------|------|
| 字符级（Character） | 词汇表极小，无未登录词 | 序列极长，难以捕获语义 |
| 词级（Word） | 语义清晰 | 词汇表巨大，未登录词（OOV）问题严重 |
| 子词级（Subword） | 平衡词汇表大小与语义 | 需要训练算法来确定切分方式 |

现代主流模型几乎都采用**子词级（Subword）**分词，它是字符级和词级的完美折中。

### 1.2 BPE：字节对编码 — 最主流的子词算法

**Byte Pair Encoding（BPE）**是目前最广泛使用的子词分词算法，被 GPT 系列、LLaMA、Mistral 等主流模型采用。

其核心思想极其优雅：**从最小的字符单元出发，不断合并出现频率最高的相邻对，直到达到目标词汇表大小。**

用修炼的比喻来说：

> *"从散碎的灵气微粒（单个字符）出发，将最常共同出现的微粒凝聚在一起（合并高频对），反复凝聚，直到形成一颗颗标准大小的灵气结晶（子词 token）。"*

#### BPE 算法详解

**训练阶段**（学习合并规则）：

```
输入：训练语料
输出：合并规则列表 + 词汇表

步骤：
1. 将所有单词拆分为字符序列，并统计词频
2. 统计所有相邻字符对的出现频率
3. 找到频率最高的字符对，将其合并为新符号
4. 更新所有序列，重复步骤 2-3
5. 直到词汇表达到目标大小，停止
```

**具体示例**：

假设我们的训练语料产生了如下词频统计：

```
"low"    : 5次
"lower"  : 2次
"newest" : 6次
"widest" : 3次
```

**第 0 步**：初始化，将每个词拆分为字符（加上词尾标记 `</w>`）：

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w e s t </w>: 6
w i d e s t </w>: 3
```

初始词汇表：`{l, o, w, e, r, n, s, t, i, d, </w>}`

**第 1 步**：统计所有相邻对的频率：

```
(e, s) : 6 + 3 = 9  ← 最高频！
(s, t) : 6 + 3 = 9
(l, o) : 5 + 2 = 7
(o, w) : 5 + 2 = 7
(t, </w>) : 6 + 3 = 9
...
```

频率最高的对有多个（并列时取第一个遇到的），假设合并 `(e, s)` → `es`：

```
l o w </w>       : 5
l o w e r </w>   : 2
n e w es t </w>  : 6
w i d es t </w>  : 3
```

词汇表新增：`es`

**第 2 步**：重新统计，合并 `(es, t)` → `est`：

```
l o w </w>        : 5
l o w e r </w>    : 2
n e w est </w>    : 6
w i d est </w>    : 3
```

**第 3 步**：合并 `(est, </w>)` → `est</w>`：

```
l o w </w>        : 5
l o w e r </w>    : 2
n e w est</w>     : 6
w i d est</w>     : 3
```

**第 4 步**：合并 `(l, o)` → `lo`：

```
lo w </w>         : 5
lo w e r </w>     : 2
n e w est</w>     : 6
w i d est</w>     : 3
```

如此反复，直到词汇表达到预设大小（如 32000、50000）。

**编码阶段**（对新文本分词）：

```
1. 将输入文本拆分为字符
2. 按训练阶段学到的合并规则，依次应用每条规则
3. 最终得到子词序列
```

例如，对 "lowest" 进行编码：
- 初始：`l o w e s t </w>`
- 应用规则 `(e,s)→es`：`l o w es t </w>`
- 应用规则 `(es,t)→est`：`l o w est </w>`
- 应用规则 `(est,</w>)→est</w>`：`l o w est</w>`
- 应用规则 `(l,o)→lo`：`lo w est</w>`
- 应用规则 `(lo,w)→low`：`low est</w>`
- 结果：`["low", "est</w>"]`

即便 "lowest" 从未在训练语料中出现，BPE 也能将其合理切分为已知子词！

### 1.3 WordPiece：BERT 的选择

**WordPiece** 是 Google 开发的子词算法，被 BERT、DistilBERT 等模型采用。它与 BPE 非常相似，但合并策略不同：

| | BPE | WordPiece |
|--|-----|-----------|
| 合并标准 | 频率最高的对 | 使语言模型似然度提升最大的对 |
| 数学表达 | `argmax count(pair)` | `argmax P(merged) / (P(a) * P(b))` |
| 标记方式 | 无特殊标记 | 非词首子词加 `##` 前缀 |

WordPiece 的 `##` 前缀是个优雅的设计：

```python
# WordPiece 分词示例
"unaffable" → ["un", "##aff", "##able"]
"playing"   → ["play", "##ing"]
```

`##` 表示 "这个子词是前一个词的延续，不是独立的词"。这让模型在解码时能准确地还原原文。

### 1.4 SentencePiece 与 Unigram

**SentencePiece** 是 Google 开发的开源分词工具，特点是：

- **语言无关**：不依赖预分词（pre-tokenization），直接在原始文本上操作
- **将空格也视为普通字符**（用 `▁` 表示），对中文、日文等无空格语言特别友好
- 支持 BPE 和 Unigram 两种算法

**Unigram** 算法与 BPE 的方向相反：

```
BPE：    从小到大，不断合并    （自底向上）
Unigram：从大到小，不断删减    （自顶向下）
```

Unigram 的步骤：
1. 初始化一个很大的候选词汇表
2. 对每个候选子词计算其对训练数据似然度的贡献
3. 移除贡献最小的子词（保留一定比例）
4. 重复步骤 2-3，直到词汇表达到目标大小

LLaMA、T5 等模型使用 SentencePiece + Unigram/BPE 的组合。

### 1.5 特殊 Token：阵法标记

每个 Tokenizer 都有一组**特殊 token**，它们不代表实际文字，而是起**结构标记**的作用——如同阵法中的关键节点：

```python
# BERT 的特殊 token
[CLS]   # 分类标记 — 整个序列的代表，如同阵眼
[SEP]   # 分隔标记 — 区分两个句子，如同阵法分界线
[PAD]   # 填充标记 — 补齐序列长度，如同阵法空位
[MASK]  # 遮蔽标记 — MLM 任务中被隐藏的位置
[UNK]   # 未知标记 — 词汇表外的词（理想情况下不应出现）

# GPT 的特殊 token
<|endoftext|>  # 文本结束标记
<|pad|>        # 填充标记

# 通用特殊 token
<bos>   # 序列开始（Beginning of Sequence）
<eos>   # 序列结束（End of Sequence）
```

在 BERT 中，一个典型的输入序列长这样：

```
[CLS] 今天 天气 真 好 [SEP] 我们 去 公园 吧 [SEP] [PAD] [PAD]
  ↑                    ↑                    ↑      ↑
 阵眼              句子分界线           句子结束   空位填充
```

### 1.6 实战：训练一个 BPE Tokenizer

现在，让我们亲手炼制一个 BPE Tokenizer。我们使用 HuggingFace 的 `tokenizers` 库，它由 Rust 编写，速度极快。

```python
"""
=== 丹方 3.1：训练 BPE Tokenizer ===
药材：任意文本语料
丹炉：CPU 即可
火候：无需 GPU
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ============================
# 第一步：准备药材（训练语料）
# ============================
# 实际应用中，这应该是大量文本文件
# 这里用少量示例演示
sample_texts = [
    "深度学习是人工智能的核心技术之一",
    "自然语言处理让计算机理解人类语言",
    "Transformer 架构改变了整个深度学习领域",
    "预训练模型通过大规模语料学习语言知识",
    "BERT 和 GPT 是两种经典的预训练范式",
    "分词器将文本转换为模型能处理的数字序列",
]

# 将样本写入临时文件
with open("/tmp/train_corpus.txt", "w", encoding="utf-8") as f:
    for text in sample_texts:
        f.write(text + "\n")

# ============================
# 第二步：配置丹炉（初始化 Tokenizer）
# ============================
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 预分词器：先按空格初步切分
tokenizer.pre_tokenizer = Whitespace()

# ============================
# 第三步：调配丹方（设置训练参数）
# ============================
trainer = BpeTrainer(
    vocab_size=1000,              # 词汇表大小
    min_frequency=1,              # 最低出现频率
    special_tokens=[              # 特殊 token（阵法标记）
        "[UNK]",   # 未知
        "[CLS]",   # 分类
        "[SEP]",   # 分隔
        "[PAD]",   # 填充
        "[MASK]",  # 遮蔽
    ],
    show_progress=True,
)

# ============================
# 第四步：开炉炼制（训练 Tokenizer）
# ============================
tokenizer.train(files=["/tmp/train_corpus.txt"], trainer=trainer)
print(f"词汇表大小：{tokenizer.get_vocab_size()}")

# ============================
# 第五步：验证成果（测试分词效果）
# ============================
test_text = "深度学习模型需要大规模预训练"
output = tokenizer.encode(test_text)

print(f"\n原始文本：{test_text}")
print(f"Token 序列：{output.tokens}")
print(f"Token ID：{output.ids}")

# 解码还原
decoded = tokenizer.decode(output.ids)
print(f"解码还原：{decoded}")

# ============================
# 第六步：保存丹方（保存 Tokenizer）
# ============================
tokenizer.save("/tmp/my_bpe_tokenizer.json")
print("\nTokenizer 已保存！可随时加载复用。")

# 加载已保存的 Tokenizer
loaded_tokenizer = Tokenizer.from_file("/tmp/my_bpe_tokenizer.json")
```

#### 使用 HuggingFace tokenizers 的高级 API

在实际项目中，更常使用 `transformers` 库的 `AutoTokenizer`：

```python
"""
=== 使用预训练 Tokenizer ===
"""
from transformers import AutoTokenizer

# 加载 BERT 的 Tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "深度学习改变了世界"
tokens = bert_tokenizer.tokenize(text)
ids = bert_tokenizer.encode(text)

print(f"BERT 分词结果：{tokens}")
print(f"Token IDs：{ids}")
# BERT 中文版是字级别分词：['深', '度', '学', '习', '改', '变', '了', '世', '界']

# 加载 GPT-2 的 Tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

text_en = "Deep learning has transformed the world"
tokens = gpt2_tokenizer.tokenize(text_en)
ids = gpt2_tokenizer.encode(text_en)

print(f"\nGPT-2 分词结果：{tokens}")
print(f"Token IDs：{ids}")
# GPT-2 使用 BPE：['Deep', ' learning', ' has', ' transformed', ' the', ' world']
```

### 1.7 避坑符：Tokenizer 常见陷阱

> **避坑符 3.1**：Tokenizer 不匹配
>
> 使用预训练模型时，**必须使用与其配套的 Tokenizer**。用 BERT 的 Tokenizer 处理文本后送入 GPT 模型，就像把药材按 A 方剂的规格研磨后塞进 B 方剂的丹炉——必然反噬。

> **避坑符 3.2**：中文分词的特殊性
>
> 英文天然有空格作为分词线索，中文没有。BERT-Chinese 采用字级别分词（每个汉字一个 token），而 GPT 系列的 BPE 可能将多个汉字合并为一个 token，也可能将一个汉字拆成多个 bytes。务必检查你的 Tokenizer 对中文的处理方式。

> **避坑符 3.3**：词汇表大小的权衡
>
> 词汇表越大，每个 token 的信息量越大，序列越短，但 Embedding 层参数量也越大。词汇表越小，序列越长（增加计算量），但参数更少。常见大小：BERT 30522，GPT-2 50257，LLaMA 32000。

---

## 第二章：斗气空间 — Embedding 的本质

> *"凡间众生以肉眼观万物，修炼者却能以灵魂感知开辟一方精神空间，在其中，相似的事物自然靠近，对立的事物彼此远离。"*

### 2.1 从离散到连续：为什么需要 Embedding？

经过 Tokenizer 处理后，文本变成了整数 ID 序列。但整数 ID 本身没有语义——ID 1024 和 ID 1025 之间没有 "相似" 或 "不同" 的概念，它们只是编号。

我们需要将每个 token 映射到一个**连续向量空间**中，使得：

- **语义相似的词，向量距离近**（"猫" 和 "狗" 应该比 "猫" 和 "火箭" 更近）
- **语义关系可以通过向量运算表达**（"国王" - "男人" + "女人" ≈ "女王"）

这个映射过程就是 **Embedding**——将离散的 token ID 映射为稠密的连续向量。

```
Token ID: 1024
    ↓ 查 Embedding 表（本质是矩阵的第 1024 行）
向量: [0.12, -0.34, 0.56, ..., 0.78]  (维度通常为 768 或 1024)
```

### 2.2 One-Hot vs Dense Embedding

**One-Hot 编码**是最朴素的表示方式：

```
假设词汇表大小 V = 5: [猫, 狗, 鱼, 树, 花]

猫 → [1, 0, 0, 0, 0]
狗 → [0, 1, 0, 0, 0]
鱼 → [0, 0, 1, 0, 0]
树 → [0, 0, 0, 1, 0]
花 → [0, 0, 0, 0, 1]
```

One-Hot 的致命缺陷：
1. **维度灾难**：词汇表有 50000 个词，每个向量就是 50000 维，极度稀疏
2. **无语义信息**：任何两个 One-Hot 向量的内积都是 0，"猫" 和 "狗" 的距离与 "猫" 和 "火箭" 完全相同
3. **无法泛化**：不能表达词与词之间的关系

**Dense Embedding** 将高维稀疏向量压缩为低维稠密向量：

```
猫 → [0.82, -0.15, 0.44, ..., 0.31]   (768维)
狗 → [0.79, -0.12, 0.41, ..., 0.28]   (768维)  ← 和"猫"很接近！
火箭 → [-0.53, 0.87, -0.21, ..., 0.66] (768维)  ← 和"猫"很远
```

这就如同将散乱的原始灵气（One-Hot，稀疏、无序）压缩凝练为精纯斗气（Dense Embedding，紧凑、有序）。

### 2.3 Word2Vec：Embedding 的先驱

在 Transformer 和预训练大模型之前，**Word2Vec**（2013, Mikolov et al.）是 Embedding 领域最重要的工作。它提出了两种训练方法：

#### Skip-gram：由中心词预测上下文

```
训练目标：给定中心词，预测其周围的词

句子："the cat sat on the mat"
窗口大小=2

中心词 "sat" 的训练样本：
  sat → the  (左2)
  sat → cat  (左1)
  sat → on   (右1)
  sat → the  (右2)
```

> *"给你一味核心药材，推断它通常与哪些药材搭配使用。"*

#### CBOW (Continuous Bag of Words)：由上下文预测中心词

```
训练目标：给定周围的词，预测中心词

上下文 [the, cat, on, the] → 预测 "sat"
```

> *"给你一堆配方中的辅料，推断核心药材是什么。"*

Word2Vec 的训练过程本质上是在训练一个浅层神经网络，训练完成后，取其中的权重矩阵作为词向量。

### 2.4 Embedding 的魔法：向量算术

Word2Vec 最令人震撼的发现是，训练得到的词向量居然能进行有意义的**向量运算**：

```
king - man + woman ≈ queen
（国王 - 男人 + 女人 ≈ 女王）

Paris - France + Italy ≈ Rome
（巴黎 - 法国 + 意大利 ≈ 罗马）

bigger - big + small ≈ smaller
（更大 - 大 + 小 ≈ 更小）
```

这意味着 Embedding 空间自动学会了**语义关系的向量化表示**：

- `king - man` 提取了 "皇室" 的概念向量
- 将 "皇室" 向量加到 `woman` 上，得到 "女性皇室成员" = `queen`

> *"这如同发现了灵气运行的深层法则——不同属性的斗气之间存在精确的数学关系。火属性斗气减去'热'的特征，加上'冷'的特征，就变成了冰属性斗气。"*

```python
"""
=== 丹方 3.2：探索 Word2Vec Embedding ===
"""
import gensim.downloader as api

# 加载预训练的 Word2Vec 模型
model = api.load('word2vec-google-news-300')

# 向量算术：king - man + woman = ?
result = model.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=3
)
print("king - man + woman ≈")
for word, score in result:
    print(f"  {word}: {score:.4f}")
# 输出：queen: 0.7118

# 查看相似词
print("\n与 'python' 最相似的词：")
for word, score in model.most_similar('python', topn=5):
    print(f"  {word}: {score:.4f}")

# 计算两个词的相似度
similarity = model.similarity('cat', 'dog')
print(f"\ncat 与 dog 的相似度：{similarity:.4f}")

similarity2 = model.similarity('cat', 'rocket')
print(f"cat 与 rocket 的相似度：{similarity2:.4f}")
```

### 2.5 静态 Embedding 的局限与上下文 Embedding

Word2Vec 的 Embedding 是**静态的**——每个词无论出现在什么上下文中，都是同一个向量。这带来一个根本问题：**一词多义**。

```
"我去银行取钱"    → 银行 = 金融机构
"长江银行两岸风景如画" → 银行 = 不对，这里是"岸"才对；但如果是：
"我坐在河bank上"  → bank = 河岸（英文示例更明显）

静态 Embedding 中，"bank" 只有一个向量，无法区分这两种含义。
```

这就是为什么我们需要**上下文相关的 Embedding（Contextual Embedding）**——BERT、GPT 等预训练模型产生的 Embedding 会根据上下文动态变化：

```python
# BERT 的上下文 Embedding（概念示例）
sentence1 = "I went to the bank to deposit money"
sentence2 = "I sat on the river bank"

# 同一个词 "bank"，在两个句子中会得到不同的向量！
# 因为 BERT 的每一层都在做 Attention，融合了上下文信息
embedding1_bank ≠ embedding2_bank
```

> *"静态 Embedding 如同一本固定的药材图鉴——'银花'永远是同一种描述。而上下文 Embedding 如同一位经验丰富的药师，他会根据整个药方来判断这味'银花'在此处的具体药性。"*

### 2.6 Position Embedding：编码序列顺序

Transformer 架构中没有 RNN 的循环结构，天然不知道 token 的顺序。因此需要**位置编码（Position Embedding）**来注入序列信息。

主流方案有两种：

**1. 正弦位置编码（Sinusoidal，原始 Transformer）**：

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

特点：固定的数学公式生成，不需要学习参数，理论上可以泛化到任意长度。

**2. 可学习位置编码（Learnable，BERT/GPT）**：

```python
# PyTorch 实现
self.position_embeddings = nn.Embedding(max_position, hidden_size)
# 将位置 0, 1, 2, ..., max_position-1 各映射为一个向量
# 这些向量通过训练学习得到
```

特点：参数通过训练学习，通常效果更好，但受限于训练时的最大长度。

**BERT 的完整输入 Embedding**：

```
最终 Embedding = Token Embedding + Position Embedding + Segment Embedding

Token Embedding:    [CLS]  今天  天气  真  好  [SEP]  我们  去  公园  吧  [SEP]
                      ↓     ↓    ↓   ↓  ↓    ↓     ↓   ↓    ↓   ↓    ↓
Position Embedding:   0     1    2   3  4    5     6   7    8   9   10
                      ↓     ↓    ↓   ↓  ↓    ↓     ↓   ↓    ↓   ↓    ↓
Segment Embedding:    A     A    A   A  A    A     B   B    B   B    B
                      ↓     ↓    ↓   ↓  ↓    ↓     ↓   ↓    ↓   ↓    ↓
                    三者相加，得到最终输入向量
```

### 2.7 Embedding 可视化：灵气空间的投影

高维 Embedding 空间无法直接观察，但我们可以用降维技术将其投影到 2D/3D：

```python
"""
=== 丹方 3.3：Embedding 可视化 ===
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设我们有一些词和它们的 Embedding 向量
# 实际使用时从模型中提取
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'cat', 'dog', 'fish', 'bird',
         'python', 'java', 'code', 'program']

# 从预训练模型获取向量（此处用随机向量示意）
# 实际中：vectors = [model[w] for w in words]
np.random.seed(42)
vectors = np.random.randn(len(words), 300)

# t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_2d = tsne.fit_transform(vectors)

# 可视化
plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
    x, y = vectors_2d[i]
    plt.scatter(x, y, marker='o', s=100)
    plt.annotate(word, (x, y), fontsize=12,
                 xytext=(5, 5), textcoords='offset points')

plt.title('Word Embedding Space (t-SNE Projection)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('embedding_visualization.png', dpi=150)
plt.show()

print("在真实的预训练 Embedding 中，你会看到：")
print("  - 皇室词汇（king, queen, prince）聚成一簇")
print("  - 动物词汇（cat, dog, fish）聚成一簇")
print("  - 编程词汇（python, java, code）聚成一簇")
```

> *"t-SNE 如同一面灵镜，将高维灵气空间的结构映射到凡人可见的二维平面上。虽然有所失真，但依然能展现斗气空间的大致格局。"*

---

## 第三章：双面法则 — BERT 预训练原理

> *"双面法则，同修阴阳。左手过去，右手未来，双向感知，方能洞察全局。"*

### 3.1 BERT 的诞生与意义

**BERT**（Bidirectional Encoder Representations from Transformers）由 Google 于 2018 年发布，开创了 "Pre-train then Fine-tune" 的范式，堪称 NLP 历史上的分水岭。

在 BERT 之前，NLP 模型要么：
- 从零训练（耗时耗力，小数据易过拟合）
- 使用静态词向量（Word2Vec/GloVe）作为初始化（只有浅层语义）

BERT 证明了：**在大规模语料上无监督预训练，然后在下游任务上微调**，可以在几乎所有 NLP 任务上取得巨大提升。

> *"在 BERT 之前，每个修炼者都要从零开始修炼基础功法。BERT 的出现如同一部万能基础功法——先用海量灵气（数据）打通全身经脉（学习语言知识），然后只需少量修炼（微调）就能精通任何具体招式（下游任务）。"*

### 3.2 架构：纯 Encoder 结构

BERT 使用的是 **Transformer Encoder** 堆叠：

```
BERT-Base:  12 层 Encoder, 768 hidden, 12 heads,  110M 参数
BERT-Large: 24 层 Encoder, 1024 hidden, 16 heads, 340M 参数
```

关键点：**BERT 是双向的**——每个位置都能看到序列中所有其他位置（包括前面和后面）。

```
输入:  [CLS] 我 爱 北京 天安门 [SEP]

       [CLS] ←→ 我 ←→ 爱 ←→ 北京 ←→ 天安门 ←→ [SEP]
       ↑  所有位置之间都有 Attention 连接（双向）
```

这与 GPT 的单向（只能看到前面）形成鲜明对比。

### 3.3 预训练任务一：Masked Language Modeling (MLM)

MLM 是 BERT 最核心的预训练任务，灵感来自完形填空：

> *"将药方中随机涂抹掉 15% 的药材名称，让学徒根据上下文猜测被隐藏的药材。如果学徒能准确猜出，说明他真正理解了药方的逻辑。"*

#### 具体策略

对输入序列中 **15%** 的 token 进行操作，但并非全部替换为 [MASK]：

```
被选中的 15% token：
  - 80% 的概率 → 替换为 [MASK]
  - 10% 的概率 → 替换为随机 token
  - 10% 的概率 → 保持不变

示例：
原句：  "我 爱 北京 天安门"
选中 "北京"（15%的其中一个）

80% 情况：  "我 爱 [MASK] 天安门"  → 模型要预测 [MASK] = "北京"
10% 情况：  "我 爱 香蕉   天安门"  → 模型要预测 "香蕉" 位置原本是 "北京"
10% 情况：  "我 爱 北京   天安门"  → 模型仍需输出 "北京"（但输入就是正确的）
```

**为什么不全部用 [MASK]？**

如果总是用 [MASK]，模型会产生**预训练-微调不匹配**问题——在微调时，输入中没有 [MASK] 标记，模型会困惑。10% 随机替换和 10% 不变，迫使模型**对每个位置都保持警觉**，不能只关注 [MASK] 位置。

```python
"""
=== MLM 训练示意 ===
"""
import torch
import torch.nn as nn

def create_mlm_data(tokens, tokenizer, mlm_prob=0.15):
    """
    创建 MLM 训练数据
    如同准备完形填空的试卷
    """
    labels = tokens.clone()
    
    # 创建概率矩阵，决定哪些位置被选中
    probability_matrix = torch.full(tokens.shape, mlm_prob)
    
    # 特殊 token 不参与 masking
    special_tokens_mask = [
        1 if tok in [tokenizer.cls_token_id, 
                     tokenizer.sep_token_id, 
                     tokenizer.pad_token_id] else 0
        for tok in tokens.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), 
        value=0.0
    )
    
    # 随机选择被 mask 的位置
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 只计算被 mask 位置的 loss
    
    # 80% 替换为 [MASK]
    indices_replaced = (torch.bernoulli(
        torch.full(tokens.shape, 0.8)
    ).bool() & masked_indices)
    tokens[indices_replaced] = tokenizer.mask_token_id
    
    # 10% 替换为随机 token
    indices_random = (torch.bernoulli(
        torch.full(tokens.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced)
    random_words = torch.randint(
        len(tokenizer), tokens.shape, dtype=torch.long
    )
    tokens[indices_random] = random_words[indices_random]
    
    # 剩余 10% 保持不变
    
    return tokens, labels
```

### 3.4 预训练任务二：Next Sentence Prediction (NSP)

NSP 是 BERT 的第二个预训练任务：

> *"给学徒两段药方文字，让他判断这两段是否来自同一张药方——即它们是否具有逻辑连续性。"*

```
正样本（IsNext, 50%）：
  [CLS] 今天天气很好 [SEP] 我们一起去公园散步吧 [SEP]
  标签：IsNext（连贯的两句话）

负样本（NotNext, 50%）：
  [CLS] 今天天气很好 [SEP] 量子力学是现代物理的基石 [SEP]
  标签：NotNext（随机拼凑的两句话）
```

NSP 的目的是让模型理解**句子间的关系**，这对问答、推理等任务有帮助。

> **注**：后来的研究（RoBERTa 等）发现 NSP 的效果有争议，RoBERTa 直接移除了 NSP，仅保留 MLM，性能反而更好。但理解 NSP 的思路对掌握预训练范式仍有重要价值。

### 3.5 BERT 的预训练数据与规模

```
BERT 预训练数据：
  - BooksCorpus: 8亿词
  - English Wikipedia: 25亿词
  - 总计约 33亿词

训练配置（BERT-Base）：
  - Batch Size: 256
  - 序列长度: 512
  - 训练步数: 1,000,000 步
  - 学习率: 1e-4（Adam with warmup）
  - 硬件: 4 个 Cloud TPU Pod（16 个 TPU 芯片）
  - 训练时间: 4 天
```

> *"BERT 的预训练如同在一座灵药山脉中闭关修炼——需要吞噬海量灵气（33亿词），经过百万次吐纳（训练步数），方能将语言知识凝聚为稳固的斗气结晶。"*

### 3.6 Fine-tuning：从通用到专精

预训练完成后，BERT 拥有了丰富的语言知识。接下来，只需在其基础上添加少量任务特定的层，用少量标注数据微调，就能适应各种下游任务：

```
              预训练 BERT（通用语言知识）
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   文本分类      命名实体识别     问答系统
  (+ 分类头)   (+ 序列标注头)  (+ Span 预测头)
   
[CLS] → 分类    每个token → 标签  Start/End → 答案
```

> *"通用功法修炼完毕后，只需少量针对性修炼（微调），就能转化为具体招式——分类术、识别术、问答术。根基越牢固（预训练越好），转化越快（微调所需数据越少）。"*

### 3.7 实战：用 BERT 做文本分类

```python
"""
=== 丹方 3.4：BERT 文本分类微调 ===
药材：文本分类数据集
丹炉：单张 GPU（推荐 ≥ 8GB 显存）
火候：学习率 2e-5, Batch Size 16, Epochs 3
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split

# ============================
# 第一步：准备药材（数据集）
# ============================
class TextClassificationDataset(Dataset):
    """文本分类数据集 — 标准化后的药材"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 示例数据（实际应使用真实数据集）
texts = [
    "这部电影太精彩了，强烈推荐！",
    "剧情拖沓，浪费时间",
    "演员演技在线，导演功力深厚",
    "特效粗糙，剧本毫无逻辑",
    "年度最佳影片，没有之一",
    "看完想退票，太烂了",
] * 50  # 扩展数据量

labels = [1, 0, 1, 0, 1, 0] * 50  # 1=正面, 0=负面

# ============================
# 第二步：加载预训练模型（载入基础功法）
# ============================
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # 二分类
)

# ============================
# 第三步：准备训练（配置火候）
# ============================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = TextClassificationDataset(
    train_texts, train_labels, tokenizer
)
val_dataset = TextClassificationDataset(
    val_texts, val_labels, tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * 3  # 3 个 epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ============================
# 第四步：开炉炼制（训练循环）
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    # --- 训练阶段 ---
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    
    # --- 验证阶段 ---
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_val = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels_val).sum().item()
            total += labels_val.size(0)
    
    accuracy = correct / total
    print(f"Epoch {epoch+1}/3 | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}")

print("\n微调完成！BERT 已从通用功法特化为情感分析专家。")
```

---

## 附录 A：斗气进化论 — 预训练模型族谱与演进

> *"双面法则（BERT）横空出世后，天下修炼者纷纷效仿，衍生出众多进化功法。这些功法各有所长，或弥补双面法则的缺陷，或开拓全新的修炼路径。修炼者需知其然，更知其所以然。"*

### A.1 RoBERTa：稳健改良 — 去芜存菁

**RoBERTa**（Robustly Optimized BERT Pretraining Approach，2019，Facebook）并不改变 BERT 的架构，而是通过**精细的训练策略优化**大幅提升性能。

> *"双面法则的招式不变，但修炼方式更科学——更深的闭关、更多的灵药、更严苛的修炼纪律。"*

#### 关键改进

| 改进点 | BERT | RoBERTa | 效果 |
|--------|------|---------|------|
| 预训练数据 | 16GB (BooksCorpus + Wikipedia) | 160GB (含 CC-News, OpenWebText, Stories) | **数据量 ×10** |
| 训练步数 | 1M steps (100K tokens/step) | 500K steps (2K tokens/step) | 更长训练 |
| NSP 任务 | 保留 | **移除** | 简化目标 |
| MLM 策略 | 静态 mask (预处理) | **动态 mask** (每次送入模型时重新 mask) | 防止记忆 |
| BPE 词汇表 | WordPiece 30K | **Byte-level BPE** 50K + 250K chars | 更细粒度 |
| Batch Size | 256 | **8K** | 更大 Batch |
| 学习率 | 1e-4 | 1e-4 (相同) | — |

#### 动态 Masking 详解

BERT 的静态 masking 在预处理阶段就确定了哪些位置被 mask，整个 epoch 内不变。RoBERTa 的动态 masking **在每个 epoch 对同一份数据重新随机 mask**：

```python
"""
=== 丹方 3.A.1：动态 Masking 实现 ===
"""
import torch
import torch.nn.functional as F

def dynamic_mask(inputs, tokenizer, mask_prob=0.15):
    """
    动态 Masking — 每次调用都产生不同的 mask 结果
    如同每次修炼前都重新布置迷阵，防止学徒死记硬背
    """
    labels = inputs.clone()

    # 创建 mask 矩阵
    probability_matrix = torch.full(inputs.shape, mask_prob)

    # 特殊 token 不参与 mask
    special_token_ids = set([
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ])
    special_mask = torch.tensor([
        1 if t in special_token_ids else 0
        for t in inputs.flatten().tolist()
    ], dtype=torch.bool).reshape(inputs.shape)
    probability_matrix.masked_fill_(special_mask, 0.0)

    # 随机选择要 mask 的位置
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 非目标位置

    # 80% → 替换为 [MASK]
    replace_mask = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
    inputs[replace_mask] = tokenizer.mask_token_id

    # 10% → 替换为随机 token
    random_mask = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~replace_mask
    random_tokens = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long)
    inputs[random_mask] = random_tokens[random_mask]

    # 剩余 10% → 保持不变（已隐含在逻辑中）

    return inputs, labels


# 演示：同一段文本被 mask 两次，结果不同
text = "深度学习是人工智能的核心技术"
token_ids = tokenizer.encode(text, return_tensors='pt')

masked1, labels1 = dynamic_mask(token_ids.clone(), tokenizer)
masked2, labels2 = dynamic_mask(token_ids.clone(), tokenizer)

print("第 1 次 mask:", tokenizer.decode(masked1[0]))
print("第 2 次 mask:", tokenizer.decode(masked2[0]))
# 两次结果不同！每次都是新的挑战
```

> *"静态 mask 如同固定的考题——做多了自然记住答案。动态 mask 如同随机出题——你必须真正理解药方的原理才能应对。RoBERTa 证明了后者的修炼效果远超前者。"*

### A.2 ALBERT：以小博大 — 参数压缩术

**ALBERT**（A Lite BERT，2019，Google）针对 BERT 参数量过大的问题，提出了两种精妙压缩术：

> *"大斗师的结晶虽然强大，但过于庞大沉重，携带不便。ALBERT 创造了两门化形术，能在保持实力的同时大幅缩减结晶体积。"*

#### 1. 嵌入层参数解耦（Factorized Embedding Parameterization）

```
BERT 的 Embedding:
  Embedding(V=30K, H=768) → 参数量 = 30,000 × 768 = 23M

ALBERT 的 Embedding:
  Embedding(V=30K, E=128) → 参数量 = 30,000 × 128 = 3.8M
  线性投影(E=128, H=768)  → 参数量 = 128 × 768 = 0.1M
  总计 = 3.9M（减少 83%！）
```

原理：词汇表中的 token 只需要在一个低维空间（E=128）中区分彼此，而深层语义的表达由 Transformer 层负责（H=768）。两者解耦。

```python
"""
=== ALBERT 嵌入解耦实现 ===
"""
class FactorizedEmbedding(nn.Module):
    """
    嵌入解耦术 — 将大斗师的结晶拆分为轻量外壳和核心
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=768):
        super().__init__()
        # 轻量词嵌入（词汇表 → 低维空间）
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 投影层（低维 → 隐藏维度）
        self.projection = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, input_ids):
        # 先映射到低维空间，再投影到隐藏维度
        x = self.word_embeddings(input_ids)  # (B, T, E)
        x = self.projection(x)                # (B, T, H)
        return x


# 参数量对比
bert_emb = nn.Embedding(30000, 768)
albert_emb = FactorizedEmbedding(30000, 128, 768)

bert_params = sum(p.numel() for p in bert_emb.parameters())
albert_params = sum(p.numel() for p in albert_emb.parameters())

print(f"BERT Embedding 参数量: {bert_params:,} ({bert_params/1e6:.1f}M)")
print(f"ALBERT Embedding 参数量: {albert_params:,} ({albert_params/1e6:.1f}M)")
print(f"压缩率: {albert_params/bert_params:.1%}")
# BERT: 23,040,000 (23.0M)
# ALBERT: 3,898,368 (3.9M)
# 压缩率: 16.9%
```

#### 2. 跨层参数共享（Cross-Layer Parameter Sharing）

```
BERT-Base: 12 层，每层独立参数
  参数量 = 12 × (每层参数)

ALBERT: 12 层，所有层共享同一套参数
  参数量 = 1 × (每层参数)
```

共享策略有三种：
- **all-shared**：Attention + FFN 全共享（最激进）
- **shared-attention**：只共享 Attention，FFN 独立
- **shared-ffn**：只共享 FFN，Attention 独立

实验表明 all-shared 效果最好且参数量最少。

```python
"""
=== 跨层参数共享 ===
"""
class SharedTransformerBlock(nn.Module):
    """
    跨层共享 Transformer 块
    如同大斗师只有一套核心功法，但反复运转12次
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)  # 同一个 Attention 模块
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(                  # 同一个 FFN 模块
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ALBERTModel(nn.Module):
    def __init__(self, config, n_layers=12):
        super().__init__()
        self.embeddings = FactorizedEmbedding(
            config.vocab_size, config.embedding_dim, config.hidden_size
        )
        # 关键：12 层共享同一个 block！
        self.shared_block = SharedTransformerBlock(config)
        self.n_layers = n_layers
        self.ln_f = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = self.embeddings(x)
        for _ in range(self.n_layers):
            x = self.shared_block(x)  # 同一个模块，反复调用
        x = self.ln_f(x)
        return x
```

#### ALBERT 规模对比

| 模型 | 层数 | Hidden | Embedding | 参数量 |
|------|------|--------|-----------|--------|
| BERT-Base | 12 | 768 | 768 | **110M** |
| BERT-Large | 24 | 1024 | 1024 | **340M** |
| ALBERT-Base | 12 | 768 | 128 | **12M** |
| ALBERT-Large | 24 | 1024 | 128 | **18M** |
| ALBERT-XLarge | 24 | 2048 | 128 | **60M** |
| ALBERT-XXLarge | 12 | 4096 | 128 | **235M** |

> *"以 12M 参数达到 BERT-Base 110M 参数的效果——这就是化形术的精髓：用更少的资源做更多的事。"*

### A.3 ELECTRA：替身修炼术

**ELECTRA**（Efficiently Learning an Encoder that Classifies Token Replacements Accurately，2020，Stanford）提出了一个极其巧妙的预训练目标，大幅提升训练效率。

> *"普通修炼中，学徒需要猜测被隐藏的药材（MLM）——但只有被选中的 15% 位置产生学习信号，其余 85% 的位置白白浪费。ELECTRA 发明了'替身甄别术'——不再猜测被隐藏的药材，而是甄别每个位置是否被替换了。"*

#### 核心思想：替换检测（Replaced Token Detection）

```
MLM（BERT）：
  输入: 我 爱 仒 北京 天安门    （只有 15% 被 mask）
  任务: 猜测 仒 是什么 → 北京
  问题: 85% 的 token 没有学习信号

RTD（ELECTRA）：
  输入: 我 爱 仒 北京 天安门    （15% 被替换为假 token）
  任务: 逐个判断每个 token 是真是假
    我 → 真(True)
    爱 → 真(True)
    仒 → 假(False) ← 检测到了替换！
    北京 → 真(True)
    天安门 → 真(True)
  优势: 100% 的 token 都参与学习！
```

#### 双网络架构

ELECTRA 使用**两个协同工作的网络**：

```
Generator（生成器）：
  - 小型 Masked LM（如 BERT-Tiny，~14M 参数）
  - 任务：生成替换 token（和 BERT 的 MLM 一样）
  - 作用：制造"假 token"供 Discriminator 甄别

Discriminator（判别器）：
  - 主模型（如 BERT-Base，~110M 参数）
  - 任务：判断每个 token 是原始的还是被 Generator 替换的
  - 输出：每个位置输出 True/False（二分类）
  - 作用：通过甄别真假来学习语言知识
```

```python
"""
=== 丹方 3.A.2：ELECTRA 核心原理实现 ===
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ElectraGenerator(nn.Module):
    """
    生成器 — 制造假 token 的小型 MLM
    如同一位初级药师，故意在药方中掺入假药材来考验学徒
    """
    def __init__(self, vocab_size, hidden_size=256, n_layers=4, n_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size*4,
                activation='gelu', batch_first=True
            ),
            num_layers=n_layers
        )
        self.proj = nn.Linear(hidden_size, vocab_size)  # 预测词汇表

    def forward(self, input_ids, masked_positions):
        x = self.token_emb(input_ids) + self.pos_emb(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        x = self.encoder(x)
        # 只在被 mask 的位置生成替换 token
        logits = self.proj(x)
        return logits


class ElectraDiscriminator(nn.Module):
    """
    判别器 — 甄别真假 token 的主模型
    如同经验丰富的药师，逐一检查药方中每味药材的真伪
    """
    def __init__(self, vocab_size, hidden_size=768, n_layers=12, n_heads=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)

        # 关键：在 Embedding 层加入"判别头"
        # 每个位置输出一个标量，判断是真是假
        self.discrim_head = nn.Linear(hidden_size, 1)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size*4,
                activation='gelu', batch_first=True
            ),
            num_layers=n_layers
        )

    def forward(self, input_ids):
        x = self.token_emb(input_ids) + self.pos_emb(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        x = self.encoder(x)

        # 每个位置输出一个 logit（True=真, False=假）
        logits = self.discrim_head(x)  # (B, T, 1)
        return logits.squeeze(-1)       # (B, T)


# ============================
# ELECTRA 训练步骤
# ============================
def electra_training_step(generator, discriminator, input_ids, tokenizer, device):
    """
    ELECTRA 的一步训练 — 替身甄别术
    """
    batch_size, seq_len = input_ids.shape

    # Step 1: 随机选择 15% 的位置进行替换
    mask_prob = 0.15
    probability_matrix = torch.full(input_ids.shape, mask_prob, device=device)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Step 2: 用 Generator 生成替换 token
    with torch.no_grad():
        gen_logits = generator(input_ids, masked_indices)
        # 在被 mask 的位置，取概率最高的 token 作为替换
        gen_probs = F.softmax(gen_logits, dim=-1)
        # 保留原始 token 的 10% 概率（避免 Generator 生成正确答案导致无信号）
        gen_probs = gen_probs * (1 - 0.1 * masked_indices.float().unsqueeze(-1))
        # 在被 mask 位置的原始 token 上也分配一些概率
        original_onehot = F.one_hot(input_ids, gen_logits.size(-1)).float()
        gen_probs = gen_probs + 0.1 * original_onehot * masked_indices.float().unsqueeze(-1)
        sampled_tokens = torch.multinomial(gen_probs.view(-1, gen_probs.size(-1)), 1)
        sampled_tokens = sampled_tokens.view(batch_size, seq_len)

    # 构造替换后的输入（仅在 mask 位置替换）
    replaced = input_ids.clone()
    replaced[masked_indices] = sampled_tokens[masked_indices]

    # Step 3: 判别器的目标 — 判断每个位置是否被替换
    # True = 原始 token（未被替换），False = 被 Generator 替换的假 token
    is_original = (~masked_indices).float()

    # Step 4: 训练判别器
    disc_logits = discriminator(replaced)
    disc_loss = F.binary_cross_entropy_with_logits(
        disc_logits, is_original, reduction='mean'
    )

    # Step 5: 训练生成器（标准 MLM loss）
    gen_logits = generator(input_ids, masked_indices)
    gen_loss = F.cross_entropy(
        gen_logits.view(-1, gen_logits.size(-1)),
        input_ids.view(-1),
        ignore_index=-100,
        reduction='none'
    )
    # 只计算被 mask 位置的 loss
    gen_loss = gen_loss[masked_indices.flatten()].mean()

    # 总 loss = Generator Loss + λ × Discriminator Loss
    total_loss = gen_loss + 50 * disc_loss  # λ=50 是 ELECTRA 论文的推荐值

    return total_loss, gen_loss.item(), disc_loss.item()
```

#### ELECTRA 的效率优势

| 指标 | BERT | ELECTRA |
|------|------|---------|
| 训练信号利用率 | 15% 的 token | **100% 的 token** |
| 相同算力下的效果 | 基线 | **显著更好** |
| 同等效果的训练成本 | 1× | **~1/4** |

> *"ELECTRA 的替身术是预训练领域最优雅的创新之一——不是改进网络结构，而是改进学习目标，让每一份算力都产生学习信号。修炼效率提升 6-7 倍！"*

### A.4 T5：编码-解码统一 — 万法归一

**T5**（Text-to-Text Transfer Transformer，2019，Google）提出了一个极其统一的框架：**所有 NLP 任务都转化为"文本到文本"（text-to-text）格式**。

> *"无论是双面法则（编码）、因果法则（解码），还是分类、翻译、摘要——在 T5 看来，世间一切任务都是'读入文本、输出文本'。万法归一，大道至简。"*

#### Text-to-Text 范式

```
传统方式（不同任务用不同输出头）：
  文本分类 → 神经网络 → 标签概率分布
  机器翻译 → 神经网络 → 目标语言词表概率
  摘要生成 → 神经网络 → 摘要文本

T5 方式（统一为 text-to-text）：
  文本分类 → "translate English to German: I love you" → "Ich liebe dich"
  分类任务 → "sst sentence: This movie is great" → "positive"
  摘要任务 → "summarize: [长文本...]" → "[摘要]"
  QA 任务   → "question: 什么是深度学习? context: [段落]" → "[答案]"
```

所有任务共享完全相同的模型架构和训练流程！

#### Span Corruption：T5 的预训练目标

T5 使用了不同于 MLM 的预训练策略——**Span Corruption**（片段损坏）：

```
输入：
  "感谢 填充文本 填充文本 感谢支持，请填写 填充文本 的反馈意见。"
  ↑ 两个长度为2和3的片段被替换为哨兵token

输出：
  "额外ID_0 额外ID_1 深度学习 技术 的"

任务：预测被损坏的片段，按原始顺序输出
```

与 MLM 的关键区别：
- MLM：每次只 mask 单个 token（或偶尔 span）
- Span Corruption：随机 mask 连续片段（span），span 长度服从几何分布
- 输出格式：按原始顺序输出所有被损坏的 span，用哨兵 ID 分隔

```python
"""
=== 丹方 3.A.3：Span Corruption 实现 ===
"""
import torch
import random
import math

def create_span_corruption(
    input_ids, tokenizer,
    mean_span_length=3,   # 平均 span 长度
    corruption_rate=0.15,  # 被损坏的 token 比例
    mask_token_id=None,    # 哨兵 token 起始 ID
):
    """
    Span Corruption — T5 的预训练数据构造
    如同在药方中随机挖掉若干连续的药材片段，
    学徒需要按原始顺序补全所有被挖掉的部分
    """
    if mask_token_id is None:
        mask_token_id = tokenizer.vocab_size - 1  # 使用词汇表末尾的 ID

    seq_len = input_ids.size(0)
    num_tokens_to_corrupt = int(seq_len * corruption_rate)

    # 按几何分布生成 span 长度
    spans = []
    while num_tokens_to_corrupt > 0:
        # 几何分布采样 span 长度
        span_length = torch.geometric(
            torch.tensor(1.0 / mean_span_length), size=(1,)
        ).item()
        span_length = min(span_length, num_tokens_to_corrupt)
        if span_length > 0:
            spans.append(span_length)
        num_tokens_to_corrupt -= span_length

    if not spans:
        # 没有生成有效 span，直接返回
        return input_ids, input_ids

    # 随机选择 span 起始位置（不重叠）
    # 简化实现：按顺序放置
    masked_positions = set()
    corrupted_input = input_ids.clone()
    target_tokens = []
    sentinel_idx = 0

    # 可用位置
    available = list(range(seq_len))
    random.shuffle(available)
    pos_ptr = 0

    for span_len in spans:
        # 找到连续可用位置
        if pos_ptr + span_len > len(available):
            break

        start = available[pos_ptr]
        end = start + span_len

        # 检查连续性
        if available[pos_ptr:pos_ptr + span_len] != list(range(start, end)):
            pos_ptr += 1
            continue

        # 记录被损坏的 token（加入目标序列）
        target_tokens.extend(input_ids[start:end].tolist())

        # 在输入中用哨兵 token 替换
        corrupted_input[start:end] = mask_token_id + sentinel_idx

        # 哨兵 ID 也加入目标序列（标记不同 span）
        target_tokens.insert(0, mask_token_id + sentinel_idx)
        sentinel_idx += 1
        pos_ptr += span_len

    # 构造目标序列
    target = torch.tensor(target_tokens, dtype=input_ids.dtype)

    return corrupted_input, target


# 演示
text = "深度学习是人工智能领域的核心技术之一"
token_ids = tokenizer.encode(text, return_tensors='pt')[0]

corrupted_input, target = create_span_corruption(
    token_ids, tokenizer, mean_span_length=3, corruption_rate=0.15
)

print("原始文本:", text)
print("损坏输入:", tokenizer.decode(corrupted_input))
print("目标输出:", tokenizer.decode(target))
```

#### T5 架构：Encoder-Decoder 完整体

T5 是**第一个大规模成功使用完整 Encoder-Decoder 的预训练模型**（BERT 只有 Encoder，GPT 只有 Decoder）：

```
T5 架构：
┌──────────────────┐     ┌──────────────────┐
│    Encoder        │     │    Decoder        │
│  (双向注意力)      │     │  (因果注意力)      │
│                  │     │                  │
│  Self-Attention  │     │  Masked Self-Attn │
│       ↓          │     │       ↓          │
│  Cross-Attention │────→│  Cross-Attention │  ← Decoder 看向 Encoder
│       ↓          │     │       ↓          │
│  FFN             │     │  FFN             │
└──────────────────┘     └──────────────────┘
```

T5 模型规模：

| 版本 | 层数 | d_model | Heads | 参数量 |
|------|------|---------|-------|--------|
| T5-Small | 6 | 512 | 8 | 60M |
| T5-Base | 12 | 768 | 12 | 220M |
| T5-Large | 24 | 1024 | 16 | 770M |
| T5-XL | 24 | 2048 | 32 | 3B |
| T5-XXL | 24 | 4096 | 128 | 11B |

> *"T5 的 Text-to-Text 理念深刻影响了后来所有大模型的训练方式。现代的 Instruction Tuning（指令微调）本质上就是 Text-to-Text 的扩展——将一切任务统一为 '指令 → 回复' 的格式。"*

### A.5 BART：双向编码、自回归解码

**BART**（Bidirectional and Auto-Regressive Transformers，2019，Facebook）结合了 BERT 和 GPT 的优势：用双向编码器理解输入，用自回归解码器生成输出。

> *"BART 采纳了双面法则的洞察力（双向编码），又保留了因果法则的创造力（自回归生成），是两者的完美结合。"*

#### BART 的预训练：降噪自编码（Denoising Autoencoding）

BART 的预训练目标是对**被噪声损坏的文本进行重建**：

```python
"""
=== BART 降噪预训练策略 ===
"""

# 噪声类型 1：Token Masking（和 BERT 类似）
# "我 爱 [MASK] 学习 [MASK] 法"

# 噪声类型 2：Token Deletion（随机删除 token）
# "我 爱 人工智能 学习 方法"
# → "我爱人工智能方法"（删除了"的"和"深"）

# 噪声类型 3：Text Infilling（用单个 [MASK] 替换整个 span）
# "我 爱 [MASK] 习 [MASK]"
# 注意：span 无论多长，只用一个 [MASK] 替换！

# 噪声类型 4：Sentence Permutation（打乱句子顺序）
# "今天天气很好。我们去公园散步。午饭吃了面条。"
# → "午饭吃了面条。今天天气很好。我们去公园散步。"

# 噪声类型 5：Document Rotation（随机旋转文档）
# 从随机位置开始，模型需要推断文档的真正起始位置

def apply_noise(input_ids, tokenizer, noise_type="infill", noise_prob=0.15):
    """
    对输入文本施加噪声 — BART 预训练数据构造
    """
    seq_len = input_ids.size(0)

    if noise_type == "masking":
        # 类似 BERT 的 token masking
        masked = input_ids.clone()
        mask_positions = torch.rand(seq_len) < noise_prob
        masked[mask_positions] = tokenizer.mask_token_id
        return masked, input_ids

    elif noise_type == "infilling":
        # Text Infilling — BART 默认使用
        # 将 token 分成若干 span，每个 span 用一个 [MASK] 替换
        masked = input_ids.clone()
        positions = list(range(seq_len))

        # 生成 span（泊松分布采样长度）
        spans = []
        remaining = seq_len
        idx = 0
        while remaining > 0:
            span_len = torch.poisson(torch.tensor(3.5)).item()
            span_len = min(int(span_len), remaining)
            if span_len > 0 and random.random() < 0.35:
                spans.append((idx, idx + span_len))
            remaining -= span_len
            idx += span_len

        # 用 [MASK] 替换每个 span
        for start, end in spans:
            masked[start:end] = tokenizer.mask_token_id

        return masked, input_ids

    elif noise_type == "deletion":
        # 随机删除 token
        mask = torch.rand(seq_len) >= noise_prob
        deleted = input_ids[mask]
        if deleted.size(0) == 0:
            deleted = input_ids[:1]
        return deleted, input_ids

    elif noise_type == "sentence_permutation":
        # 打乱句子顺序
        # 需要已知句子边界（通过 [SEP] 或句号识别）
        return input_ids, input_ids  # 简化实现

    return input_ids, input_ids
```

#### 预训练模型族谱总结

```
2018-BERT ──→ 2019-RoBERTa（训练策略优化）
    │      ──→ 2019-ALBERT（参数压缩）
    │      ──→ 2019-ELECTRA（替换检测）
    │      ──→ 2019-StructBERT（结构预训练）
    │
2019-T5 ────→ Encoder-Decoder 范式
    │
2019-BART ──→ 降噪自编码
    │
2020-DeBERTa → 解耦注意力
    │
2022-ChatGPT ─→ Decoder-Only 走向主流
```

```python
"""
=== 预训练模型选择决策树 ===
"""
print("""
╔══════════════════════════════════════════════════╗
║           预训练模型选择决策树                     ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  你的任务是什么？                                  ║
║                                                  ║
║  ├─ 文本生成 / 对话 / 写作                        ║
║  │   → GPT 系列 / LLaMA / Mistral                ║
║  │   （Decoder-Only，因果语言模型）                ║
║  │                                                ║
║  ├─ 文本理解 / 分类 / NER / 匹配                   ║
║  │   → BERT / RoBERTa / DeBERTa                   ║
║  │   （Encoder-Only，双向理解）                    ║
║  │                                                ║
║  ├─ 翻译 / 摘要 / Seq2Seq                        ║
║  │   → T5 / BART / mBART                        ║
║  │   （Encoder-Decoder，编码理解+解码生成）         ║
║  │                                                ║
║  ├─ 参数受限 / 部署场景                           ║
║  │   → ALBERT / DistilBERT / TinyBERT            ║
║  │   （压缩变体）                                  ║
║  │                                                ║
║  └─ 中文场景优先                                  ║
║      → BERT-Chinese / RoBERTa-wwm-ext             ║
║      → MacBERT / ERNIE                            ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")
```

> *"修炼至此，你已掌握预训练模型的全景图。BERT 双面法则开山立派，RoBERTa 优化细节，ALBERT 以小博大，ELECTRA 替身修炼，T5 万法归一，BART 编码解码合一。选择哪门功法，取决于你的具体任务。然而，万变不离其宗——预训练-微调范式，才是大斗师之境的核心心法。"*

---

## 第四章：因果法则 — GPT 预训练原理

> *"因果法则，只知过去，不窥未来。以已知推未知，以过往定将来。一字一句，如命运之链，前因注定后果。"*

### 4.1 GPT 的哲学：自回归生成

如果说 BERT 是 "理解大师"——通过双向感知理解文本，那么 GPT 就是 "创造大师"——通过自回归生成创造文本。

**自回归（Autoregressive）**意味着：**一次只生成一个 token，每个 token 的生成都依赖于前面所有已生成的 token**。

```
输入：  "今天天气"
步骤1: P("真" | "今天天气") = 0.35   → 生成 "真"
步骤2: P("好" | "今天天气真") = 0.42  → 生成 "好"
步骤3: P("，" | "今天天气真好") = 0.51 → 生成 "，"
步骤4: P("适" | "今天天气真好，") = 0.28 → 生成 "适"
...

最终生成：  "今天天气真好，适合出去散步。"
```

> *"如同推演命运之线——每一步只能根据已知的过去推断下一个节点。过去的每一个选择都影响未来的走向。"*

### 4.2 Causal Language Modeling (CLM)

GPT 的预训练目标非常简单：**给定前面的所有 token，预测下一个 token**。

```
训练目标：最大化 P(x_t | x_1, x_2, ..., x_{t-1})

句子："深度学习改变世界"

训练样本：
  P(度 | 深)
  P(学 | 深度)
  P(习 | 深度学)
  P(改 | 深度学习)
  P(变 | 深度学习改)
  P(世 | 深度学习改变)
  P(界 | 深度学习改变世)

Loss = -Σ log P(x_t | x_1, ..., x_{t-1})  （交叉熵损失）
```

这个目标看似简单，但让模型在海量文本上学习这个目标，模型就必须理解语法、语义、常识、推理等各种语言能力——因为只有真正 "理解" 了语言，才能准确预测下一个词。

### 4.3 因果注意力掩码：只看过去，不窥未来

GPT 使用 **Causal Attention Mask（因果注意力掩码）**来确保每个位置只能关注它前面（包括自身）的位置：

```
           看  深  度  学  习
看          1   0   0   0   0
深          1   1   0   0   0
度          1   1   1   0   0
学          1   1   1   1   0
习          1   1   1   1   1

1 = 可以看到, 0 = 看不到（被掩盖）

这是一个下三角矩阵！
```

在 PyTorch 中的实现：

```python
"""
=== 因果注意力掩码 ===
"""
import torch

def create_causal_mask(seq_len):
    """
    创建因果掩码 — 命运之链只能向前延伸
    """
    # 生成下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 将 0 的位置设为 -inf（在 softmax 后变为 0）
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask

# 示例
mask = create_causal_mask(5)
print("因果掩码（0 表示可见，-inf 表示不可见）：")
print(mask)
```

### 4.4 GPT-2 架构详解

GPT-2（2019, OpenAI）是 GPT 系列的重要里程碑。它使用 **Transformer Decoder**（但去掉了交叉注意力层，因为没有 Encoder 输入）：

```
GPT-2 架构（Decoder-Only Transformer）：

输入 Token IDs
      ↓
Token Embedding + Position Embedding
      ↓
┌─────────────────────────────┐
│  Transformer Decoder Block  │  × N 层
│  ┌───────────────────────┐  │
│  │ Masked Self-Attention  │  │  ← 因果掩码！
│  │ (只看过去)              │  │
│  └──────────┬────────────┘  │
│             ↓ + Residual     │
│  ┌───────────────────────┐  │
│  │    Layer Norm          │  │
│  └──────────┬────────────┘  │
│             ↓                │
│  ┌───────────────────────┐  │
│  │    Feed Forward (MLP)  │  │
│  └──────────┬────────────┘  │
│             ↓ + Residual     │
│  ┌───────────────────────┐  │
│  │    Layer Norm          │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
      ↓
  Layer Norm
      ↓
  Linear → Vocab Size（预测下一个 token 的概率分布）
```

**GPT-2 的不同规模**：

| 版本 | 层数 | Hidden | Heads | 参数量 |
|------|-----|--------|-------|-------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 774M |
| XL | 48 | 1600 | 25 | 1.5B |

注意 GPT-2 与原始 Transformer Decoder 的区别：

- **去掉了交叉注意力层**（Cross-Attention）：因为没有 Encoder，只有 Decoder
- **Pre-Norm**：Layer Norm 放在 Attention/FFN 之前（而非之后），训练更稳定
- 使用 **GELU** 激活函数（而非 ReLU）

### 4.5 生成策略：控制创造的火候

训练好的 GPT 模型输出的是下一个 token 的概率分布。如何从这个分布中选择 token，直接决定了生成文本的质量和多样性。

#### Greedy Search（贪心搜索）

```
每一步都选概率最高的 token。

优点：确定性，可复现
缺点：生成文本单调、重复
```

> *"每次都取最顺手的药材——稳定但缺乏变化，炼出的丹药千篇一律。"*

#### Temperature Scaling（温度缩放）

```python
# Temperature 控制概率分布的 "尖锐度"
logits = model_output.logits / temperature

# temperature = 1.0: 原始分布
# temperature < 1.0: 分布更尖锐，高概率token更突出（更保守）
# temperature > 1.0: 分布更平坦，低概率token获得更多机会（更随机/创造性）
# temperature → 0: 退化为 Greedy Search
# temperature → ∞: 退化为均匀随机采样
```

> *"温度如同丹炉的火候——文火（低温度）炼出的丹药稳定但平庸，猛火（高温度）炼出的丹药充满变数，可能是奇丹也可能是废丹。"*

#### Top-k Sampling

```python
# 只从概率最高的 k 个 token 中采样
# 过滤掉概率极低的不合理选项

top_k = 50  # 只考虑概率最高的 50 个 token
```

> *"只从最可能的 k 味药材中选择，排除明显不合理的选项。"*

#### Top-p (Nucleus) Sampling

```python
# 从概率之和达到 p 的最小 token 集合中采样
# 动态调整候选集大小

top_p = 0.9  # 选择概率之和达到 90% 的最小 token 集
```

Top-p 比 Top-k 更智能：当模型对下一个 token 很确定时（少数 token 就占了 90% 概率），候选集很小；当模型不确定时（需要很多 token 才凑满 90%），候选集自动变大。

> *"根据当前局势动态调整选材范围——当方向明确时精确取材，当方向不明时广泛尝试。"*

#### 综合实战

```python
"""
=== 丹方 3.5：GPT-2 文本生成 ===
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练 GPT-2
model_name = "gpt2"  # 可换为 "gpt2-medium", "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置 pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

prompt = "Artificial intelligence will"

input_ids = tokenizer.encode(prompt, return_tensors='pt')

# ---- 贪心搜索 ----
output_greedy = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=False,  # 贪心
)
print("【贪心搜索】")
print(tokenizer.decode(output_greedy[0], skip_special_tokens=True))

# ---- 低温度采样（保守） ----
output_low_temp = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.3,
    top_k=50,
)
print("\n【低温度 (T=0.3)】")
print(tokenizer.decode(output_low_temp[0], skip_special_tokens=True))

# ---- 高温度采样（创造性） ----
output_high_temp = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=1.2,
    top_p=0.9,
)
print("\n【高温度 (T=1.2)】")
print(tokenizer.decode(output_high_temp[0], skip_special_tokens=True))

# ---- Nucleus Sampling（推荐默认配置） ----
output_nucleus = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,  # 惩罚重复
)
print("\n【Nucleus Sampling (T=0.7, top_p=0.9)】")
print(tokenizer.decode(output_nucleus[0], skip_special_tokens=True))
```

### 4.6 BERT vs GPT：感知大师 vs 创造大师

| 维度 | BERT | GPT |
|------|------|-----|
| 架构 | Transformer Encoder | Transformer Decoder (only) |
| 注意力方向 | 双向（全局可见） | 单向（只看过去） |
| 预训练任务 | MLM + NSP | Causal LM（下一个 token 预测） |
| 核心能力 | **理解**：文本分类、NER、QA | **生成**：文本续写、对话、创作 |
| 微调方式 | 加任务头，微调全部参数 | Prompt + 少样本 / 微调 |
| 比喻 | 感知型功法：洞察万物 | 因果型功法：创造万物 |
| 代表模型 | BERT, RoBERTa, ALBERT, DeBERTa | GPT-2, GPT-3, GPT-4, LLaMA |
| 适用场景 | 需要深度理解的任务 | 需要文本生成的任务 |

> *"BERT 如同一位洞察秋毫的阵法大师——将文本放入阵法中，从四面八方分析其精髓。"*
>
> *"GPT 如同一位妙笔生花的丹方创作者——从开头起笔，一字一句推演出完整的丹方。"*

在现代 LLM 发展中，GPT 的自回归范式已成为主流。原因在于：

1. **生成能力**天然更强（BERT 无法直接生成文本）
2. **Scaling Law** 在 decoder-only 架构上表现更好
3. **统一范式**：几乎所有任务都可以转化为 "生成" 任务
4. **In-context Learning**：大模型展现出无需微调、通过提示即可解决新任务的能力

但 BERT 类模型在**判别式任务**（分类、匹配、检索）上仍有独特优势，特别是在资源受限或需要高效推理的场景下。

---

## 附录 B：斗技百炼 — 下游任务微调实战

> *"预训练如铸剑胚，微调如淬火开刃。本附录将带领你在各类实战任务中磨砺你的预训练结晶，使之锋芒毕露。"*

### B.1 命名实体识别（NER）— 炼丹识药术

**NER（Named Entity Recognition）** 是 NLP 最经典的下游任务之一：从文本中识别出人名、地名、组织名等实体，并为每个实体标注类型。

> *"如同药师需要在复杂的药方中迅速识别出每一味药材的名称和种类——哪里是主药，哪里是辅药，哪里是引经药。"*

#### NER 作为序列标注问题

```
输入文本:  清华大学的张三去了北京大学
标注结果:  O  B-ORG I-ORG O B-PER O B-ORG I-ORG

标注体系（BIO）：
  B-X = 实体 X 的开头
  I-X = 实体 X 的内部
  O   = 非实体
```

```python
"""
=== 丹方 3.B.1：BERT NER 微调 — 实体识别术 ===
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

# ============================
# 第一步：定义标签体系
# ============================
label_list = [
    "O",           # 非实体
    "B-PER", "I-PER",      # 人名
    "B-ORG", "I-ORG",      # 组织
    "B-LOC", "I-LOC",      # 地点
    "B-MISC", "I-MISC",    # 其他
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)


# ============================
# 第二步：构建 NER 数据集
# ============================
class NERDataset(Dataset):
    """
    NER 数据集 — 每个样本是一句话及其对应的标签序列
    """
    def __init__(self, texts, tags_list, tokenizer, max_length=128,
                 label2id=None):
        self.texts = texts
        self.tags_list = tags_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def _align_labels(self, tags, word_ids):
        """
        对齐标签 — BPE 分词可能将一个字拆成多个 sub-token
        只给第一个 sub-token 分配真实标签，其余标为 -100（忽略）
        """
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # 特殊 token
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label2id[tags[word_idx]])
            else:
                aligned_labels.append(-100)  # 同一个字的后续 sub-token
            previous_word_idx = word_idx
        return aligned_labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.tags_list[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,  # 用于标签对齐
            return_tensors='pt',
        )

        # 获取 word_ids（每个 sub-token 对应原始文本的哪个字）
        word_ids = []
        for offset in encoding['offset_mapping'][0]:
            if offset[0] == 0 and offset[1] == 0:
                word_ids.append(None)  # 特殊 token
            else:
                # 找到对应的原始字索引
                char_idx = None
                pos = 0
                for i, char in enumerate(text):
                    if pos == offset[0].item():
                        char_idx = i
                        break
                    pos += len(char)
                word_ids.append(char_idx)

        # 对齐标签
        # 简化处理：假设中文每字一个 token
        token_labels = [-100] * self.max_length
        for i in range(min(len(tags), self.max_length - 2)):
            token_labels[i + 1] = self.label2id.get(tags[i], 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(token_labels, dtype=torch.long),
        }


# ============================
# 第三步：示例数据（真实项目请用 CoNLL-2003 / CLUENER 等数据集）
# ============================
texts = [
    "清华大学位于北京市海淀区",
    "张三是北京大学计算机系的学生",
    "李四在上海参加了人工智能大会",
    "华为公司在深圳发布了新款手机",
    "王五是中国科学院的研究员",
] * 40  # 扩展数据量

tags_list = [
    ["B-ORG", "I-ORG", "O", "B-LOC", "I-LOC", "I-LOC"],
    ["B-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O"],
    ["B-PER", "O", "B-LOC", "O", "O", "B-MISC", "I-MISC", "I-MISC"],
    ["B-ORG", "I-ORG", "O", "B-LOC", "O", "O", "O", "O", "O"],
    ["B-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O"],
] * 40


# ============================
# 第四步：训练 NER 模型
# ============================
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(
    model_name, num_labels=num_labels
)

# 为模型设置标签映射
model.config.id2label = id2label
model.config.label2id = label2id

# 准备数据
dataset = NERDataset(texts, tags_list, tokenizer, label2id=label2id)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 训练循环
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/3 | Loss: {total_loss/len(dataloader):.4f}")


# ============================
# 第五步：推理与评估
# ============================
@torch.no_grad()
def predict_ner(text, model, tokenizer, id2label, device):
    """
    NER 推理 — 从文本中提取实体
    """
    model.eval()
    encoding = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # 解码预测结果
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pred_labels = [id2label[p.item()] for p in predictions[0]]

    # 组合实体（BIO 解码）
    entities = []
    current_entity = None

    for token, label in zip(tokens, pred_labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": label[2:], "text": token}
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # 清理 token（去掉 ## 前缀等）
    for ent in entities:
        ent["text"] = ent["text"].replace("##", "")

    return entities


# 测试
test_text = "清华大学位于北京市海淀区，张三在这里学习"
entities = predict_ner(test_text, model, tokenizer, id2label, device)

print(f"\n输入: {test_text}")
print("识别到的实体:")
for ent in entities:
    print(f"  [{ent['type']}] {ent['text']}")
```

#### NER 评估指标：实体级 F1

```python
"""
=== NER 评估 — 实体级 F1 计算 ===
"""
def evaluate_ner(pred_entities, gold_entities):
    """
    实体级评估：精确匹配才算正确
    """
    pred_set = {(e['type'], e['text']) for e in pred_entities}
    gold_set = {(e['type'], e['text']) for e in gold_entities}

    tp = len(pred_set & gold_set)   # 正确预测
    fp = len(pred_set - gold_set)   # 误报
    fn = len(gold_set - pred_set)   # 漏报

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


# 示例
pred = [{"type": "ORG", "text": "清华大学"}, {"type": "LOC", "text": "北京市"}]
gold = [{"type": "ORG", "text": "清华大学"}, {"type": "LOC", "text": "北京"}]

result = evaluate_ner(pred, gold)
print(f"NER 评估结果: P={result['precision']:.2%} R={result['recall']:.2%} F1={result['f1']:.2%}")
```

### B.2 机器阅读理解（MRC）— 天眼通术

**MRC（Machine Reading Comprehension）** 是给定一段文本和一个问题，要求模型从文本中找到答案。

> *"如同天眼通——给你一卷古籍和一个问题，你需要在古籍中找到确切答案。"*

#### SQuAD 风格的抽取式 QA

```python
"""
=== 丹方 3.B.2：BERT 阅读理解微调 ===
"""
from transformers import BertForQuestionAnswering

# ============================
# 构建阅读理解数据集
# ============================
class QADataset(Dataset):
    """
    阅读理解数据集 — 给定 context + question，预测答案的起始和结束位置
    """
    def __init__(self, contexts, questions, answers, tokenizer, max_length=384):
        self.encodings = tokenizer(
            questions, contexts,
            max_length=max_length,
            truncation="only_second",  # 只截断 context
            padding="max_length",
            return_tensors="pt",
        )
        self.start_positions = []
        self.end_positions = []

        for i in range(len(contexts)):
            # 在 context 中找到答案的起止位置
            context_encoding = tokenizer(
                contexts[i],
                return_offsets_mapping=True,
            )
            answer_text = answers[i]

            # 找到答案在 context 中的字符偏移
            answer_start = contexts[i].find(answer_text)
            answer_end = answer_start + len(answer_text) - 1

            # 映射到 token 索引（考虑 question 部分的偏移）
            # 简化处理：实际需要精确的 offset_mapping
            # 这里使用简化版
            self.start_positions.append(answer_start)
            self.end_positions.append(answer_end)

        self.start_positions = torch.tensor(self.start_positions)
        self.end_positions = torch.tensor(self.end_positions)

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'start_positions': self.start_positions[idx],
            'end_positions': self.end_positions[idx],
        }


# ============================
# 示例数据
# ============================
contexts = [
    "深度学习是机器学习的一个分支，它试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。深度学习的核心是神经网络，特别是深度神经网络。",
    "自然语言处理是人工智能的重要方向，它研究如何让计算机理解和生成人类语言。预训练模型如 BERT 和 GPT 大大推动了 NLP 的发展。",
] * 30

questions = [
    "深度学习的核心是什么？",
    "推动 NLP 发展的模型有哪些？",
] * 30

answers = [
    "神经网络",
    "BERT 和 GPT",
] * 30

# ============================
# 训练 QA 模型
# ============================
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 使用 BertForQuestionAnswering（自动添加 span 预测头）
model = BertForQuestionAnswering.from_pretrained(model_name)
model.to(device)

dataset = QADataset(contexts, questions, answers, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            start_positions=batch['start_positions'].to(device),
            end_positions=batch['end_positions'].to(device),
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")


# ============================
# QA 推理
# ============================
@torch.no_grad()
def answer_question(question, context, model, tokenizer, device):
    """
    回答问题 — 从 context 中抽取答案
    """
    model.eval()
    inputs = tokenizer(question, context, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 取概率最高的起止位置
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # 确保 end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    # 解码答案
    answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer


# 测试
context = "PyTorch 是由 Facebook AI Research 团队开发的深度学习框架，于 2016 年首次发布。它支持动态计算图，在研究社区中广受欢迎。"
question = "PyTorch 是由谁开发的？"
answer = answer_question(question, context, model, tokenizer, device)
print(f"\n问题: {question}")
print(f"答案: {answer}")
```

### B.3 语义相似度匹配 — 双星识别术

**语义匹配**判断两段文本是否语义相关，广泛应用于搜索、问答、推荐等场景。

```python
"""
=== 丹方 3.B.3：BERT 语义相似度 ===
"""
from transformers import BertModel

class SemanticSimilarityModel(nn.Module):
    """
    语义相似度模型 — BERT + 相似度计算头
    """
    def __init__(self, model_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # 分类头：输入两个 [CLS] 向量的拼接 → 输出相似度
        self.classifier = nn.Sequential(
            nn.Linear(768 * 3, 256),  # cls1 + cls2 + |cls1-cls2|
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # 编码两段文本
        outputs1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)

        cls1 = outputs1.pooler_output   # (B, 768)
        cls2 = outputs2.pooler_output

        # 拼接：[cls1; cls2; |cls1-cls2|]
        diff = torch.abs(cls1 - cls2)
        combined = torch.cat([cls1, cls2, diff], dim=-1)

        # 输出相似度分数
        logits = self.classifier(combined)
        return logits


# 另一种高效方式：Sentence-BERT（Siamese Network）
class SentenceBert(nn.Module):
    """
    Sentence-BERT — 孪生网络
    先分别编码两段文本为向量，再计算余弦相似度
    优势：编码后的向量可缓存复用，适合大规模检索
    """
    def __init__(self, model_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 或 mean pooling
        cls = outputs.last_hidden_state[:, 0]  # (B, 768)
        return cls

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        vec1 = self.encode(input_ids1, attention_mask1)
        vec2 = self.encode(input_ids2, attention_mask2)
        similarity = self.cosine_sim(vec1, vec2)
        return similarity


# ============================
# 使用 Sentence-BERT 计算文本相似度
# ============================
print("""
Sentence-BERT 使用场景：

1. 语义搜索
   query = "如何学习深度学习"
   docs = ["深度学习入门教程", "机器学习实战", "Python 编程指南", ...]
   → 编码所有文档为向量（只需一次）
   → 编码 query 为向量
   → 计算余弦相似度，返回最相似的文档

2. 问答匹配
   question = "什么是 PyTorch？"
   answer1 = "PyTorch 是一个深度学习框架"
   answer2 = "今天天气不错"
   → similarity(q, a1) = 0.85 (高)
   → similarity(q, a2) = 0.12 (低)

3. 语义去重
   text1 = "如何学习人工智能"
   text2 = "AI 学习方法有哪些"
   → similarity(t1, t2) = 0.92 (语义相同，可去重)

4. 聚类分析
   将一组文本编码为向量后，用 K-Means 等算法聚类
   可用于话题发现、意图识别等
""")
```

### B.4 实战：完整中文文本分类流水线

```python
"""
=== 丹方 3.B.4：完整中文文本分类流水线 ===
从数据加载到模型导出的一站式方案
"""
import json
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import EarlyStoppingCallback

# ============================
# 第一步：加载真实数据集（以 THUCNews 子集为例）
# ============================
from datasets import load_dataset

# 使用 HuggingFace 上的中文分类数据集
# 可选：tnews（新闻分类）、iflytek（APP分类）、eprstmt（情感分析）
dataset = load_dataset("thucnews", "subset", split="train[:1000]")
test_dataset = load_dataset("thucnews", "subset", split="validation[:200]")

# 提取文本和标签
train_texts = [item['text'] for item in dataset]
train_labels = [item['label'] for item in dataset]
test_texts = [item['text'] for item in test_dataset]
test_labels = [item['label'] for item in test_dataset]

num_classes = len(set(train_labels))
print(f"训练集大小: {len(train_texts)}, 类别数: {num_classes}")

# ============================
# 第二步：使用 Trainer API 简化训练
# ============================
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


# 准备 Dataset
from datasets import Dataset as HFDataset

train_hf = HFDataset.from_dict({"text": train_texts, "label": train_labels})
test_hf = HFDataset.from_dict({"text": test_texts, "label": test_labels})

train_tokenized = train_hf.map(tokenize_function, batched=True)
test_tokenized = test_hf.map(tokenize_function, batched=True)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_classes
)

# 训练配置
training_args = TrainingArguments(
    output_dir="./bert_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # 自动使用混合精度
)

# 计算指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average='macro')
    accuracy = (predictions == labels).mean()
    return {"f1": f1, "accuracy": accuracy}

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 开始训练！
trainer.train()

# ============================
# 第三步：评估
# ============================
eval_results = trainer.evaluate()
print(f"\n评估结果: {eval_results}")

# ============================
# 第四步：保存和导出模型
# ============================
trainer.save_model("./bert_classifier_best")
tokenizer.save_pretrained("./bert_classifier_best")

print("模型已保存到 ./bert_classifier_best")


# ============================
# 第五步：使用导出模型进行推理
# ============================
from transformers import pipeline

# 一行代码加载分类器
classifier = pipeline(
    "text-classification",
    model="./bert_classifier_best",
    tokenizer="./bert_classifier_best",
    device=0 if torch.cuda.is_available() else -1,
)

# 测试
test_texts = [
    "中国足球队在世界杯预选赛中取得胜利",
    "新款 iPhone 发布，售价 9999 元起",
    "人工智能技术在医疗领域取得突破",
    "财政部宣布下调存款准备金率",
]

for text in test_texts:
    result = classifier(text, top_k=3)
    print(f"\n文本: {text}")
    for r in result:
        print(f"  {r['label']}: {r['score']:.4f}")
```

> *"至此，你已修炼了四种下游斗技——NER 识药术、MRC 天眼通、语义匹配双星术、文本分类判别术。每种斗技都以预训练结晶（BERT）为基础，只需少量针对性修炼（微调）即可精通。这正是预训练-微调范式的威力所在。"*

---

## 第五章：首次凝晶 — 从零训练文本模型

> *"纸上得来终觉浅，绝知此事要躬行。万卷丹书不如一炉实炼。本章，你将亲手从空白开始，凝聚出你的第一颗斗气结晶。"*

### 5.1 项目概览

我们将完成以下完整流程：

```
原始文本数据
     ↓  (1) 训练 Tokenizer
自定义 BPE Tokenizer
     ↓  (2) 数据预处理
Token ID 序列数据集
     ↓  (3) 构建模型
Mini GPT 模型 (4-6 层, ~10M 参数)
     ↓  (4) 训练
在单张 GPU 上训练语言模型
     ↓  (5) 生成
输入 prompt, 生成连贯文本
     ↓  (6) 评估
计算 Perplexity
```

完整代码见配套丹方：`notebooks/03_从零训练语言模型.py`

### 5.2 数据准备：采集药材

对于中文语言模型，可以使用以下数据源：

```python
"""
=== 数据准备方案 ===
"""

# 方案一：使用 HuggingFace Datasets（推荐入门）
from datasets import load_dataset

# 中文维基百科
dataset = load_dataset("wikipedia", "20220301.zh", split="train[:5000]")

# 或者使用中文小说、新闻等数据
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # 英文入门

# 方案二：自己收集文本文件
import glob
texts = []
for filepath in glob.glob("data/*.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        texts.append(f.read())

# 方案三：使用经典中文语料
# - 中文维基百科 dump
# - 中文新闻语料 (THUCNews)
# - 中文网络小说（注意版权）
```

**数据清洗要点**：

```python
import re

def clean_text(text):
    """
    药材洗炼 — 去除杂质
    """
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 统一标点符号（可选）
    text = text.strip()
    return text
```

### 5.3 训练自定义 Tokenizer

```python
"""
=== 为你的语料训练专用 Tokenizer ===
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing

# 初始化 BPE Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 中文不需要按空格预分词，使用字符级预分词
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 训练配置
trainer = trainers.BpeTrainer(
    vocab_size=8000,           # 小型实验用 8000 即可
    min_frequency=2,
    special_tokens=[
        "<pad>",    # 0: 填充
        "<unk>",    # 1: 未知
        "<bos>",    # 2: 序列开始
        "<eos>",    # 3: 序列结束
    ],
    show_progress=True,
)

# 在语料上训练
# tokenizer.train(files=["data/corpus.txt"], trainer=trainer)
# tokenizer.save("my_tokenizer.json")
```

### 5.4 构建 Mini GPT 模型

以下是一个完整的小型 GPT 模型实现。详细代码参见配套丹方 `notebooks/03_从零训练语言模型.py`，此处展示核心架构：

```python
"""
=== Mini GPT 模型架构 ===
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniGPTConfig:
    """模型配置 — 丹方参数"""
    vocab_size: int = 8000        # 词汇表大小
    max_seq_len: int = 256        # 最大序列长度
    n_layers: int = 6             # Transformer 层数
    n_heads: int = 6              # 注意力头数
    d_model: int = 384            # 隐藏层维度
    d_ff: int = 1536              # FFN 中间维度（通常 4x d_model）
    dropout: float = 0.1          # Dropout 比率
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # 确保 d_model 能被 n_heads 整除
        assert self.d_model % self.n_heads == 0


class CausalSelfAttention(nn.Module):
    """因果自注意力 — 只看过去的命运之眼"""
    
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Q, K, V 投影（合并为一个线性层提高效率）
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # 因果掩码（下三角矩阵）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                 .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence Length, Channels
        
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力分数
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, T)
        
        # 应用因果掩码
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, 
            float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer 解码块 — 功法的基本单元"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        # Pre-Norm 结构（GPT-2 风格）
        x = x + self.attn(self.ln1(x))   # 残差连接
        x = x + self.mlp(self.ln2(x))    # 残差连接
        return x


class MiniGPT(nn.Module):
    """
    Mini GPT 语言模型
    从零构建的小型斗气结晶
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding 层
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer 层
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        
        # 输出层
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享：Embedding 和输出层共享权重（减少参数量）
        self.token_emb.weight = self.head.weight
        
        # 初始化参数
        self.apply(self._init_weights)
        
        # 统计参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量：{n_params / 1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token + Position Embedding
        tok_emb = self.token_emb(idx)                        # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)                          # (T, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer 层
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 输出 logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        # 计算 loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1  # 忽略 padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """
        自回归生成 — 命运推演
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_crop = idx[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature  # 取最后一个位置
            
            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx
```

### 5.5 训练循环

```python
"""
=== 训练循环核心代码 ===
"""

# 配置
config = MiniGPTConfig(
    vocab_size=8000,
    max_seq_len=256,
    n_layers=6,
    n_heads=6,
    d_model=384,
    d_ff=1536,
    dropout=0.1,
)

model = MiniGPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器（使用 AdamW，这是 GPT 训练的标配）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# 学习率调度：Warmup + Cosine Decay
from torch.optim.lr_scheduler import CosineAnnealingLR

warmup_steps = 100
max_steps = 5000

def get_lr(step):
    """学习率调度 — 火候控制"""
    if step < warmup_steps:
        # Warmup：缓慢升温
        return 3e-4 * step / warmup_steps
    else:
        # Cosine Decay：逐渐降温
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))

# 训练循环
model.train()
for step in range(max_steps):
    # 获取数据批次
    xb, yb = get_batch('train')  # 需自行实现
    xb, yb = xb.to(device), yb.to(device)
    
    # 更新学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 前向传播
    logits, loss = model(xb, yb)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪（防止反噬！）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 更新参数
    optimizer.step()
    
    # 日志
    if step % 100 == 0:
        print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f}")
    
    # 定期验证
    if step % 500 == 0:
        model.eval()
        val_loss = estimate_loss('val')  # 需自行实现
        print(f"  → Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
        model.train()
    
    # 定期生成样本
    if step % 1000 == 0:
        model.eval()
        prompt = torch.tensor([[tokenizer.token_to_id("<bos>")]], device=device)
        generated = model.generate(prompt, max_new_tokens=100)
        text = tokenizer.decode(generated[0].tolist())
        print(f"  → 生成样本：{text[:200]}")
        model.train()
```

### 5.6 评估指标：Perplexity（困惑度）

**Perplexity（困惑度）**是评估语言模型的核心指标：

```
Perplexity = exp(average cross-entropy loss)
           = exp(-1/N * Σ log P(x_t | x_1, ..., x_{t-1}))
```

直觉理解：**Perplexity 表示模型在预测下一个 token 时的 "平均困惑程度"**。

```
Perplexity = 1:    模型完全确定下一个词（完美预测）
Perplexity = 10:   模型平均在 10 个候选词中犹豫
Perplexity = 100:  模型平均在 100 个候选词中犹豫
Perplexity = V:    模型完全随机猜测（V 为词汇表大小）
```

> *"困惑度如同考核学徒的成绩——困惑度越低，说明学徒对丹方的理解越深。困惑度为 1 意味着他能完美预测每一味药材；困惑度等于药材种类总数，意味着他不过是在随机猜测。"*

常见语言模型的 Perplexity 参考值：

| 模型 | 数据集 | Perplexity |
|------|--------|------------|
| GPT-2 (Small) | WikiText-2 | ~29.4 |
| GPT-2 (Medium) | WikiText-2 | ~22.8 |
| GPT-2 (Large) | WikiText-2 | ~19.9 |
| GPT-2 (XL) | WikiText-2 | ~18.3 |

```python
"""
=== 计算 Perplexity ===
"""
import math

@torch.no_grad()
def calculate_perplexity(model, dataloader, device):
    """
    计算模型困惑度 — 检验结晶纯度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        logits, loss = model(input_ids, targets)
        
        # 统计有效 token 数（排除 padding）
        valid_tokens = (targets != -1).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    return perplexity
```

### 5.7 训练过程监控

训练语言模型时，需要监控以下关键指标：

```python
"""
=== 训练监控要点 ===
"""

# 1. Loss 曲线：应该平稳下降
#    - 如果 Loss 震荡剧烈 → 学习率太大
#    - 如果 Loss 下降极慢 → 学习率太小
#    - 如果 Loss 突然变成 NaN → 反噬！检查梯度

# 2. 学习率曲线：Warmup + Cosine Decay
#    - Warmup 防止初期大梯度导致训练不稳定
#    - Cosine Decay 让后期精细调整

# 3. 梯度范数：
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#    - 正常范围：0.1 ~ 10
#    - 如果持续很大（>100）：模型可能即将反噬
#    - 如果持续很小（<0.001）：可能梯度消失

# 4. 生成样本质量：定期查看模型输出
#    - 初期：乱码
#    - 中期：有结构但语义混乱
#    - 后期：语法通顺，语义基本合理

# 5. 训练 Loss vs 验证 Loss
#    - 两者同步下降：正常
#    - 训练 Loss 下降但验证 Loss 上升：过拟合！
#    - 两者都不下降：模型容量不足或数据有问题
```

> **避坑符 3.4**：从零训练的常见反噬
>
> | 现象 | 原因 | 解决方案 |
> |------|------|---------|
> | Loss 变 NaN | 学习率过大 / 数据异常 | 降低 LR，检查数据中的异常值 |
> | Loss 不下降 | LR 太小 / 模型太小 / 数据有问题 | 调大 LR，增加模型容量，检查数据管道 |
> | 生成重复 | Temperature 太低 / 训练不足 | 提高 Temperature，增加训练步数 |
> | OOM | Batch Size 太大 / 序列太长 | 减小 Batch Size，使用梯度累积 |
> | 生成乱码 | Tokenizer 和数据不匹配 | 确保 Tokenizer 在同一数据上训练 |

---

## 附录 C：凝晶心法 — 训练进阶技巧与数据工程

> *"凝晶之法，火候为先。本附录将传授你在预训练和微调过程中的高级调火之术——让每一份灵气都发挥最大效用。"*

### C.1 学习率调度：火候之道

预训练和微调的成功，极大程度上取决于学习率的调度策略。

#### Warmup + Cosine Decay — 最经典的选择

```python
"""
=== 丹方 3.C.1：学习率调度策略 ===
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0
):
    """
    Warmup + Cosine Decay — 标准预训练学习率调度
    如同炼丹：先文火慢热（warmup），后逐步降温（cosine decay）
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_linear_warmup_decay(optimizer, num_warmup_steps, num_training_steps):
    """线性预热 + 线性衰减（适用于微调）"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


# 可视化不同调度策略
print("""
╔══════════════════════════════════════════════════════════╗
║            学习率调度策略选择指南                         ║
╠══════════════════════════════════════════════════════════╣
║                                                         ║
║  预训练（从零训练）：                                     ║
║    → Warmup + Cosine Decay（最推荐）                     ║
║    → Warmup 步数：总步数的 1-10%                         ║
║    → Peak LR: 1e-4 ~ 6e-4（取决于 batch size）          ║
║                                                         ║
║  微调预训练模型：                                        ║
║    → Warmup + Linear Decay                              ║
║    → Warmup 步数：总步数的 6-10%                         ║
║    → Peak LR: 1e-5 ~ 5e-5（比预训练小 10 倍）           ║
║                                                         ║
║  差分学习率（层 LR 不同）：                               ║
║    → Embedding/底层: 1e-5（小学习率，少改动）             ║
║    → 中间层: 2e-5                                       ║
║    → 顶层/分类头: 5e-5（大学习率，快速适应）              ║
║                                                         ║
╚══════════════════════════════════════════════════════════╝
""")
```

#### 差分学习率（Differential Learning Rates）

```python
"""
=== 差分学习率 — 不同层使用不同学习率 ===
底层保留预训练知识（小 LR），顶层快速适应新任务（大 LR）
"""
def get_differential_optimizer(model, base_lr=2e-5):
    """为 BERT 模型的不同层设置差分学习率"""
    param_groups = [
        {"params": model.bert.embeddings.parameters(), "lr": base_lr * 0.1},
    ]
    n_layers = len(model.bert.encoder.layer)
    for i, layer in enumerate(model.bert.encoder.layer):
        lr_mult = 0.5 + 0.5 * (i / (n_layers - 1))
        param_groups.append({"params": layer.parameters(), "lr": base_lr * lr_mult})
    if hasattr(model, 'classifier'):
        param_groups.append({"params": model.classifier.parameters(), "lr": base_lr * 2.0})
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
optimizer = get_differential_optimizer(model, base_lr=2e-5)
print("差分学习率配置完成：底层 2e-6 → 中间层 2e-5 → 顶层 4e-5")
```

### C.2 梯度检查点：内存压缩术

预训练大模型时，显存往往不够用。**梯度检查点（Gradient Checkpointing）** 以时间换空间：

```
标准前向传播：
  存储所有中间激活值 → 反向传播时直接读取
  显存：O(n_layers × batch × seq_len × hidden)
  时间：1×

梯度检查点：
  只保存部分层的输出 → 需要时重新计算
  显存：减少 60-80%
  时间：~1.3×（增加约 30% 训练时间）
```

```python
"""
=== 梯度检查点 ===
"""
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    """
    带梯度检查点的 Transformer Block
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(0.1),
        )

    def _custom_forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x):
        return checkpoint(self._custom_forward, x, use_reentrant=False)


# HuggingFace 一行开启
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-chinese")
model.gradient_checkpointing_enable()
print("梯度检查点已开启，显存占用将减少 60-80%")
```

### C.3 数据工程深化

> *"药材质量决定丹药品质。数据工程是预训练中经常被低估但极为关键的环节。"*

```python
"""
=== 大规模数据工程流水线 ===
"""
import hashlib
import re

class DataPipeline:
    """从原始语料到训练数据的完整流水线"""
    def __init__(self, min_length=50, max_length=100000):
        self.min_length = min_length
        self.max_length = max_length

    def deduplicate(self, documents):
        """文档去重 — 精确哈希 + 指纹去重"""
        seen = set()
        unique = []
        for doc in documents:
            h1 = hashlib.md5(doc.encode('utf-8')).hexdigest()
            h2 = hashlib.md5(doc[:200].encode('utf-8')).hexdigest()
            if h1 not in seen and h2 not in seen:
                seen.add(h1)
                seen.add(h2)
                unique.append(doc)
        return unique

    def filter_quality(self, documents):
        """质量过滤"""
        filtered = []
        for doc in documents:
            if len(doc) < self.min_length or len(doc) > self.max_length:
                continue
            # 中文比例检查
            cn = len(re.findall(r'[\u4e00-\u9fff]', doc))
            if len(doc) > 0 and cn / len(doc) < 0.3:
                continue
            # 特殊字符检查
            sp = len(re.findall(r'[<>\[\]{}#$%^&*]', doc))
            if len(doc) > 0 and sp / len(doc) > 0.1:
                continue
            # 重复行检查
            lines = doc.split('\n')
            if len(lines) > 10 and len(set(lines)) / len(lines) < 0.5:
                continue
            filtered.append(doc)
        return filtered

    def clean_text(self, text):
        """文本清洗"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def process(self, documents):
        documents = [self.clean_text(doc) for doc in documents]
        documents = self.deduplicate(documents)
        documents = self.filter_quality(documents)
        return documents

print("""
╔══════════════════════════════════════════════════════════════╗
║              数据质量 vs 预训练效果                         ║
╠══════════════════════════════════════════════════════════════╣
║  数据规模（控制质量）：                                     ║
║    1B tokens  → PPL ~45     10B tokens → PPL ~25          ║
║  数据质量（控制 ~10B tokens）：                             ║
║    原始网页      → PPL ~30  +基础去重    → PPL ~27         ║
║    +质量过滤     → PPL ~23  +精细清洗    → PPL ~20         ║
║  关键发现：                                                 ║
║    "10B 高质量 tokens > 100B 低质量 tokens"                 ║
║  最佳数据配比：                                             ║
║    网页 60% + 书籍 20% + 学术论文 10% + 代码 10%            ║
╚══════════════════════════════════════════════════════════════╝
""")
```

### C.4 训练诊断面板

```python
"""
=== 训练过程诊断工具 ===
"""
import math

class TrainingDiagnostics:
    """训练过程诊断 — 如同炼丹师的诊断手册"""

    @staticmethod
    def diagnose(loss_history, val_loss_history=None):
        print("=== 训练诊断报告 ===\n")

        # NaN 检查
        if any(math.isnan(l) or math.isinf(l) for l in loss_history):
            print("⚠️  检测到 NaN/Inf！降低 LR，检查数据异常值")
            return

        # 下降趋势
        if len(loss_history) >= 10:
            recent = sum(loss_history[-10:]) / 10
            early = sum(loss_history[:10]) / 10
            imp = (early - recent) / early
            if imp < 0.01:
                print("⚠️  损失未下降：LR 太小 / 模型容量不足")
            elif imp > 0.5:
                print(f"✅ 损失下降 {imp:.1%}，进展良好")
            else:
                print(f"ℹ️  损失下降 {imp:.1%}，正常范围")

        # 震荡检测
        if len(loss_history) >= 20:
            recent = loss_history[-20:]
            mean_l = sum(recent) / len(recent)
            std_l = (sum((l - mean_l)**2 for l in recent) / len(recent)) ** 0.5
            cv = std_l / mean_l if mean_l > 0 else 0
            if cv > 0.3:
                print("⚠️  损失震荡剧烈！增大 Batch Size 或降低 LR")
            else:
                print(f"✅ 损失平稳（变异系数 = {cv:.3f}）")

        # 过拟合检测
        if val_loss_history and len(val_loss_history) >= 5:
            if min(val_loss_history[-5:]) > min(val_loss_history):
                print("⚠️  可能过拟合！增加 Dropout 或 Early Stopping")
            else:
                print("✅ 训练/验证损失同步下降，无过拟合")

        print("\n梯度范数参考范围: 0.1 ~ 10.0")
        print("  < 0.01 → 梯度消失 | > 100 → 梯度爆炸")


# 示例
losses = [8.5, 7.2, 6.1, 5.3, 4.7, 4.2, 3.8, 3.5, 3.3, 3.1,
          2.9, 2.8, 2.7, 2.65, 2.6, 2.55, 2.5, 2.48, 2.45, 2.43]
TrainingDiagnostics.diagnose(losses)
```

> *"凝晶之法的核心心法：数据为药材，模型为丹炉，学习率为火候。三者缺一不可。药材要精（高质量数据），丹炉要稳（正确的架构），火候要准（恰当的学习率调度）。修炼至此，你已不再是盲目开炉的学徒，而是能够精确掌控每一个细节的炼丹大师。"*

---

---

## 附录 D：天地法则 — Scaling Law 与大模型涌现

> *"修炼者的斗气容量，并非线性增长。当修炼到某个临界点，量变终将引发质变——这是天地间最根本的法则。"*

### D.1 Scaling Law：模型规模的铁律

2020年，Kaplan 等人在 OpenAI 的论文《Scaling Laws for Neural Language Models》中发现了一条令人震惊的规律：

**模型性能（以 Loss 衡量）随三个因素呈幂律下降：**
- 模型参数量 N
- 训练数据量 D
- 计算量 C

```
L(N) ∝ N^(-0.076)   — 参数量每增大10倍，Loss 下降约 17%
L(D) ∝ D^(-0.095)   — 数据量每增大10倍，Loss 下降约 20%
L(C) ∝ C^(-0.050)   — 计算量每增大10倍，Loss 下降约 11%
```

```python
# === Scaling Law 可视化 ===
import math

def scaling_loss(base_loss, factor, exponent):
    """计算 Scaling Law 下的 Loss"""
    return base_loss * (factor ** exponent)

# Chinchilla 最优比例
def chinchilla_optimal(model_params_billion, tokens_per_param=20):
    """
    Chinchilla 最优训练数据量
    每个模型参数需要约 20 个 token 的训练数据
    """
    total_tokens = model_params_billion * 1e9 * tokens_per_param
    return total_tokens

# 不同规模的最优配置
print("=" * 60)
print("  Chinchilla 最优训练配置")
print("=" * 60)
print(f"  {'模型':12s} {'参数量':>10s} {'最优数据量':>12s} {'估算Loss':>10s}")
print("-" * 60)

configs = [
    ("小模型", 0.1),    # 100M
    ("GPT-2 级", 1.5),  # 1.5B
    ("LLaMA-7B", 7),
    ("Qwen-14B", 14),
    ("LLaMA-33B", 33),
    ("GPT-3 级", 175),
    ("PaLM 级", 540),
]

base_loss = 3.5  # 基准 Loss
for name, params in configs:
    opt_tokens = chinchilla_optimal(params)
    opt_billion = opt_tokens / 1e9
    # Loss 估算（简化公式）
    est_loss = base_loss * (params ** (-0.076))
    print(f"  {name:12s} {params:>8.1f}B {opt_billion:>10.0f}B tok {est_loss:>10.3f}")

print()
print("核心发现：")
print("  1. 参数和数据应该同步增长（Chinchilla 定律）")
print("  2. 过大的模型 + 过少的数据 = 浪费算力")
print("  3. 过小的模型 + 过多的数据 = 同样浪费")
print("  4. 最优比例：每1个参数 ≈ 20个训练token")
```

### D.2 涌现能力：量变到质变的临界点

当模型规模超过某个阈值时，会突然展现出之前小模型完全不具备的能力——这就是**涌现（Emergence）**。

```
┌─────────────────────────────────────────────────────────────────┐
│                    涌现能力临界点示意                              │
│                                                                  │
│  能力表现                                                       │
│    ▲                                                            │
│    │              ╱ 涌现！多步推理                               │
│    │            ╱╱                                               │
│    │          ╱╱                                                 │
│    │        ╱╱    ╱ 代码生成                                    │
│    │      ╱╱    ╱╱                                               │
│    │    ╱╱    ╱╱  ╱ 指令遵循                                   │
│    │  ╱╱    ╱╱  ╱╱                                               │
│    │╱╱    ╱╱  ╱╱╱  ╱ 文本补全（非涌现）                        │
│    └──────────────────────────────────────→ 模型参数量           │
│         1B   7B   13B   30B   70B   175B                       │
│                                                                  │
│  非涌现能力：随规模平滑提升（如语言流畅度）                      │
│  涌现能力：在特定规模突然跳跃（如多步数学推理）                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# === 涌现能力的量化特征 ===

# 涌现能力典型案例
emergent_abilities = {
    "多步算术推理": {
        "阈值": "~70B 参数",
        "表现": "小模型几乎无法完成 3 位数的加减法，70B+ 模型突然可以",
        "论文": "Wei et al., 2022 (Chain-of-Thought Prompting)"
    },
    "思维链推理": {
        "阈值": "~60B 参数",
        "表现": "给定中间推理步骤后，大模型能完成复杂逻辑推理",
        "论文": "Wei et al., 2022"
    },
    "少样本指令遵循": {
        "阈值": "~6B 参数",
        "表现": "仅看 3-5 个示例就能理解任务要求",
        "论文": "Brown et al., 2020 (GPT-3)"
    },
    "代码生成与调试": {
        "阈值": "~7B 参数",
        "表现": "能生成可运行的 Python 函数，大模型能理解复杂逻辑",
        "论文": "Chen et al., 2021 (Codex)"
    },
}

print("已确认的涌现能力：\n")
for ability, info in emergent_abilities.items():
    print(f"  {ability}")
    print(f"    临界点: {info['阈值']}")
    print(f"    现象:   {info['表现']}")
    print(f"    来源:   {info['论文']}")
    print()
```

### D.3 DeepSpeed 配置实战

在凝晶篇训练小模型时，DeepSpeed 是提升训练效率和稳定性的重要工具。

```python
# === DeepSpeed ZeRO Stage 2 配置示例 ===
# 保存为: ds_config.json

ds_config_stage2 = """
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "CosineAnnealingLR",
        "params": {
            "total_num_steps": "auto",
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
"""

# === DeepSpeed ZeRO Stage 3 配置（大模型必备）===
ds_config_stage3 = """
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
"""

print("DeepSpeed ZeRO Stage 对比：\n")
print("  Stage 0: 无分片（等同于普通 DDP）")
print("  Stage 1: 分片优化器状态（省约 4x 显存）")
print("  Stage 2: 分片优化器状态 + 梯度（省约 8x 显存）")
print("  Stage 3: 分片参数 + 优化器状态 + 梯度（省约 N× 显存，N=GPU数）")
print()
print("推荐：")
print("  - < 7B 模型、单卡/双卡 → Stage 2")
print("  - 7B+ 模型、多卡训练 → Stage 3")
print("  - 极大模型（70B+）→ Stage 3 + Activation Checkpointing")
```

```python
# === 使用 DeepSpeed 进行训练 ===
# 安装: pip install deepspeed

# 方式一：通过 transformers Trainer 使用 DeepSpeed
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,

    # DeepSpeed 配置
    deepspeed="./ds_config_stage2.json",
    # 或者直接使用字符串（Trainer 会自动创建配置文件）
    # deepspeed="zero2",  # 简写形式
)

# 方式二：命令行启动
# deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json

# === DeepSpeed 核心参数调优 ===
print("DeepSpeed 调优要点：")
print()
print("  1. gradient_accumulation_steps:")
print("     = (目标 global batch size) / (GPU数 × per_device_batch_size)")
print("     例: global_bs=128, 4 GPUs, micro_bs=4 → grad_acc=8")
print()
print("  2. ZeRO Stage 选择:")
print("     - 显存够用 → Stage 2（通信开销更小）")
print("     - 显存紧张 → Stage 3（通信开销更大但更省显存）")
print()
print("  3. offload_optimizer:")
print("     - 开启后将优化器状态卸载到 CPU（ZeRO-Offload）")
print("     - 用少量速度换取大量显存")
print("     - 适合: 显存极度紧张的微调场景")
```

### D.4 训练效率优化技巧

```python
# === 大模型训练效率优化清单 ===

optimization_checklist = {
    "混合精度": {
        "技巧": "使用 BF16 而非 FP16",
        "原因": "BF16 动态范围更大，几乎不会溢出，A100/H100 原生支持",
        "代码": "torch_dtype=torch.bfloat16",
    },
    "梯度检查点": {
        "技巧": "用计算换显存，只保存部分激活值",
        "原因": "显存减少 60-70%，训练速度降低 20-30%",
        "代码": "gradient_checkpointing_enable()",
    },
    "Flash Attention": {
        "技巧": "IO 感知注意力算法",
        "原因": "显存从 O(N²) 降到 O(N)，速度提升 2-4x",
        "代码": "attn_implementation='flash_attention_2'",
    },
    "数据预取": {
        "技巧": "在 GPU 计算时 CPU 提前加载下一批数据",
        "原因": "避免 GPU 空等数据",
        "代码": "dataloader_kwargs={'num_workers': 4, 'prefetch_factor': 2}",
    },
    "编译优化": {
        "技巧": "torch.compile() 编译模型图",
        "原因": "PyTorch 2.0+ 可自动融合算子，提速 10-30%",
        "代码": "model = torch.compile(model)",
    },
}

print("训练效率优化清单：\n")
for i, (name, info) in enumerate(optimization_checklist.items(), 1):
    print(f"  {i}. {name}")
    print(f"     技巧: {info['技巧']}")
    print(f"     原因: {info['原因']}")
    print(f"     代码: {info['代码']}")
    print()

# === 数据并行 vs 模型并行选型 ===
print("=" * 60)
print("  并行策略选型决策树")
print("=" * 60)
print("""
你的模型能装入单卡吗？
├── 能 → 用 DDP（数据并行）或 DeepSpeed ZeRO-2
│       优点：实现简单，接近线性加速
│       缺点：每卡都要存完整模型
│
└── 不能 → 单卡显存不够？
        ├── 差一点 → DeepSpeed ZeRO-3 + Gradient Checkpointing
        │            优点：可用多卡训练远大于单卡的模型
        │            缺点：通信开销大
        │
        ├── 差很多 → FSDP（Fully Sharded Data Parallel）
        │            优点：PyTorch 原生，ZeRO-3 的官方实现
        │            缺点：配置稍复杂
        │
        └── 差非常远（70B+ 预训练）→ 3D 并行
                     ├─ 数据并行（DP）：不同数据
                     ├─ 张量并行（TP）：切分单层
                     └─ 流水线并行（PP）：切分层
""")
```

---

## 附录 E：凝晶实战 — 使用 HuggingFace Trainer 完整训练流程

> *"从零到一，完整走一遍凝晶之旅，方能真正理解每一层的含义。"*

### E.1 端到端训练脚本

```python
# === 完整的 HuggingFace Trainer 训练流程 ===
# 此脚本展示了从数据准备到模型评估的完整流程
# 适用于 BERT/GPT-2 级别的模型训练

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import numpy as np
import evaluate

# ========== 第一步：选择基础模型 ==========
model_name = "bert-base-uncased"  # 或 "gpt2" 用于生成任务
tokenizer = AutoTokenizer.from_pretrained(model_name)

if "gpt" in model_name:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 第二步：准备数据 ==========
# 示例：文本分类任务（情感分析）
raw_dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

# 小数据子集（快速验证）
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_test = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

# ========== 第三步：加载模型 ==========
num_labels = 2  # 正面 / 负面
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

# ========== 第四步：定义评估指标 ==========
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# ========== 第五步：配置训练参数 ==========
training_args = TrainingArguments(
    output_dir="./results/imdb-classifier",
    num_train_epochs=3,

    # 批次与梯度累积
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,

    # 优化器配置
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,

    # 精度
    fp16=False,   # 有 NVIDIA GPU 时设为 True
    bf16=False,   # A100/H100 时设为 True

    # 日志与保存
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    # 性能优化
    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    # 其他
    report_to="none",
    seed=42,
)

# ========== 第六步：创建 Trainer 并训练 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 开始训练
print("=" * 60)
print("  开始凝晶修炼...")
print("=" * 60)
train_result = trainer.train()

# ========== 第七步：评估与保存 ==========
metrics = train_result.metrics
print(f"\n训练完成！")
print(f"  总训练时间: {metrics['train_runtime']:.2f} 秒")
print(f"  训练 Loss:  {metrics['train_loss']:.4f}")

# 最终评估
eval_results = trainer.evaluate()
print(f"  评估准确率: {eval_results['eval_accuracy']:.4f}")

# 保存模型
trainer.save_model("./results/imdb-classifier/final")
tokenizer.save_pretrained("./results/imdb-classifier/final")
print("  模型已保存至 ./results/imdb-classifier/final")

# ========== 第八步：推理测试 ==========
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="./results/imdb-classifier/final",
    tokenizer=tokenizer,
)

test_texts = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible film. Complete waste of time. I fell asleep.",
    "It was okay, nothing special but not bad either.",
]

print("\n推理测试：")
for text in test_texts:
    result = classifier(text)[0]
    label = "正面" if result["label"] == "POSITIVE" else "负面"
    print(f"  [{label} {result['score']:.3f}] {text[:50]}...")
```

### E.2 Trainer 核心参数速查

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `learning_rate` | 学习率 | 2e-5（BERT类）/ 5e-5（GPT类） |
| `num_train_epochs` | 训练轮数 | 3-5（看 Loss 曲线） |
| `per_device_train_batch_size` | 每卡批次大小 | 尽量大（受限于显存） |
| `gradient_accumulation_steps` | 梯度累积步数 | = 目标BS / (GPU数 × per_device_BS) |
| `warmup_ratio` | 预热比例 | 0.06（训练步数的6%） |
| `lr_scheduler_type` | 学习率调度 | "cosine"（最常用） |
| `weight_decay` | 权重衰减 | 0.01（L2正则化） |
| `fp16 / bf16` | 混合精度 | 有GPU开启（优先bf16） |
| `gradient_checkpointing` | 梯度检查点 | 显存不够时开启 |
| `save_strategy` | 保存策略 | "epoch" 或 "steps" |
| `eval_strategy` | 评估策略 | "epoch"（配合early stopping） |
| `load_best_model_at_end` | 加载最优 | True（配合 eval） |

---



## 修炼总结与境界突破条件

### 本卷修炼成果

经过凝晶篇的修炼，你已掌握：

```
✓ 文字炼化之术（Tokenizer）
  - 理解 BPE、WordPiece、Unigram 三种子词算法的原理与区别
  - 能够训练自定义 Tokenizer
  - 理解特殊 Token 的作用

✓ 斗气空间之理（Embedding）
  - 理解从 One-Hot 到 Dense Embedding 的演进
  - 掌握 Word2Vec 的基本原理
  - 理解上下文 Embedding 的必要性
  - 理解 Position Embedding 的两种方案

✓ 双面法则（BERT）
  - 理解 MLM 和 NSP 两个预训练任务
  - 理解 BERT 的双向注意力机制
  - 能够使用 BERT 进行文本分类微调
  - 理解 Pre-train + Fine-tune 范式

✓ 因果法则（GPT）
  - 理解 Causal LM 的自回归预训练目标
  - 理解因果注意力掩码的实现
  - 掌握 Temperature, Top-k, Top-p 等生成策略
  - 理解 BERT 与 GPT 的本质区别与适用场景

✓ 首次凝晶（从零训练）
  - 完成从数据准备到模型训练的全流程
  - 理解训练循环的每个环节
  - 能够监控训练过程，诊断常见问题
  - 理解 Perplexity 作为评估指标的含义
```

### 境界自测

回答以下问题，检验你的修炼是否扎实：

1. **BPE 算法的核心思想是什么？它是自底向上还是自顶向下？**
   - 参考：从字符出发，不断合并频率最高的相邻对，自底向上

2. **BERT 的 MLM 任务中，为什么不把所有选中的 token 都替换为 [MASK]？**
   - 参考：避免预训练和微调之间的不匹配（微调时输入没有 [MASK]）

3. **GPT 的因果掩码是什么形状？为什么需要它？**
   - 参考：下三角矩阵。确保模型只能看到过去的 token，不能 "偷看" 未来

4. **Temperature 参数如何影响生成质量？T=0.1 和 T=2.0 分别会怎样？**
   - 参考：T=0.1 几乎等于贪心，输出确定但单调；T=2.0 非常随机，输出多样但可能不合理

5. **Perplexity 为 50 意味着什么？**
   - 参考：模型在预测下一个 token 时，平均在 50 个候选中犹豫

6. **为什么现代 LLM 几乎都采用 GPT 的 decoder-only 架构而非 BERT 的 encoder？**
   - 参考：生成能力更强，Scaling Law 效果更好，可统一处理各类任务

### 突破到第四卷的条件

在进入第四卷（化形篇 · 斗灵）之前，确保你能够：

```
□ 独立训练一个 BPE Tokenizer 并理解其工作原理
□ 用 BERT 完成至少一个文本分类任务
□ 用 GPT-2 生成连贯的文本段落
□ 从零训练一个小型语言模型（即使效果不完美）
□ 解释预训练-微调范式的核心思想
□ 理解 Perplexity 指标并能计算它
```

---

### 进入下一卷：化形篇

> *"斗气凝为结晶，已是大斗师之境。然而，结晶虽坚，却也庞大沉重。下一卷，你将修炼化形之术——以 LoRA、量化等精妙手法，将庞大的结晶化为精巧的形态，以最小的代价释放最大的力量。"*

第四卷将修炼：
- **LoRA / QLoRA**：以极少的参数撬动整个模型（Parameter-Efficient Fine-Tuning）
- **量化技术**：将 32 位浮点压缩为 4/8 位整数，大幅减少显存占用
- **Prompt Engineering**：不修改模型参数，仅通过精妙的输入引导模型行为

**[继续修炼 → 第四卷：化形篇（斗灵）](../04-化形篇-斗灵/README.md)**

---

*"斗气固化，结晶已成。昔日飘渺之气，今已凝为实体。此为大斗师之证，亦为更高境界之基。"*

**凝晶篇 · 完**

---

## 本卷增强补全（2026）— 预训练范式闭环 · 开源基座选型 · 最小 Recipe

> 本节为《焚诀》深度研究版的**回填内容**，用于补齐“预训练目标 → 能力涌现 → 评测与选型”的因果链，并给出可落地的预训练/续训检查清单。  
> 完整增强总览见：[焚诀-深度研究版-全卷增强补全（2026）](../焚诀-深度研究版-全卷增强补全.md)

### 1) 语言建模的两大范式：你到底在学什么？

- **自回归（Causal LM / GPT 系）**：  
  \\[
  P(x)=\\prod_t P(x_t\\mid x_{<t})
  \\]  
  适合生成、补全、工具调用；但要记住：它的目标是“高概率续写”，不是“事实检索”，幻觉是结构性风险。

- **掩码建模（MLM / BERT 系）**：预测被 mask 的 token。  
  更擅长表示学习与检索/分类；在 LLM 时代常作为 Embedding/Encoder 侧组件存在。

### 2) 开源基座模型选型四问（建议加入卷末“选型矩阵”）

1. **任务**：对话/代码/多语/检索/长上下文/多模态？  
2. **资源**：单机单卡？多机多卡？推理成本上限？  
3. **许可**：是否可商用？是否有分发/再训练限制？  
4. **生态**：推理引擎（vLLM/TensorRT-LLM）、微调工具链、权重格式与社区成熟度？

参考：Llama 3 官方发布信息：<https://ai.meta.com/blog/meta-llama-3/>

### 3) Best Practices：预训练/续训的“最小 Recipe”

- **数据**：去重（MinHash/SimHash）、质量过滤（困惑度/分类器/规则）、隐私脱敏（PII）  
- **训练**：batch/seq 配比、梯度累加、checkpoint 与恢复策略、学习率 warmup  
- **评测**：困惑度 + 下游任务（分类/检索/生成）+ 抽样人工审计  
- **风险**：benchmark contamination（训练数据污染评测）→ 需要评测集隔离与版本管理
