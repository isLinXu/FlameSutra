# 药材库 — 数据集指南

> *"药材者，炼丹之基。*
> *无上好药材，纵有天阶异火、绝世丹方，也炼不出神丹妙药。*
> *此册遍录天下药材，按品类分门，按品质评级。*
> *愿每一位炼丹师，都能为自己的丹方找到最合适的药材。"*
>
> — 《焚诀》附录·药材库

---

## 写在最前

**数据是大模型的灵魂。**

模型架构是斗技，训练方法是丹方，GPU 是异火，但没有好的数据（药材），
一切都是空中楼阁。本册全面梳理当前主流的数据集资源，
涵盖预训练、微调、偏好对齐、评测等各个阶段。

### 术语对照

| 修炼术语 | 技术含义 |
|---------|---------|
| 药材 | 数据集 |
| 天材地宝 | 高质量数据 |
| 炼化 | 数据预处理/清洗 |
| 药材品质 | 数据质量 |
| 药方 | 数据配比/混合策略 |
| 药材库 | 数据仓库 |
| 采药 | 数据收集 |
| 鉴药 | 数据质量评估 |

### 药材品质评级体系

| 品质 | 含义 | 说明 |
|------|------|------|
| ★★★★★ 极品 | 顶级品质 | 业界公认的高质量数据，被广泛使用 |
| ★★★★☆ 上品 | 高品质 | 质量较高，但有一定局限性 |
| ★★★☆☆ 中品 | 中等品质 | 可用但需要额外清洗或筛选 |
| ★★☆☆☆ 下品 | 较低品质 | 有明显质量问题，需谨慎使用 |
| ★☆☆☆☆ 凡品 | 基础品质 | 仅作补充使用 |

---

## 目录

- [第一章：筑基药材 — 预训练数据集](#第一章筑基药材--预训练数据集)
- [第二章：炼心药材 — SFT 微调数据集](#第二章炼心药材--sft-微调数据集)
- [第三章：淬体药材 — 偏好对齐数据集](#第三章淬体药材--偏好对齐数据集)
- [第四章：试丹之石 — 评估基准数据集](#第四章试丹之石--评估基准数据集)
- [第五章：万象药材 — 多模态数据集](#第五章万象药材--多模态数据集)
- [第六章：炼化之术 — 数据清洗工具](#第六章炼化之术--数据清洗工具)
- [附录：药材选用指南](#附录药材选用指南)

---

## 第一章：筑基药材 — 预训练数据集

> *"筑基之药，贵在量大质纯。*
> *预训练需要海量文本数据，如同筑基需要源源不断的天地灵气。"*

预训练数据集是模型的"基础功力"来源。需要特点：**量大、多样、干净**。

---

### 1.1 英文通用语料

#### Common Crawl

```
品质：★★★☆☆（原始数据噪声大，清洗后可达 ★★★★）
规模：PB 级别（原始数据），数万亿 tokens
语言：多语言（以英文为主）
许可：自由使用
获取：https://commoncrawl.org/

描述：
  互联网网页的大规模爬取，是几乎所有大规模预训练数据的基础。
  原始数据包含大量垃圾（广告、导航栏、重复内容等），
  必须经过严格清洗才能使用。

特点：
  ├── 覆盖面极广，几乎涵盖所有互联网内容
  ├── 每月定期更新
  ├── 需要大量清洗（去重、过滤、质量评估）
  └── 是 C4、FineWeb、RedPajama 等数据集的原料

用途：预训练的原始数据来源
```

#### The Pile

```
品质：★★★★☆
规模：825 GB，约 300B tokens
语言：英文
许可：MIT License（但部分子集有各自的许可）
获取：https://pile.eleuther.ai/

描述：
  EleutherAI 精心策划的 22 个高质量子集的混合数据集。
  是开源 LLM 预训练数据的经典之作。

22 个子集包括：
  ├── Pile-CC：清洗后的 Common Crawl
  ├── PubMed Central：生物医学论文
  ├── Books3：书籍
  ├── OpenWebText2：Reddit 高赞链接的网页
  ├── ArXiv：学术论文
  ├── Github：代码
  ├── FreeLaw：法律文本
  ├── StackExchange：技术问答
  ├── USPTO Backgrounds：专利
  ├── Wikipedia (en)：英文维基百科
  ├── PubMed Abstracts：医学摘要
  ├── Gutenberg (PG-19)：古典书籍
  ├── OpenSubtitles：电影字幕
  ├── DM Mathematics：数学问题
  ├── Ubuntu IRC：技术讨论
  ├── BookCorpus2：书籍
  ├── EuroParl：欧洲议会文本
  ├── HackerNews：技术新闻
  ├── YoutubeSubtitles：YouTube 字幕
  ├── PhilPapers：哲学论文
  ├── NIH ExPorter：医学研究
  └── Enron Emails：商业邮件

用途：预训练，GPT-NeoX、Pythia 等模型基于此训练
```

#### FineWeb / FineWeb-Edu

```
品质：★★★★★
规模：FineWeb: 15T tokens / FineWeb-Edu: 1.3T tokens
语言：英文
许可：ODC-By 1.0
获取：https://huggingface.co/datasets/HuggingFaceFW/fineweb
      https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

描述：
  HuggingFace 发布的超大规模高质量网页数据集。
  FineWeb 是对 96 个 Common Crawl dump 的精细清洗和去重。
  FineWeb-Edu 进一步用质量分类器筛选出"教育性"内容。

清洗流程：
  ├── URL 过滤：去除已知低质量网站
  ├── 文本提取：使用 trafilatura 提取正文
  ├── 语言识别：保留英文内容
  ├── 质量过滤：基于启发式规则过滤低质量文本
  ├── 去重：MinHash + LSH 近似去重
  └── 教育性评估（FineWeb-Edu）：训练分类器评估内容的教育价值

用途：当前最推荐的英文预训练数据集之一
```

#### RedPajama / RedPajama-v2

```
品质：★★★★☆
规模：v1: 1.2T tokens / v2: 30T tokens（原始）
语言：英文为主，含多语言
许可：Apache 2.0
获取：https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

描述：
  Together AI 发布的开源预训练数据集，旨在复现 LLaMA 的训练数据。
  v1 严格按照 LLaMA 论文中的数据配比构建。
  v2 大幅扩大规模并提供了质量信号标注。

数据配比（v1, 模仿 LLaMA）：
  ├── CommonCrawl: 878B tokens (72.6%)
  ├── C4: 175B tokens (14.4%)
  ├── GitHub: 59B tokens (4.9%)
  ├── Wikipedia: 24B tokens (2.0%)
  ├── Books: 26B tokens (2.1%)
  ├── ArXiv: 28B tokens (2.3%)
  └── StackExchange: 20B tokens (1.7%)

用途：预训练，可复现 LLaMA 的训练过程
```

#### SlimPajama

```
品质：★★★★☆
规模：627B tokens
语言：英文
许可：Apache 2.0
获取：https://huggingface.co/datasets/cerebras/SlimPajama-627B

描述：
  Cerebras 对 RedPajama v1 进行严格去重后的精简版本。
  通过 MinHash 去重，将数据量从 1.2T 精简到 627B，
  去除了约 49% 的重复内容。
  实验表明，去重后的数据训练效果更好。

用途：预训练，质量优于原始 RedPajama
```

#### StarCoder Data (The Stack)

```
品质：★★★★★（代码领域）
规模：The Stack v2: 约 4TB，涵盖 600+ 编程语言
语言：代码（多编程语言）
许可：各文件保留原始许可，提供 opt-out 机制
获取：https://huggingface.co/datasets/bigcode/the-stack-v2

描述：
  BigCode 项目收集的超大规模代码数据集。
  从 Software Heritage 存档中提取，
  包含许可信息和质量标注。

特点：
  ├── 600+ 编程语言
  ├── 保留了许可信息（可按许可过滤）
  ├── 提供 opt-out 机制（尊重开发者意愿）
  ├── 经过去重和质量过滤
  └── StarCoder 2 模型基于此训练

用途：代码 LLM 预训练
```

---

### 1.2 中文语料

#### WuDaoCorpora（悟道语料库）

```
品质：★★★★☆
规模：约 200GB 文本，5TB 多模态
语言：中文
许可：研究用途
获取：需向 BAAI 申请

描述：
  北京智源人工智能研究院（BAAI）构建的大规模中文语料库。
  是国内最早的大规模预训练数据集之一。

包含：
  ├── 百科知识
  ├── 新闻文章
  ├── 学术论文
  ├── 社交媒体
  └── 多模态数据

用途：中文模型预训练
```

#### MNBVC（Massive Never-ending BT Vast Chinese corpus）

```
品质：★★★★☆
规模：持续增长中，已超过数 TB
语言：中文
许可：各子集不同，具体查看 README
获取：https://huggingface.co/datasets/liwu/MNBVC

描述：
  社区驱动的超大规模中文语料库项目。
  目标是构建中文版的"The Pile"。

包含：
  ├── 网页数据
  ├── 新闻
  ├── 书籍
  ├── 代码
  ├── 百科
  ├── 论坛/问答
  └── 持续更新中

用途：中文模型预训练
```

#### CulturaX

```
品质：★★★★☆
规模：6.3T tokens，涵盖 167 种语言
语言：多语言（包含大量中文）
许可：各语言子集许可不同
获取：https://huggingface.co/datasets/uonlp/CulturaX

描述：
  基于 mC4 和 OSCAR 构建的大规模多语言预训练数据集。
  经过仔细的语言识别、去重和质量过滤。

用途：多语言模型预训练，特别适合需要中文能力的模型
```

---

## 第二章：炼心药材 — SFT 微调数据集

> *"筑基之后，需炼心——即让模型学会遵循指令、对话、完成任务。*
> *SFT 药材不需要太多，但必须精炼。"*

SFT（Supervised Fine-Tuning）数据集的特点：**指令-回答对，质量 > 数量**。

---

### 2.1 英文 SFT 数据集

#### Alpaca (Stanford)

```
品质：★★★☆☆
规模：52,000 条指令-回答对
语言：英文
许可：CC BY-NC 4.0
获取：https://huggingface.co/datasets/tatsu-lab/alpaca

描述：
  Stanford 使用 GPT-3.5（text-davinci-003）基于 175 个种子任务
  生成的指令微调数据集。是指令微调领域的先驱之一。

优点：简单易用，入门微调的经典选择
缺点：由单一模型生成，多样性有限；质量参差不齐

用途：入门级指令微调
```

#### ShareGPT

```
品质：★★★★☆
规模：约 90,000+ 条多轮对话
语言：英文为主，含多语言
许可：社区共享（许可不明确）
获取：https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

描述：
  用户分享的与 ChatGPT/GPT-4 的真实对话记录。
  Vicuna 模型即基于 ShareGPT 数据微调而来。

优点：来自真实用户对话，分布接近实际使用场景
缺点：许可问题不明确；可能包含有害内容；质量不均匀

用途：对话模型微调，多轮对话能力训练
```

#### OpenHermes 2.5

```
品质：★★★★★
规模：约 1,000,000 条指令数据
语言：英文
许可：Apache 2.0（数据部分许可取决于来源）
获取：https://huggingface.co/datasets/teknium/OpenHermes-2.5

描述：
  由 Teknium 整合多个来源构建的大规模高质量指令数据集。
  包含 GPT-4 生成数据、代码数据、数学数据等多种类型。

来源包括：
  ├── GPT-4 生成的各类任务数据
  ├── Airoboros（GPT-4 生成的复杂指令）
  ├── CamelAI（角色扮演对话）
  ├── 代码相关数据
  └── 数学和推理数据

用途：高质量通用指令微调
```

#### LIMA

```
品质：★★★★★（质量极高，但量少）
规模：仅 1,000 条
语言：英文
许可：CC BY-NC-SA
获取：https://huggingface.co/datasets/GAIR/lima

描述：
  Meta 的论文"LIMA: Less Is More for Alignment"中使用的数据集。
  核心观点：仅需 1,000 条精心策划的高质量数据，
  就能训练出与大量数据微调效果相当的模型。

数据来源：
  ├── Stack Exchange（精选高赞回答）
  ├── wikiHow（操作指南）
  ├── Reddit（精选高质量帖子）
  └── 手工编写的样本

用途：研究"少而精"的微调策略
```

#### Dolly 2.0

```
品质：★★★★☆
规模：约 15,000 条
语言：英文
许可：CC BY-SA 3.0（商业可用！）
获取：https://huggingface.co/datasets/databricks/databricks-dolly-15k

描述：
  Databricks 员工手工编写的指令数据集。
  覆盖 8 个任务类型：开放 QA、封闭 QA、信息提取、
  摘要、头脑风暴、分类、创意写作、其他。

优点：纯人类编写，许可友好（可商用）
缺点：数量较少，覆盖面有限

用途：商业场景的指令微调（许可无忧）
```

---

### 2.2 中文 SFT 数据集

#### BELLE

```
品质：★★★★☆
规模：约 2,000,000 条（多个版本合计）
语言：中文
许可：Apache 2.0
获取：https://huggingface.co/BelleGroup

描述：
  中文 LLM 微调数据集，由 GPT-3.5 生成，参考 Alpaca 方法。
  包含多个子集：0.5M、1M、2M 版本。
  覆盖翻译、编程、数学等多种任务。

用途：中文模型指令微调
```

#### Firefly

```
品质：★★★★☆
规模：约 1,650,000 条
语言：中文
许可：Apache 2.0
获取：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M

描述：
  流萤大模型的训练数据，包含 23 种中文 NLP 任务。
  任务类型包括：对话、摘要、翻译、代码、作文、
  诗歌、对联、文言文、阅读理解等。

特点：中文任务覆盖面极广，特别适合训练中文通用模型

用途：中文模型指令微调
```

---

## 第三章：淬体药材 — 偏好对齐数据集

> *"炼心之后需淬体——让模型学会分辨好与坏、安全与危险。*
> *偏好数据是 RLHF/DPO 的核心药材。"*

偏好数据集格式：**prompt + chosen（好回答）+ rejected（差回答）**

---

#### HH-RLHF (Anthropic)

```
品质：★★★★★
规模：约 170,000 条对比数据
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/Anthropic/hh-rlhf

描述：
  Anthropic 发布的人类偏好数据集，是 RLHF 领域的经典之作。
  包含两个维度：
  ├── Helpfulness（有用性）：哪个回答更有帮助
  └── Harmlessness（无害性）：哪个回答更安全

数据收集方式：
  人类评估者与模型对话，选择更好/更安全的回答。

用途：训练奖励模型，RLHF/DPO 对齐训练
```

#### UltraFeedback

```
品质：★★★★★
规模：约 64,000 条，每条包含 4 个回答的排序
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/openbmb/UltraFeedback

描述：
  由多个强模型（GPT-4, Claude, LLaMA 等）生成回答，
  然后由 GPT-4 进行质量评分和排序。

评分维度：
  ├── Instruction Following（指令遵循）
  ├── Truthfulness（真实性）
  ├── Honesty（诚实性）
  └── Helpfulness（有用性）

特点：
  ├── 每个 prompt 有 4 个不同模型的回答
  ├── 提供细粒度的评分（1-5 分）
  └── Zephyr 模型的训练数据之一

用途：DPO/RLHF 训练，奖励模型训练
```

#### Nectar

```
品质：★★★★☆
规模：约 182,000 条
语言：英文
许可：Apache 2.0
获取：https://huggingface.co/datasets/berkeley-nest/Nectar

描述：
  由 Berkeley 收集，每个 prompt 包含 7 个回答的排序。
  回答来自多个不同的模型。

特点：回答数量多（7 个），排序信息丰富

用途：奖励模型训练，偏好学习研究
```

#### Chatbot Arena 数据

```
品质：★★★★★（来自真实用户投票）
规模：持续增长中，已超过 1,000,000+ 投票
语言：多语言
许可：CC BY-NC 4.0（部分数据）
获取：https://huggingface.co/datasets/lmsys/chatbot_arena_conversations

描述：
  来自 LMSYS Chatbot Arena 的真实用户偏好数据。
  用户同时与两个匿名模型对话，然后投票选择更好的回答。

特点：
  ├── 数据来自真实用户的盲评
  ├── 覆盖大量模型（GPT-4, Claude, LLaMA, Gemini 等）
  ├── 持续增长，是最大的人类偏好数据集之一
  └── Elo 排名系统是 LLM 评测的金标准之一

用途：偏好对齐训练，奖励模型研究
```

---

## 第四章：试丹之石 — 评估基准数据集

> *"丹药炼成，需以试丹石验其品质。*
> *评估基准就是验证模型能力的试金石。"*

---

### 4.1 综合能力评估

#### MMLU (Massive Multitask Language Understanding)

```
品质：★★★★★（评估金标准之一）
规模：57 个学科，约 14,000+ 题
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/cais/mmlu

描述：
  覆盖 STEM、人文、社科等 57 个学科的多选题。
  难度从高中到研究生级别不等。
  是衡量模型知识广度的最常用基准之一。

学科覆盖：
  ├── STEM：数学、物理、化学、计算机科学、工程等
  ├── 人文：历史、哲学、法律等
  ├── 社科：经济学、心理学、社会学等
  └── 其他：医学、会计、营养学等

评估方式：多选题准确率
当前 SOTA：GPT-4o ~88%, Claude 3.5 Sonnet ~89%
```

#### HellaSwag

```
品质：★★★★☆
规模：约 70,000 题
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/Rowan/hellaswag

描述：
  常识推理基准。给定一个场景的开头，
  选择最合理的后续发展。
  使用"对抗过滤"方法生成困难的干扰项。

评估方式：选择最合理的续写
当前 SOTA：>95%（对大模型来说已基本饱和）
```

#### TruthfulQA

```
品质：★★★★★
规模：817 道题
语言：英文
许可：Apache 2.0
获取：https://huggingface.co/datasets/truthful_qa

描述：
  测试模型是否会给出虚假但看似合理的回答。
  专门针对 LLM 容易"一本正经胡说八道"的弱点设计。

题目覆盖：
  ├── 常见误解（"金鱼的记忆只有 7 秒"→ 错误）
  ├── 阴谋论
  ├── 迷信
  ├── 逻辑陷阱
  └── 常见的事实性错误

评估方式：回答的真实性和信息量
```

---

### 4.2 推理能力评估

#### GSM8K

```
品质：★★★★★
规模：8,500 道小学数学应用题
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/openai/gsm8k

描述：
  OpenAI 发布的小学级别数学推理基准。
  每道题需要 2-8 步推理，不涉及复杂数学知识。
  是测试 LLM 基本数学推理能力的标准基准。

示例：
  "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast
  every morning and bakes muffins for friends using 4. She sells
  the remainder at $2 each. How much does she earn per day?"
  答案：(16 - 3 - 4) * 2 = $18

当前 SOTA：>97%（DeepSeek-R1, o1 等推理模型）
```

#### MATH

```
品质：★★★★★
规模：12,500 道竞赛级数学题
语言：英文
许可：MIT License
获取：https://huggingface.co/datasets/hendrycks/competition_math

描述：
  Hendrycks 等人收集的数学竞赛题目。
  难度远超 GSM8K，涵盖代数、几何、数论、
  组合数学、概率等领域。
  分为 5 个难度级别。

当前 SOTA：~97%（DeepSeek-R1, o1）
注意：仅推理模型能达到如此高分，标准 LLM 通常在 50-80% 之间
```

#### ARC (AI2 Reasoning Challenge)

```
品质：★★★★☆
规模：7,787 道科学题
语言：英文
许可：CC BY-SA 4.0
获取：https://huggingface.co/datasets/allenai/ai2_arc

描述：
  来自美国小学到初中科学考试的多选题。
  分为 Easy 和 Challenge 两个子集。
  Challenge 子集专门收录了检索模型和共现方法难以回答的题目。

用途：测试科学常识和推理能力
```

#### HumanEval

```
品质：★★★★★（代码领域）
规模：164 道编程题
语言：Python
许可：MIT License
获取：https://huggingface.co/datasets/openai/openai_humaneval

描述：
  OpenAI 发布的代码生成基准。
  给定函数签名和文档字符串，模型需要生成完整的函数实现。
  使用 pass@k 指标评估（k 次尝试中至少一次正确的概率）。

当前 SOTA（pass@1）：>92%（Claude 3.5 Sonnet, DeepSeek-R1）
```

---

### 4.3 中文评估基准

#### C-Eval

```
品质：★★★★★（中文评估金标准）
规模：13,948 道题，52 个学科
语言：中文
许可：CC BY-NC-SA 4.0
获取：https://huggingface.co/datasets/ceval/ceval-exam

描述：
  中文版 MMLU。覆盖从初中到研究生、从人文到理工的 52 个学科。
  是评估中文模型综合能力的首选基准。

学科分布：
  ├── STEM（20 个学科）：数学、物理、化学、计算机等
  ├── 社科（10 个学科）：政治、经济、法律等
  ├── 人文（10 个学科）：语文、历史、地理等
  └── 其他（12 个学科）：医学、教育、会计等

评估方式：多选题准确率
```

#### CMMLU

```
品质：★★★★★
规模：11,528 道题，67 个学科
语言：中文
许可：CC BY-NC 4.0
获取：https://huggingface.co/datasets/haonan-li/cmmlu

描述：
  另一个中文综合评估基准，学科覆盖比 C-Eval 更广。
  特别增加了中国特色内容（中国历史、中医、中国法律等）。

特点：
  ├── 67 个学科（比 C-Eval 的 52 个更多）
  ├── 包含中国特有的知识领域
  └── 与 C-Eval 互补使用效果最佳

用途：中文模型综合能力评估
```

---

## 第五章：万象药材 — 多模态数据集

> *"万法归一之道，需要万象药材——文字、图像、声音、影像，无所不包。"*

---

#### LAION-5B

```
品质：★★★★☆
规模：约 58 亿图文对
语言：多语言
许可：CC-BY 4.0（元数据），图像版权归原作者
获取：https://laion.ai/blog/laion-5b/

描述：
  当前最大的开放图文对数据集。
  Stable Diffusion 即基于 LAION-5B 的子集训练。

子集：
  ├── LAION-2B-en：20 亿英文图文对
  ├── LAION-2B-multi：20 亿多语言图文对
  ├── LAION-1B-nolang：10 亿未标注语言的图文对
  └── LAION-Aesthetics：按美学评分筛选的子集

注意：
  由于包含网络爬取的图像，可能存在版权和内容安全问题。
  2023 年曾因被发现包含 CSAM 内容而引发争议并临时下线。

用途：文生图模型训练、CLIP 对比学习
```

#### ShareGPT4V

```
品质：★★★★★
规模：约 100,000 条高质量图文描述
语言：英文
许可：Apache 2.0
获取：https://huggingface.co/datasets/Lin-Chen/ShareGPT4V

描述：
  使用 GPT-4V 对图像生成详细描述的数据集。
  描述质量远超普通图文对数据（LAION 等），
  因为 GPT-4V 能识别图像中的细节、空间关系、文字等。

用途：多模态 LLM 微调（如 LLaVA 1.6）
```

#### LLaVA-Instruct

```
品质：★★★★★
规模：约 665,000 条视觉指令数据
语言：英文
许可：Apache 2.0
获取：https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

描述：
  LLaVA 模型的训练数据。使用 GPT-4 基于图像的描述
  生成多种类型的视觉问答和对话数据。

包含：
  ├── 视觉问答（Visual QA）
  ├── 详细描述（Detailed Description）
  └── 复杂推理（Complex Reasoning）

用途：训练视觉语言模型（VLM）
```

---

## 第六章：炼化之术 — 数据清洗工具

> *"再好的药材，若不经炼化去除杂质，也会让丹药带毒。*
> *数据清洗是炼丹过程中最被低估但最重要的环节。"*

---

### 6.1 数据清洗工具总览

| 工具 | 开发者 | 特点 | 适用场景 |
|------|--------|------|---------|
| **data-juicer** | Alibaba | 全面的数据处理框架，100+ 算子 | 通用数据清洗 |
| **datatrove** | HuggingFace | FineWeb 的清洗流程 | 大规模网页清洗 |
| **RedPajama pipeline** | Together AI | RedPajama 的清洗工具 | 预训练数据准备 |
| **dolma** | AI2 | OLMo 模型的数据工具 | 预训练数据准备 |
| **NeMo Curator** | NVIDIA | GPU 加速的数据处理 | 大规模高效清洗 |
| **text-dedup** | Google Research | 专注文本去重 | 去重处理 |

---

### 6.2 data-juicer

```
项目地址：https://github.com/modelscope/data-juicer

描述：
  阿里巴巴开发的全面数据处理框架。
  提供 100+ 算子，覆盖数据清洗的全流程。

核心功能：
  ├── Formatter：数据格式转换（JSON, Parquet, CSV 等）
  ├── Mapper：数据转换（文本清洗、语言检测、脱敏等）
  ├── Filter：数据过滤（长度过滤、语言过滤、质量过滤等）
  ├── Deduplicator：去重（精确去重、模糊去重）
  └── Selector：数据选择（基于质量分数的采样等）

使用示例：
  # 安装
  pip install py-data-juicer

  # 配置文件 config.yaml
  dataset_path: './data/input.jsonl'
  export_path: './data/output.jsonl'

  process:
    - language_id_score_filter:    # 语言识别
        lang: 'zh'
        min_score: 0.8
    - text_length_filter:          # 长度过滤
        min_len: 100
        max_len: 100000
    - word_repetition_filter:      # 重复词过滤
        rep_len: 10
        max_ratio: 0.5
    - document_deduplicator:       # 文档级去重
        lowercase: true
    - minhash_deduplicator:        # MinHash 去重
        tokenization: 'space'
        window_size: 5
        num_permutations: 256
        jaccard_threshold: 0.7

  # 运行
  python -m data_juicer.core.executor --config config.yaml
```

---

### 6.3 datatrove

```
项目地址：https://github.com/huggingface/datatrove

描述：
  HuggingFace 开发的大规模数据处理库。
  FineWeb 数据集就是用 datatrove 处理的。
  设计目标：可扩展、高效、易用。

核心组件：
  ├── Readers：数据读取（WARC, JSON, Parquet 等）
  ├── Extractors：文本提取（trafilatura, resiliparse）
  ├── Filters：数据过滤（语言、质量、C4 规则等）
  ├── Dedup：去重（sentence-level, MinHash, exact）
  ├── Stats：统计分析
  └── Writers：数据写入

使用示例：
  from datatrove.pipeline.readers import WARCReader
  from datatrove.pipeline.filters import (
      URLFilter,
      LanguageFilter,
      GopherQualityFilter,
  )
  from datatrove.pipeline.dedup import MinhashDedupSignature
  from datatrove.pipeline.writers import JsonlWriter
  from datatrove.executor import LocalPipelineExecutor

  pipeline = [
      WARCReader("s3://commoncrawl/..."),
      URLFilter(),                      # URL 黑名单过滤
      LanguageFilter(language="en"),     # 语言过滤
      GopherQualityFilter(),            # 质量过滤（DeepMind 的规则）
      MinhashDedupSignature(),          # MinHash 去重
      JsonlWriter("output/clean/"),     # 输出
  ]

  executor = LocalPipelineExecutor(
      pipeline=pipeline,
      tasks=100,
      workers=8,
  )
  executor.run()
```

---

### 6.4 数据清洗最佳实践

```
数据清洗的标准流程：

┌──────────────────────────────────────────────────┐
│                数据清洗流水线                       │
├──────────────────────────────────────────────────┤
│                                                   │
│  1. 文本提取（从 HTML/PDF 等格式中提取纯文本）      │
│     ├── 去除 HTML 标签、导航栏、页脚等              │
│     ├── 工具：trafilatura, readability             │
│     └── 保留正文内容                               │
│                                                   │
│  2. 语言识别                                       │
│     ├── 识别文本语言                               │
│     ├── 过滤非目标语言                              │
│     ├── 工具：fasttext, langdetect                 │
│     └── 建议阈值：confidence > 0.8                 │
│                                                   │
│  3. 基础过滤                                       │
│     ├── 长度过滤：去除过短（< 100 字符）和过长的文本  │
│     ├── 字符过滤：去除乱码、过多特殊字符的文本       │
│     ├── 关键词过滤：去除广告、色情、垃圾等内容       │
│     └── 个人信息脱敏：去除邮箱、电话等              │
│                                                   │
│  4. 质量评估                                       │
│     ├── Perplexity 过滤：用小模型计算困惑度          │
│     │   困惑度极高或极低的都可能是低质量内容         │
│     ├── 教育价值评估：训练分类器评分                 │
│     ├── 重复率检查：过多重复词/句的文本              │
│     └── 连贯性检查                                 │
│                                                   │
│  5. 去重                                           │
│     ├── 精确去重：基于 hash 的完全相同文本           │
│     ├── 近似去重：MinHash + LSH 找相似文本          │
│     │   ├── 设置 Jaccard 相似度阈值（通常 0.7-0.8）│
│     │   └── 可在文档级或段落级去重                  │
│     └── URL 去重：同一 URL 只保留一份               │
│                                                   │
│  6. 安全过滤                                       │
│     ├── 毒性检测：去除有害内容                      │
│     ├── PII（个人可识别信息）检测和脱敏              │
│     └── 版权敏感内容过滤                            │
│                                                   │
│  7. 数据混合与采样                                  │
│     ├── 按领域/来源设定配比                         │
│     ├── 上采样高质量数据（如 Wikipedia 可重复采样）   │
│     └── 下采样过多的领域                            │
│                                                   │
└──────────────────────────────────────────────────┘

常见陷阱：
├── 过度清洗：过于严格的过滤会损失多样性
├── 去重不足：重复数据会导致模型记忆而非泛化
├── 忽视数据配比：不同来源数据的比例影响很大
└── 不做抽样检查：清洗后必须人工抽查样本质量
```

---

## 附录：药材选用指南

### 按修炼阶段推荐

```
阶段 1：入门学习（斗之气 → 斗者）
  ├── 不需要自己准备数据
  ├── 使用 HuggingFace 上的现成数据集
  └── 推荐：Alpaca, Dolly 2.0（体量小，易上手）

阶段 2：SFT 微调（斗师 → 斗灵）
  ├── 中文微调：BELLE + Firefly
  ├── 英文微调：OpenHermes 2.5 或 ShareGPT
  ├── 高质量少量：LIMA（1000 条验证方法论）
  └── 领域微调：准备自己的领域数据

阶段 3：偏好对齐（斗王 → 斗皇）
  ├── DPO 训练：UltraFeedback
  ├── RLHF 训练：HH-RLHF
  └── 奖励模型：Nectar, Chatbot Arena 数据

阶段 4：预训练（斗宗 → 斗帝）
  ├── 英文：FineWeb-Edu + SlimPajama + The Stack
  ├── 中文：MNBVC + CulturaX
  ├── 代码：The Stack v2
  └── 多语言：CulturaX

阶段 5：评估（所有阶段）
  ├── 英文综合：MMLU, HellaSwag, TruthfulQA
  ├── 推理：GSM8K, MATH, ARC
  ├── 代码：HumanEval
  └── 中文：C-Eval, CMMLU
```

### 数据配比经验法则

```
预训练数据配比建议（参考 LLaMA）：

通用网页数据    : 65-75%    ← 基础语言能力
代码           : 5-10%     ← 代码能力和逻辑推理
学术论文       : 2-5%      ← 科学知识
书籍           : 2-5%      ← 长文本理解
百科           : 2-3%      ← 事实性知识
技术问答       : 2-3%      ← 技术细节
数学           : 1-3%      ← 数学能力

SFT 微调数据配比建议：

通用对话       : 30-40%
代码相关       : 15-20%
数学和推理     : 10-15%
创意写作       : 10-15%
知识问答       : 10-15%
安全拒绝       : 5-10%

关键原则：
├── 质量 > 数量：1 条高质量数据胜过 100 条低质量数据
├── 多样性很重要：单一来源的数据会限制模型能力
├── 配比需要实验：没有万能的配比，需要根据目标调整
└── 持续迭代：根据评估结果调整数据配比
```

---

> *"天下药材，尽在此库。*
> *但请记住：最好的药材不是最贵的、最多的，*
> *而是最适合你当前境界和目标的。*
> *一位斗之气的新手，不需要天材地宝；*
> *一位斗宗的强者，也不该只用凡品药材。*
>
> *根据自己的修炼目标，精心选择药材，*
> *这才是炼丹成功的第一步。"*
>
> — 《焚诀》附录·药材库 完

---

## 附录 B：2024-2025 新型药材

> *"新的修炼需求催生了新的药材。"*

### B.1 指令微调数据集（SFT）

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| Open-Orca | 420万条 | 英文 | GPT-4 合成高质量指令 |
| Firefly | 160万条 | 中文 | 中文指令微调综合数据 |
| COIG-CQIA | 50万条 | 中文 | 中文问答+指令+对话 |
| UltraFeedback | 64万条 | 英文 | 带评分高质量偏好数据 |

### B.2 对齐数据集（RLHF/DPO）

| 数据集 | 规模 | 用途 |
|--------|------|------|
| HH-RLHF | 170K 对话 | Anthropic 人类偏好 |
| UltraFeedback-binarized | 64K | DPO 标准数据 |
| Nectar | 370K | 多奖励模型标注 |
| RLHF-V | 10K | 视觉-语言偏好 |

数据质量铁律：1万条高质量 > 100万条低质量；多样性比数量更重要；GPT-4 标注效果最佳。

### B.3 合成数据（Synthetic Data）

> *"当天然药材不足时，炼丹师学会了人工合成。合成数据已成为现代修炼不可或缺的药源。"*

| 数据集/工具 | 规模 | 用途 | 说明 |
|------------|------|------|------|
| **Evol-Instruct** | 52K→250K | 指令进化 | 从种子指令通过 LLM 迭代进化生成更复杂的指令 |
| **WizardLM** | 70K | 复杂指令 | 基于 Evol-Instruct 生成的复杂指令数据 |
| **Magpie** | 100万+ | 多模型蒸馏 | 利用模型对齐模板自动提取指令数据 |
| **Self-Instruct** | 52K | 自我生成 | GPT-4 用自己的输出创建新指令（"以毒攻毒"） |
| **DataDreamer** | 工具 | 数据生成框架 | 灵活的合成数据生成流水线 |

```python
# === 使用 Magpie 提取指令数据（零人工标注）===
# Magpie 原理：利用模型的对齐模板（如 <｜assistant｜>），
# 让模型自动补全生成用户指令。

# 示例：从 Qwen2.5-7B 提取指令
# pip install magpie

# Magpie 使用方法（伪代码）：
magpie_config = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "template": "qwen2",
    "num_samples": 10000,
    "temperature": 1.0,       # 高温度增加多样性
    "max_tokens": 1024,
}
# magpie.extract_instructions(magpie_config) → 生成大量指令-回答对

print("合成数据使用原则：")
print("  1. 合成数据不能完全替代人工数据")
print("  2. 多样性 > 数量（不同来源、不同难度）")
print("  3. 用强模型（如 GPT-4）生成、弱模型微调 → 蒸馏效果")
print("  4. 务必做去重和质量过滤（MinHash / 分类器过滤）")
```

### B.4 DeepSeek-R1 蒸馏数据

> *"斗帝强者的修炼心得，蒸馏为丹方，传承后世。"*

DeepSeek-R1 开源后，其蒸馏数据集成为训练推理能力模型的宝贵资源。

| 数据集 | 来源 | 规模 | 特点 |
|--------|------|------|------|
| **R1-Distill-SFT** | DeepSeek-R1 蒸馏 | ~800K | 带 `<think推理过程>` 的完整推理链 |
| **Open-R1** | 开源社区 | 100K+ | 多模型蒸馏的推理数据 |
| **NuminaMath** | Numina | 860K | 数学竞赛推理数据 |
| **OpenMathInstruct** | 开源 | 200万 | 数学推理指令数据 |

```python
# === R1 蒸馏数据格式示例 ===
r1_data_format = """
{
  "instruction": "证明：对于任意正整数 n，1+2+3+...+n = n(n+1)/2",
  "output": "<think推理过程>\\n
  我们用数学归纳法证明：\\n
  基础步：n=1 时，左边=1，右边=1×2/2=1，成立。\\n
  归纳步：假设 n=k 时成立，即 1+...+k = k(k+1)/2\\n
  则 n=k+1 时：1+...+k+(k+1) = k(k+1)/2 + (k+1)\\n
  = (k+1)(k/2+1) = (k+1)(k+2)/2，成立。\\n
  因此对所有正整数 n 成立。<／think>\\n
  证明完成。"
}

关键要素：
  1. <think推理过程> 标签包裹推理链
  2. 推理链包含完整的思考步骤
  3. 最终结论在标签外
  4. 训练时需要保持标签格式
"""

print("R1 蒸馏数据使用注意事项：")
print("  1. 训练时不要截断推理链（保持完整性）")
print("  2. 学习率建议 1e-5（比普通 SFT 更低）")
print("  3. 混合普通对话数据（比例约 3:1），防止只学会推理格式")
print("  4. 评估时需要同时测试推理能力和对话能力")
```

### B.5 多模态数据集补充

| 数据集 | 规模 | 模态 | 特点 |
|--------|------|------|------|
| **ShareGPT4V** | 100万+ | 图文对话 | GPT-4V 生成的高质量多模态对话 |
| **LLaVA-OneVision** | 100万+ | 图文+视频 | 统一的视觉-语言指令数据 |
| **VideoInstruct-100K** | 10万 | 视频-文本 | 视频理解和问答数据 |
| **DocVQA** | 5万 | 文档图像 | 文档视觉问答基准 |
| **ChartQA** | 1.8万 | 图表理解 | 图表数据提取和推理 |
| **AI2D** | 1.5万 | 科学图表 | 科学图表理解（K-12 级别） |

### B.6 评估基准更新（2025-2026）

| 基准 | 评估维度 | 新增 |
|------|---------|------|
| **ArenaHard** | 对话质量 | 500道专家标注难题 |
| **LiveBench** | 综合能力 | 每月更新、防数据泄露 |
| **LongBench v2** | 长文本 | 10K-100K 长度覆盖 |
| **CRUXEval** | 代码推理 | 纯推理（不含执行） |
| **CVBench** | 视觉推理 | 多模态模型专用 |
| **GoldBench** | 黄金标准 | 零数据泄露评估集 |

---

## 本卷增强补全（2026）— 数据治理：许可/脱敏/版本化（让“药材”可长期使用）

> 本节为《焚诀》深度研究版的**回填内容**，用于补齐数据集管理的“长期主义”：可追溯、可审计、可复现。  
> 完整方法论总览见：[焚诀-深度研究版-全卷增强补全（2026）](../焚诀-深度研究版-全卷增强补全.md)

### 1) 数据治理三件套（强烈建议作为团队默认规范）

- **许可与来源**：来源 URL、抓取时间、使用条款、是否可商用、是否可再分发  
- **隐私与脱敏**：PII 检测（手机号/地址/证件号/邮箱）→脱敏→抽样审计  
- **版本化与审计**：数据版本号、样本哈希、抽样清单、过滤规则版本（可回溯）

### 2) 数据质量的“高收益过滤”建议（先做这几项往往立竿见影）

- 去重：大规模语料必须做（否则训练效率与泛化双杀）  
- 低质量过滤：乱码/广告/模板化重复内容  
- 有毒内容过滤：暴力/歧视/色情/诈骗  
- 评测隔离：评测题/公开基准题必须隔离，避免污染

