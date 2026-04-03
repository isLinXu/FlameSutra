"""
╔══════════════════════════════════════════════════════════════════╗
║          焚 诀 · 第四卷 · 化形篇（斗灵）                         ║
║          LoRA 微调实战 —— 斗气凝形，化虚为实                      ║
╚══════════════════════════════════════════════════════════════════╝

修炼心法：
    LoRA（Low-Rank Adaptation）乃是一门以小搏大的上乘功法。
    修炼者无需重铸整尊斗气（全参数微调），只需在原有斗气经脉
    之上，嫁接几条低秩旁支经脉（低秩矩阵），即可以极小的代价
    令上古斗气（预训练模型）适应新的战斗场景。

    此法妙处：
    - 仅需修改原有斗气的 0.1%~1%，显存消耗骤降
    - 炼丹（训练）速度提升数倍
    - 可随时取下旁支经脉，还原上古本体
    - 多套 LoRA 可如更换斗技般自由切换

所需灵材：
    pip install torch transformers peft trl datasets accelerate bitsandbytes

运行丹方：
    python 04_LoRA微调实战.py

注意：此丹方默认使用 Qwen2.5-1.5B-Instruct 作为基座。
      若你的丹炉（服务器）异火（GPU）充裕，可替换为更大的模型。
"""

import os
import json
import torch
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# 第一节：灵材准备 —— 导入所需功法与药材
# ═══════════════════════════════════════════════════════════════

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ═══════════════════════════════════════════════════════════════
# 第二节：丹方配置 —— 设定炼丹参数
# ═══════════════════════════════════════════════════════════════

# ------ 基座模型选择 ------
# 此处选用 Qwen2.5-1.5B-Instruct 作为上古斗气基座
# 若欲更换基座，只需修改此处路径即可：
#   - "Qwen/Qwen2.5-7B-Instruct"    # 七十亿参数，需 ~16GB 显存
#   - "meta-llama/Llama-3.1-8B"      # Meta 的八十亿参数斗气
#   - "mistralai/Mistral-7B-v0.3"    # Mistral 的七十亿参数斗气
#   - 也可使用本地路径: "/path/to/your/local/model"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ------ 输出路径 ------
OUTPUT_DIR = "./lora_output"              # 炼丹产物存放之处
MERGED_DIR = "./lora_merged_model"        # 合并后的完整斗气

# ------ LoRA 经脉参数 ------
LORA_RANK = 64             # 旁支经脉的宽度（秩），越大表达力越强，但消耗越多
LORA_ALPHA = 128           # 缩放系数，通常设为 rank 的 1~2 倍
LORA_DROPOUT = 0.05        # 经脉随机断流概率，防止过拟合（走火入魔）
TARGET_MODULES = [          # 要嫁接旁支经脉的斗气层
    "q_proj",               # 注意力机制 —— 查询之眼
    "k_proj",               # 注意力机制 —— 关键之眼
    "v_proj",               # 注意力机制 —— 价值之眼
    "o_proj",               # 注意力机制 —— 输出之眼
    "gate_proj",            # FFN 门控层
    "up_proj",              # FFN 上升层
    "down_proj",            # FFN 下降层
]

# ------ 炼丹参数 ------
NUM_EPOCHS = 3              # 炼丹轮次，药材反复熬煮的次数
BATCH_SIZE = 2              # 每炉投入的药材数量
GRADIENT_ACCUM_STEPS = 4    # 梯度聚灵步数（等效 batch = 2 × 4 = 8）
LEARNING_RATE = 2e-4        # 火候大小（学习率）
MAX_SEQ_LENGTH = 512        # 药材最大切割长度
WARMUP_RATIO = 0.03         # 缓慢升温比例，防止开炉时火候过猛
USE_4BIT = False            # 是否启用四比特量化（异火不足时开启）


# ═══════════════════════════════════════════════════════════════
# 第三节：药材炮制 —— 构建指令微调数据集
# ═══════════════════════════════════════════════════════════════

def prepare_dataset() -> Dataset:
    """
    炮制修炼药材 —— 构建指令格式训练数据集

    药材格式说明：
        每份药材包含三味药引：
        - instruction: 修炼指令（用户的问题/任务）
        - input:       辅助药引（可选的上下文信息）
        - output:      期望产出（模型应生成的回答）

    实战中，你应替换为自己的药材库（数据集）：
        - 从 HuggingFace Hub 加载: datasets.load_dataset("your_dataset")
        - 从本地 JSON/CSV 加载: datasets.load_dataset("json", data_files="data.json")
        - 从数据库或 API 加载后构建 Dataset 对象
    """
    # 示例药材 —— 以修仙问答为主题的指令数据
    raw_data = [
        {
            "instruction": "请解释什么是 LoRA 微调技术。",
            "input": "",
            "output": (
                "LoRA（Low-Rank Adaptation）是一种参数高效微调技术。"
                "它通过在预训练模型的权重矩阵旁添加低秩分解矩阵来实现微调，"
                "而不修改原始模型参数。具体来说，对于权重矩阵 W，LoRA 添加 "
                "ΔW = BA，其中 B 和 A 是低秩矩阵。这样只需训练极少量参数"
                "（通常不到原模型的 1%），就能达到接近全参数微调的效果。"
                "优势包括：显存占用大幅降低、训练速度快、可保留原模型能力、"
                "多个 LoRA 适配器可灵活切换。"
            ),
        },
        {
            "instruction": "什么是梯度下降？请用简单的比喻来解释。",
            "input": "",
            "output": (
                "想象你被蒙上眼睛站在山顶，需要走到山谷最低点。"
                "你无法看到全貌，但可以感受脚下的坡度。梯度下降就是这个过程："
                "每一步，你都感受地面的倾斜方向（计算梯度），然后朝着下坡方向"
                "迈一步（更新参数）。步子的大小就是学习率——太大可能越过山谷，"
                "太小则需要很久才能到达。反复迈步，最终你会到达一个低谷（局部最优），"
                "虽然不一定是最低点（全局最优），但通常已经足够好了。"
            ),
        },
        {
            "instruction": "请对以下文本进行摘要。",
            "input": (
                "Transformer 架构于 2017 年在论文《Attention Is All You Need》中提出，"
                "彻底改变了自然语言处理领域。它引入了自注意力机制，使模型能够并行处理"
                "序列中的所有位置，突破了 RNN 的顺序处理限制。此后，基于 Transformer "
                "的模型如 BERT、GPT 系列不断涌现，推动了预训练-微调范式的普及，"
                "最终催生了 ChatGPT 等强大的大语言模型。"
            ),
            "output": (
                "Transformer 架构（2017）通过自注意力机制实现了序列并行处理，"
                "突破了 RNN 的限制，催生了 BERT、GPT 等模型，"
                "推动了大语言模型的发展。"
            ),
        },
        {
            "instruction": "将以下 Python 代码改写为更 Pythonic 的风格。",
            "input": (
                "result = []\n"
                "for i in range(len(numbers)):\n"
                "    if numbers[i] > 0:\n"
                "        result.append(numbers[i] * 2)"
            ),
            "output": (
                "result = [x * 2 for x in numbers if x > 0]\n\n"
                "使用列表推导式替代了显式的 for 循环和 append 操作，"
                "代码更简洁、更具 Python 风格，且通常执行效率更高。"
            ),
        },
        {
            "instruction": "请解释 GPU 显存不足（OOM）的常见原因和解决方案。",
            "input": "",
            "output": (
                "GPU 显存不足（Out of Memory）的常见原因：\n"
                "1. 批次大小过大：每个样本都需要存储激活值和梯度\n"
                "2. 模型参数过多：大模型本身就占用大量显存\n"
                "3. 序列长度过长：注意力机制的显存消耗与序列长度平方成正比\n"
                "4. 优化器状态：Adam 需要为每个参数维护两个额外状态\n\n"
                "解决方案：\n"
                "1. 减小批次大小，配合梯度累加保持等效批次\n"
                "2. 使用混合精度训练（FP16/BF16）\n"
                "3. 启用梯度检查点（gradient checkpointing）\n"
                "4. 使用 LoRA 等参数高效微调方法\n"
                "5. 使用 DeepSpeed ZeRO 分布式策略\n"
                "6. 4-bit/8-bit 量化加载模型"
            ),
        },
        {
            "instruction": "请给出一个使用 PyTorch 实现简单线性回归的代码示例。",
            "input": "",
            "output": (
                "```python\n"
                "import torch\n"
                "import torch.nn as nn\n\n"
                "# 生成数据: y = 2x + 1 + noise\n"
                "x = torch.randn(100, 1)\n"
                "y = 2 * x + 1 + 0.1 * torch.randn(100, 1)\n\n"
                "# 定义模型\n"
                "model = nn.Linear(1, 1)\n"
                "criterion = nn.MSELoss()\n"
                "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n\n"
                "# 训练\n"
                "for epoch in range(100):\n"
                "    pred = model(x)\n"
                "    loss = criterion(pred, y)\n"
                "    optimizer.zero_grad()\n"
                "    loss.backward()\n"
                "    optimizer.step()\n\n"
                "print(f'w={model.weight.item():.2f}, b={model.bias.item():.2f}')\n"
                "```\n"
                "训练后权重应接近 2，偏置接近 1。"
            ),
        },
        {
            "instruction": "比较 Adam 和 SGD 优化器的优缺点。",
            "input": "",
            "output": (
                "SGD（随机梯度下降）：\n"
                "优点：实现简单、泛化性能好、显存占用少\n"
                "缺点：收敛慢、对学习率敏感、需要精心调参\n\n"
                "Adam：\n"
                "优点：自适应学习率、收敛快、对超参数不敏感、适合稀疏梯度\n"
                "缺点：显存多占 2 倍（维护一阶和二阶矩估计）、可能泛化不如 SGD\n\n"
                "实践建议：大语言模型预训练常用 AdamW（带权重衰减的 Adam）；"
                "计算机视觉任务中 SGD+Momentum 仍然常用；"
                "微调阶段 Adam 系列通常更稳定。"
            ),
        },
        {
            "instruction": "什么是注意力机制中的 KV Cache？为什么它对推理速度很重要？",
            "input": "",
            "output": (
                "KV Cache 是在自回归推理时缓存已计算的 Key 和 Value 向量的技术。\n\n"
                "原理：在生成每个新 token 时，模型需要对所有已生成的 token 计算注意力。"
                "如果不缓存，每生成一个 token 都要重新计算所有历史 token 的 K 和 V，"
                "计算量随序列长度线性增长。\n\n"
                "有了 KV Cache，之前 token 的 K、V 值只需计算一次并存储，"
                "后续只需计算新 token 的 K、V 并拼接，避免重复计算。\n\n"
                "代价：KV Cache 会占用额外显存，长序列时可能很大。"
                "这也是为什么长上下文模型（如 128K tokens）需要"
                " GQA（分组查询注意力）等技术来压缩 KV Cache 的原因。"
            ),
        },
    ]

    # 将药材转化为 ChatML 对话格式 —— 这是 Qwen 系列模型的标准输入格式
    formatted_data = []
    for item in raw_data:
        # 构建用户消息
        user_msg = item["instruction"]
        if item["input"]:
            user_msg += f"\n\n{item['input']}"

        # 构建 ChatML 格式的对话
        conversation = {
            "text": (
                f"<|im_start|>system\n"
                f"你是一个专业的 AI 助手，擅长解释技术概念。<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{item['output']}<|im_end|>"
            )
        }
        formatted_data.append(conversation)

    dataset = Dataset.from_list(formatted_data)
    print(f"[药材炮制完成] 共计 {len(dataset)} 份药材已就绪")
    print(f"[药材样例] {formatted_data[0]['text'][:120]}...")
    return dataset


# ═══════════════════════════════════════════════════════════════
# 第四节：量化辅阵 —— 可选的 4-bit 量化配置
# ═══════════════════════════════════════════════════════════════

def get_quantization_config() -> Optional[BitsAndBytesConfig]:
    """
    四比特量化阵法 —— 以精度换显存

    当你的丹炉异火不足（GPU 显存紧张）时，可启用此阵法。
    QLoRA = 4-bit 量化 + LoRA，可在单张 24GB 显卡上微调 7B 模型。

    注意：量化需要安装 bitsandbytes 库
          pip install bitsandbytes
    """
    if not USE_4BIT:
        return None

    print("[启动量化阵法] 四比特压缩，以精度换显存空间...")
    return BitsAndBytesConfig(
        load_in_4bit=True,                       # 以 4-bit 精度加载模型
        bnb_4bit_quant_type="nf4",               # NF4 量化类型（Normal Float 4）
        bnb_4bit_compute_dtype=torch.bfloat16,   # 计算时使用 BF16 精度
        bnb_4bit_use_double_quant=True,           # 双重量化，进一步压缩
    )


# ═══════════════════════════════════════════════════════════════
# 第五节：召唤上古斗气 —— 加载预训练模型
# ═══════════════════════════════════════════════════════════════

def load_base_model():
    """
    召唤上古斗气（加载预训练模型）

    这一步将从 HuggingFace Hub 或本地路径加载预训练模型。
    上古斗气蕴含着从海量药材（数据）中提炼的万般知识。
    """
    print("=" * 60)
    print("  [召唤上古斗气] 正在加载基座模型...")
    print(f"  模型: {MODEL_NAME}")
    print("=" * 60)

    # 获取量化配置（若启用）
    quant_config = get_quantization_config()

    # 加载分词器 —— 药材切割之法
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",        # 右侧填充，适配因果语言模型
    )

    # 确保 pad_token 存在（部分模型未设定）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[药材切割] pad_token 未设定，已自动设为 eos_token")

    # 加载模型 —— 召唤上古斗气本体
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",                        # 自动分配至可用异火（GPU）
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
        trust_remote_code=True,
        attn_implementation="eager",              # 兼容性优先；可改为 "flash_attention_2"
    )

    # 若使用量化，需额外准备模型以适配 LoRA
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
        print("[量化适配] 模型已为 k-bit 训练做好准备")

    # 统计上古斗气的规模
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[斗气规模] 总参数量: {total_params:,}")
    print(f"[斗气规模] 可训练参数量: {trainable_params:,}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# 第六节：嫁接旁支经脉 —— 配置并应用 LoRA
# ═══════════════════════════════════════════════════════════════

def apply_lora(model):
    """
    嫁接旁支经脉 —— 将 LoRA 适配器注入基座模型

    LoRA 的核心思想：
        原始权重 W (d × d) 保持冻结
        新增 ΔW = B × A，其中 B (d × r), A (r × d)，r << d
        前向传播: h = Wx + BAx = Wx + ΔWx

    r 就是我们设定的 LORA_RANK，它决定了旁支经脉的容量。
    """
    print("\n" + "=" * 60)
    print("  [嫁接旁支经脉] 正在注入 LoRA 适配器...")
    print(f"  秩(rank): {LORA_RANK}")
    print(f"  缩放(alpha): {LORA_ALPHA}")
    print(f"  目标层: {TARGET_MODULES}")
    print("=" * 60)

    # 构建 LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,       # 任务类型：因果语言模型
        r=LORA_RANK,                         # 低秩矩阵的秩
        lora_alpha=LORA_ALPHA,               # LoRA 缩放因子
        lora_dropout=LORA_DROPOUT,           # Dropout 概率
        target_modules=TARGET_MODULES,       # 注入 LoRA 的目标模块
        bias="none",                         # 不训练偏置项
    )

    # 将 LoRA 注入模型 —— 经脉嫁接
    model = get_peft_model(model, lora_config)

    # 打印嫁接后的参数统计 —— 验证经脉是否成功嫁接
    model.print_trainable_parameters()
    # 输出类似：trainable params: 13,631,488 || all params: 1,556,480,000 || trainable%: 0.876%
    # 只需修炼不到 1% 的参数！

    return model


# ═══════════════════════════════════════════════════════════════
# 第七节：开炉炼丹 —— 训练过程
# ═══════════════════════════════════════════════════════════════

def train(model, tokenizer, dataset):
    """
    开炉炼丹 —— 使用 SFTTrainer 进行监督微调

    SFTTrainer 是 trl 库提供的高级训练器，专为监督微调设计。
    它简化了数据格式化、tokenization、训练循环等繁琐步骤，
    让炼丹师能专注于药材与火候的调配。
    """
    print("\n" + "=" * 60)
    print("  [丹炉已热] 开始炼丹！")
    print(f"  炼丹轮次: {NUM_EPOCHS}")
    print(f"  每炉药材: {BATCH_SIZE}")
    print(f"  梯度聚灵: {GRADIENT_ACCUM_STEPS} 步")
    print(f"  等效批次: {BATCH_SIZE * GRADIENT_ACCUM_STEPS}")
    print(f"  火候(lr): {LEARNING_RATE}")
    print("=" * 60)

    # 构建训练参数 —— 丹方中的火候配置
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",                # 余弦退火策略，火候渐减
        warmup_ratio=WARMUP_RATIO,                 # 缓慢升温
        weight_decay=0.01,                         # 权重衰减，防止斗气过于膨胀
        logging_steps=1,                           # 每步记录丹炉内况
        save_strategy="epoch",                     # 每轮保存一次丹药
        save_total_limit=2,                        # 最多保留 2 份丹药
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,               # 梯度检查点 —— 以时间换显存
        optim="adamw_torch",                       # AdamW 优化器
        max_seq_length=MAX_SEQ_LENGTH,             # 药材最大长度
        dataset_text_field="text",                 # 数据集中文本字段名
        report_to="none",                          # 不上报至 wandb 等平台
        seed=42,                                   # 随机种子，保证炼丹可复现
    )

    # 若启用梯度检查点，需要关闭 use_cache（二者不兼容）
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        # 启用 gradient checkpointing 的输入梯度保留
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # 初始化 SFTTrainer —— 炼丹炉点火
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 开始炼丹！
    print("\n[丹炉点火] 炼丹正式开始，请静待丹成...")
    print("[提示] 若出现 CUDA OOM（反噬），请减小 BATCH_SIZE 或启用 4-bit 量化\n")

    train_result = trainer.train()

    # 炼丹完成，输出统计
    print("\n" + "=" * 60)
    print("  [丹成！] 炼丹过程顺利完成")
    print(f"  总训练步数: {train_result.metrics.get('train_steps', 'N/A')}")
    print(f"  最终 Loss:  {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print("=" * 60)

    # 保存 LoRA 适配器权重 —— 收丹入瓶
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[收丹入瓶] LoRA 适配器已保存至: {OUTPUT_DIR}")

    return trainer


# ═══════════════════════════════════════════════════════════════
# 第八节：经脉合一 —— 合并 LoRA 权重到基座模型
# ═══════════════════════════════════════════════════════════════

def merge_lora_weights():
    """
    经脉合一 —— 将旁支经脉（LoRA）融合回主经脉（基座模型）

    合并后的模型可以独立运行，无需再额外加载 LoRA 适配器。
    这就像将临时嫁接的经脉永久融合，成为自身斗气的一部分。

    合并公式: W_merged = W_original + (alpha/rank) * B @ A
    """
    print("\n" + "=" * 60)
    print("  [经脉合一] 正在将 LoRA 融合回基座模型...")
    print("=" * 60)

    # 重新加载基座模型（以完整精度）
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    print("[经脉加载] LoRA 适配器已加载")

    # 合并权重 —— 旁支经脉融入主脉
    model = model.merge_and_unload()
    print("[融合完成] LoRA 权重已融入基座模型")

    # 保存合并后的完整模型
    model.save_pretrained(MERGED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[归档] 完整模型已保存至: {MERGED_DIR}")

    return model


# ═══════════════════════════════════════════════════════════════
# 第九节：验证丹效 —— 推理测试
# ═══════════════════════════════════════════════════════════════

def test_inference(model, tokenizer, label: str = "微调后"):
    """
    验证丹效 —— 测试模型生成效果

    此步骤对比微调前后的生成质量，验证 LoRA 是否生效。
    若丹成，模型应在指令遵循和回答质量上有明显提升。
    """
    print(f"\n{'=' * 60}")
    print(f"  [验丹] {label}推理测试")
    print(f"{'=' * 60}")

    # 测试用例
    test_prompts = [
        "请用一句话解释什么是反向传播。",
        "解释 Transformer 中多头注意力的作用。",
        "如何选择合适的学习率？",
    ]

    model.eval()
    device = next(model.parameters()).device

    for i, prompt in enumerate(test_prompts):
        # 构建 ChatML 格式输入
        messages = [
            {"role": "system", "content": "你是一个专业的 AI 助手，擅长解释技术概念。"},
            {"role": "user", "content": prompt},
        ]

        # 使用 tokenizer 的 chat template 格式化
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 回退：手动构建 ChatML 格式
            text = (
                f"<|im_start|>system\n"
                f"你是一个专业的 AI 助手，擅长解释技术概念。<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,          # 最多生成 200 个 token
                temperature=0.7,             # 温度参数，控制随机性
                top_p=0.9,                   # 核采样
                do_sample=True,              # 启用采样
                repetition_penalty=1.1,      # 防止重复
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码生成结果（只取新生成的部分）
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n[问题 {i+1}] {prompt}")
        print(f"[回答] {response.strip()}")
        print("-" * 40)


# ═══════════════════════════════════════════════════════════════
# 第十节：主控丹方 —— 完整炼丹流程
# ═══════════════════════════════════════════════════════════════

def main():
    """
    主控丹方 —— 一键炼丹

    完整流程：
    1. 炮制药材（准备数据集）
    2. 召唤上古斗气（加载预训练模型）
    3. 嫁接旁支经脉（应用 LoRA）
    4. 开炉炼丹（训练）
    5. 经脉合一（合并 LoRA 权重）
    6. 验证丹效（推理测试）
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   焚 诀 · LoRA 微调实战                                   ║
    ║   Flame Sutra · LoRA Fine-tuning Practice                ║
    ║                                                          ║
    ║   以旁支经脉之法，四两拨千斤                                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # 检查丹炉状态（GPU 可用性）
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"[丹炉检测] 异火就绪: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[丹炉检测] 未检测到异火（GPU），将使用凡火（CPU）运行")
        print("[警告] CPU 炼丹极为缓慢，建议在配备 GPU 的环境下运行")

    # ---- 步骤 1: 炮制药材 ----
    print("\n" + "▶" * 20 + " 步骤 1/6: 炮制药材 " + "◀" * 20)
    dataset = prepare_dataset()

    # ---- 步骤 2: 召唤上古斗气 ----
    print("\n" + "▶" * 20 + " 步骤 2/6: 召唤上古斗气 " + "◀" * 20)
    model, tokenizer = load_base_model()

    # ---- 步骤 3: 微调前验丹（可选，了解基座模型的原始能力） ----
    print("\n" + "▶" * 20 + " 步骤 3/6: 微调前验丹 " + "◀" * 20)
    test_inference(model, tokenizer, label="微调前")

    # ---- 步骤 4: 嫁接旁支经脉 ----
    print("\n" + "▶" * 20 + " 步骤 4/6: 嫁接旁支经脉 " + "◀" * 20)
    model = apply_lora(model)

    # ---- 步骤 5: 开炉炼丹 ----
    print("\n" + "▶" * 20 + " 步骤 5/6: 开炉炼丹 " + "◀" * 20)
    trainer = train(model, tokenizer, dataset)

    # ---- 步骤 6: 经脉合一 + 验丹 ----
    print("\n" + "▶" * 20 + " 步骤 6/6: 经脉合一 " + "◀" * 20)
    merged_model = merge_lora_weights()
    merged_tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    test_inference(merged_model, merged_tokenizer, label="微调后")

    # ---- 完成 ----
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   炼丹完毕！                                              ║
    ║                                                          ║
    ║   LoRA 适配器: {:<40s} ║
    ║   合并模型:     {:<40s} ║
    ║                                                          ║
    ║   下一步修炼方向：                                         ║
    ║   - 收集更多高质量药材（数据集）                             ║
    ║   - 调整经脉参数（rank, alpha, target_modules）             ║
    ║   - 尝试 QLoRA（4-bit 量化 + LoRA）                       ║
    ║   - 进阶：第五卷·化翼篇 —— 多卡分布式炼丹                  ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(OUTPUT_DIR, MERGED_DIR))


if __name__ == "__main__":
    main()
