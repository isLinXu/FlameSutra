"""
===============================================================================
  焚诀 · 第七卷 · 空间篇（斗宗）
  实战丹方：多模态模型实战

  "空间裂缝已开，维度壁垒已破。此丹方记录了如何驾驭空间之力——
   从感知虚空影像，到撕裂维度壁垒，再到真正的空间融合。"

  本丹方包含：
    1. CLIP 图文匹配实战 —— 空间裂缝初探
    2. 视觉特征提取与分析 —— 空间感知修炼
    3. 简易多模态管线构建 —— 空间桥梁搭建
    4. 使用预训练 VLM 进行视觉问答 —— 空间法器驾驭

  丹炉要求：
    - Python 3.10+
    - PyTorch 2.0+
    - transformers >= 4.40
    - 推荐异火：NVIDIA GPU (至少 8GB 显存)
    - 无异火时可使用 CPU（部分实验会较慢）

  药材准备（依赖安装）：
    pip install torch torchvision transformers
    pip install Pillow requests matplotlib numpy
    pip install open_clip_torch  # OpenCLIP 开源实现
===============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 第零节：丹炉检测 —— 确认修炼环境
# ============================================================================

def detect_furnace():
    """
    丹炉检测：确认修炼环境中的异火（GPU）状态

    一位谨慎的炼丹师，在开炉前必先检查丹炉的火力是否充足。
    异火不足时，也可用凡火（CPU）替代，只是炼丹速度会慢许多。
    """
    import torch

    print("=" * 65)
    print("  焚诀 · 第七卷 · 空间篇 · 丹炉检测")
    print("=" * 65)

    # 检测 PyTorch 版本
    print(f"\n  PyTorch 版本: {torch.__version__}")

    # 检测异火（GPU）
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  异火检测: {gpu_name}")
        print(f"  火力强度: {gpu_mem:.1f} GB 显存")
        device = "cuda"

        if gpu_mem >= 24:
            print("  丹炉等级: 兽级丹炉 (可运行完整 VLM 实验)")
        elif gpu_mem >= 12:
            print("  丹炉等级: 灵级丹炉 (可运行量化 VLM 实验)")
        else:
            print("  丹炉等级: 凡级丹炉 (建议使用 CLIP 实验)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  异火检测: Apple Silicon MPS")
        print("  火力强度: 统一内存架构")
        print("  丹炉等级: 灵级丹炉 (Apple Silicon)")
        device = "mps"
    else:
        print("  异火检测: 未检测到 GPU")
        print("  将使用凡火（CPU）进行修炼，速度较慢")
        device = "cpu"

    print(f"\n  修炼设备: {device}")
    print("=" * 65)
    return device


# ============================================================================
# 第一节：空间裂缝初探 —— CLIP 图文匹配
# ============================================================================

def experiment_01_clip_image_text_matching(device="cpu"):
    """
    实验一：使用 CLIP 进行图文匹配

    "空间裂缝的本质，是在图像维度与文字维度之间建立共鸣。
     当裂缝稳定后，你可以通过文字'看到'图像，
     也可以通过图像'读出'文字。"

    CLIP 通过对比学习，将图像和文本映射到同一向量空间。
    在这个共享空间中，语义相关的图文对距离更近。
    """
    print("\n" + "=" * 65)
    print("  实验一：空间裂缝初探 —— CLIP 图文匹配")
    print("=" * 65)

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import requests
    from io import BytesIO

    # ------------------------------------------------------------------
    # 步骤 1：加载 CLIP 模型（打开空间裂缝）
    # ------------------------------------------------------------------
    print("\n[1/4] 加载 CLIP 模型（撕开空间裂缝）...")

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print(f"  模型: {model_name}")
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    print(f"  视觉编码器参数量: {vision_params / 1e6:.1f}M")
    print(f"  文本编码器参数量: {text_params / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # 步骤 2：准备测试图像（从虚空中召唤影像）
    # ------------------------------------------------------------------
    print("\n[2/4] 准备测试图像...")

    # 使用本地生成的测试图像（避免网络依赖）
    # 创建简单的彩色测试图像
    import numpy as np

    def create_test_image(color_name, rgb_values, size=(224, 224)):
        """创建纯色测试图像（修炼用的简易幻象）"""
        img_array = np.full((*size, 3), rgb_values, dtype=np.uint8)
        return Image.fromarray(img_array)

    test_images = {
        "红色方块": create_test_image("red", [220, 50, 50]),
        "蓝色方块": create_test_image("blue", [50, 50, 220]),
        "绿色方块": create_test_image("green", [50, 200, 50]),
    }

    # ------------------------------------------------------------------
    # 步骤 3：定义候选文本描述
    # ------------------------------------------------------------------
    print("\n[3/4] 定义候选文本描述...")

    candidate_texts = [
        "a red colored square",
        "a blue colored square",
        "a green colored square",
        "a photo of a cat",
        "a photo of a sunset",
    ]

    for text in candidate_texts:
        print(f"    - {text}")

    # ------------------------------------------------------------------
    # 步骤 4：计算图文相似度（感受空间裂缝中的共鸣）
    # ------------------------------------------------------------------
    print("\n[4/4] 计算图文相似度矩阵...")

    images_list = list(test_images.values())
    image_names = list(test_images.keys())

    inputs = processor(
        text=candidate_texts,
        images=images_list,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (num_images, num_texts)
        probs = logits_per_image.softmax(dim=1)

    # 打印相似度矩阵
    print("\n  图文相似度矩阵（空间共鸣强度）:")
    print("  " + "-" * 60)

    # 表头
    header = "  图像 \\ 文本    |"
    for j, text in enumerate(candidate_texts):
        header += f" T{j+1}   |"
    print(header)
    print("  " + "-" * 60)

    # 每行
    for i, name in enumerate(image_names):
        row = f"  {name:12s} |"
        best_j = probs[i].argmax().item()
        for j in range(len(candidate_texts)):
            marker = " *" if j == best_j else "  "
            row += f" {probs[i][j]:.3f}{marker}|"
        print(row)

    print("  " + "-" * 60)
    print("  (* 标记表示最匹配的文本)")

    # 打印文本对应表
    print("\n  文本编号对照:")
    for j, text in enumerate(candidate_texts):
        print(f"    T{j+1}: {text}")

    # 打印每张图像的最佳匹配
    print("\n  匹配结果:")
    for i, name in enumerate(image_names):
        best_idx = probs[i].argmax().item()
        best_prob = probs[i][best_idx].item()
        print(f"    {name} → '{candidate_texts[best_idx]}' "
              f"(置信度: {best_prob:.4f})")

    print("\n  [空间裂缝实验完成] CLIP 成功在图像与文本之间建立了共鸣。")
    return model, processor


# ============================================================================
# 第二节：空间感知修炼 —— 视觉特征提取与可视化
# ============================================================================

def experiment_02_vision_feature_extraction(device="cpu"):
    """
    实验二：视觉特征提取与分析

    "空间感知的第一步，是将连续的光影分割为离散的元素。
     Vision Transformer 将图像切割为 patch，每个 patch
     就像虚空中的一个坐标点——可被认知、可被计算。"

    本实验展示：
    - ViT 如何将图像切割为 patch 序列
    - 每个 patch 的特征表示
    - 不同层输出的特征差异
    """
    print("\n" + "=" * 65)
    print("  实验二：空间感知修炼 —— 视觉特征提取")
    print("=" * 65)

    import torch
    import torch.nn as nn
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from PIL import Image
    import numpy as np

    # ------------------------------------------------------------------
    # 步骤 1：加载视觉编码器
    # ------------------------------------------------------------------
    print("\n[1/3] 加载视觉编码器（空间感知法器）...")

    model_name = "openai/clip-vit-base-patch32"
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    vision_model.eval()

    # 打印模型结构概要
    config = vision_model.config
    print(f"  模型: {model_name}")
    print(f"  图像尺寸: {config.image_size}x{config.image_size}")
    print(f"  Patch 尺寸: {config.patch_size}x{config.patch_size}")
    num_patches = (config.image_size // config.patch_size) ** 2
    print(f"  Patch 数量: {num_patches}")
    print(f"  隐藏维度: {config.hidden_size}")
    print(f"  注意力头数: {config.num_attention_heads}")
    print(f"  Transformer 层数: {config.num_hidden_layers}")

    # ------------------------------------------------------------------
    # 步骤 2：创建测试图像并提取特征
    # ------------------------------------------------------------------
    print("\n[2/3] 创建测试图像并提取多层特征...")

    # 创建一个有趣的渐变图像
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            gradient[i, j, 0] = int(i / 224 * 255)    # R: 从上到下递增
            gradient[i, j, 1] = int(j / 224 * 255)    # G: 从左到右递增
            gradient[i, j, 2] = 128                     # B: 固定
    test_image = Image.fromarray(gradient)

    # 预处理
    inputs = image_processor(images=test_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    print(f"  输入张量形状: {pixel_values.shape}")  # (1, 3, 224, 224)

    # 提取所有层的隐藏状态
    with torch.no_grad():
        outputs = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            output_attentions=True,
        )

    # ------------------------------------------------------------------
    # 步骤 3：分析各层特征
    # ------------------------------------------------------------------
    print("\n[3/3] 分析各层特征（空间感知的深度）...")

    hidden_states = outputs.hidden_states  # Tuple of (layers+1) x (1, seq_len, hidden_dim)
    attentions = outputs.attentions        # Tuple of layers x (1, heads, seq_len, seq_len)

    print(f"\n  隐藏层数量: {len(hidden_states)} (包含 embedding 层)")
    print(f"  注意力层数量: {len(attentions)}")

    # 分析每层特征的统计信息
    print("\n  各层特征统计:")
    print("  " + "-" * 55)
    print(f"  {'层':>4s} | {'形状':>18s} | {'均值':>8s} | {'标准差':>8s} | {'范数':>8s}")
    print("  " + "-" * 55)

    for layer_idx, hidden_state in enumerate(hidden_states):
        h = hidden_state[0]  # 去掉 batch 维度
        mean_val = h.mean().item()
        std_val = h.std().item()
        norm_val = h.norm(dim=-1).mean().item()
        print(f"  {layer_idx:4d} | {str(tuple(h.shape)):>18s} | "
              f"{mean_val:>8.4f} | {std_val:>8.4f} | {norm_val:>8.2f}")

    # 分析注意力模式
    print("\n  注意力模式分析（[CLS] token 对各 patch 的注意力）:")
    for layer_idx in [0, len(attentions) // 2, len(attentions) - 1]:
        attn = attentions[layer_idx][0]  # (heads, seq_len, seq_len)
        cls_attn = attn[:, 0, 1:]  # CLS 对所有 patch 的注意力 (heads, num_patches)
        avg_cls_attn = cls_attn.mean(dim=0)  # 所有头的平均

        # 找到注意力最高的 patch
        top_k = 3
        top_indices = avg_cls_attn.topk(top_k).indices
        top_values = avg_cls_attn.topk(top_k).values

        print(f"\n  第 {layer_idx} 层:")
        print(f"    注意力熵: {-(avg_cls_attn * avg_cls_attn.log()).sum().item():.4f}")
        print(f"    注意力最集中的 {top_k} 个 patch:")
        for k in range(top_k):
            patch_idx = top_indices[k].item()
            attn_val = top_values[k].item()
            # 将 patch 索引转换为空间位置
            grid_size = int(num_patches ** 0.5)
            row = patch_idx // grid_size
            col = patch_idx % grid_size
            print(f"      Patch {patch_idx} (行{row}, 列{col}): "
                  f"注意力 = {attn_val:.4f}")

    # 分析最终输出
    print("\n  最终输出分析:")
    final_hidden = hidden_states[-1][0]  # (seq_len, hidden_dim)
    cls_token = final_hidden[0]           # CLS token
    patch_tokens = final_hidden[1:]       # Patch tokens

    print(f"    [CLS] token 范数: {cls_token.norm().item():.4f}")
    print(f"    Patch tokens 平均范数: {patch_tokens.norm(dim=-1).mean().item():.4f}")
    print(f"    Patch tokens 范数标准差: {patch_tokens.norm(dim=-1).std().item():.4f}")

    # Patch 间的余弦相似度
    patch_normed = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
    cos_sim = torch.mm(patch_normed, patch_normed.t())
    print(f"    Patch 间平均余弦相似度: {cos_sim.mean().item():.4f}")
    print(f"    Patch 间最小余弦相似度: {cos_sim.min().item():.4f}")

    print("\n  [空间感知实验完成] 已深入分析视觉编码器的多层特征表示。")


# ============================================================================
# 第三节：空间桥梁搭建 —— 简易多模态管线
# ============================================================================

def experiment_03_simple_vlm_pipeline(device="cpu"):
    """
    实验三：构建简易视觉-语言管线

    "空间桥梁，是连接不同维度的通道。
     最简单的桥梁，是一座线性映射的独木桥——
     虽然简陋，但足以让信息在两个维度间流动。"

    本实验构建一个极简的 Vision-Language 管线：
    1. 用 CLIP Vision Encoder 提取图像特征
    2. 用 MLP 投影层将视觉特征映射到文本空间
    3. 展示如何将视觉 token 注入语言模型
    """
    print("\n" + "=" * 65)
    print("  实验三：空间桥梁搭建 —— 简易多模态管线")
    print("=" * 65)

    import torch
    import torch.nn as nn
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from PIL import Image
    import numpy as np

    # ------------------------------------------------------------------
    # 步骤 1：定义空间桥梁（MLP Projector）
    # ------------------------------------------------------------------
    print("\n[1/4] 定义空间桥梁（MLP Projector）...")

    class SpatialBridge(nn.Module):
        """
        空间桥梁：将视觉维度的特征映射到语言维度

        这是 LLaVA 架构中最关键的组件之一。
        看似简单的两层 MLP，却承担着跨越维度壁垒的重任。

        输入: 视觉特征 (batch, num_patches, vision_dim)
        输出: 语言空间特征 (batch, num_patches, llm_dim)
        """
        def __init__(self, vision_dim, llm_dim):
            super().__init__()
            # 两层 MLP + GELU 激活（LLaVA v1.5 方案）
            self.bridge = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),       # 非线性激活：让桥梁能弯曲
                nn.Linear(llm_dim, llm_dim),
            )
            # 统计参数量
            total_params = sum(p.numel() for p in self.parameters())
            print(f"    桥梁参数量: {total_params:,}")
            print(f"    输入维度: {vision_dim} (视觉空间)")
            print(f"    输出维度: {llm_dim} (语言空间)")

        def forward(self, vision_features):
            return self.bridge(vision_features)

    # ------------------------------------------------------------------
    # 步骤 2：加载视觉编码器
    # ------------------------------------------------------------------
    print("\n[2/4] 加载视觉编码器...")

    vision_model_name = "openai/clip-vit-base-patch32"
    vision_model = CLIPVisionModel.from_pretrained(vision_model_name).to(device)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    vision_model.eval()

    vision_dim = vision_model.config.hidden_size  # 768 for ViT-B
    print(f"    视觉编码器维度: {vision_dim}")

    # ------------------------------------------------------------------
    # 步骤 3：创建完整的多模态管线
    # ------------------------------------------------------------------
    print("\n[3/4] 创建完整的多模态管线...")

    # 假设我们的目标 LLM 的隐藏维度为 2048
    llm_dim = 2048

    # 创建空间桥梁
    projector = SpatialBridge(vision_dim, llm_dim).to(device)

    class MiniVLMPipeline(nn.Module):
        """
        简易多模态管线

        架构: Vision Encoder → Spatial Bridge → [与 LLM embedding 拼接]

        这是 LLaVA 架构的极简版本。
        真正的 LLaVA 后面还接了一个完整的 LLM，
        这里我们只展示到特征融合这一步。
        """
        def __init__(self, vision_encoder, projector, llm_dim):
            super().__init__()
            self.vision_encoder = vision_encoder
            self.projector = projector
            self.llm_dim = llm_dim

            # 冻结视觉编码器（使用预训练参数，不更新）
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        def encode_image(self, pixel_values):
            """
            完整的空间感知流程：
            图像 → Patch Embedding → Transformer → 视觉特征 → 投影 → 语言空间特征
            """
            # 步骤 a: 视觉编码
            with torch.no_grad():
                vision_outputs = self.vision_encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )

            # 取倒数第二层的输出（去掉 CLS token）
            # 经验表明倒数第二层的特征比最后一层更适合多模态任务
            image_features = vision_outputs.hidden_states[-2][:, 1:, :]

            # 步骤 b: 通过空间桥梁映射到语言空间
            visual_tokens = self.projector(image_features)

            return visual_tokens

        def prepare_multimodal_input(self, pixel_values, text_embeddings):
            """
            将视觉 token 和文本 token 融合为统一序列

            这是空间融合的关键步骤：
            [visual_token_1, visual_token_2, ..., text_token_1, text_token_2, ...]

            LLM 将所有 token 一视同仁地处理，
            不区分它们来自哪个维度。
            """
            visual_tokens = self.encode_image(pixel_values)

            # 拼接视觉和文本 token
            fused_input = torch.cat([visual_tokens, text_embeddings], dim=1)

            return fused_input, visual_tokens.shape[1]

    # 实例化管线
    pipeline = MiniVLMPipeline(vision_model, projector, llm_dim).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"    总参数量: {total_params:,}")
    print(f"    可训练参数量: {trainable_params:,}")
    print(f"    可训练占比: {trainable_params / total_params * 100:.2f}%")

    # ------------------------------------------------------------------
    # 步骤 4：运行管线
    # ------------------------------------------------------------------
    print("\n[4/4] 运行多模态管线...")

    # 创建测试图像
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_img)

    # 预处理图像
    inputs = image_processor(images=test_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    print(f"    输入图像张量: {pixel_values.shape}")

    # 模拟文本 embedding（实际中由 LLM 的 embedding 层产生）
    seq_len = 20  # 假设文本有 20 个 token
    fake_text_embeddings = torch.randn(1, seq_len, llm_dim).to(device)
    print(f"    模拟文本 embedding: {fake_text_embeddings.shape}")

    # 运行管线
    with torch.no_grad():
        fused_input, num_visual_tokens = pipeline.prepare_multimodal_input(
            pixel_values, fake_text_embeddings
        )

    print(f"\n    融合结果:")
    print(f"      视觉 token 数量: {num_visual_tokens}")
    print(f"      文本 token 数量: {seq_len}")
    print(f"      融合序列长度: {fused_input.shape[1]}")
    print(f"      融合序列形状: {fused_input.shape}")
    print(f"      每个 token 维度: {fused_input.shape[-1]} (= LLM 隐藏维度)")

    # 验证：视觉部分和文本部分在同一空间
    visual_part = fused_input[:, :num_visual_tokens, :]
    text_part = fused_input[:, num_visual_tokens:, :]

    print(f"\n    维度验证:")
    print(f"      视觉部分范数: {visual_part.norm(dim=-1).mean().item():.4f}")
    print(f"      文本部分范数: {text_part.norm(dim=-1).mean().item():.4f}")
    print(f"      维度一致: {visual_part.shape[-1] == text_part.shape[-1]}")

    print("\n  [空间桥梁实验完成] 多模态管线已搭建，视觉和文本特征成功融合。")
    print("  在真正的 LLaVA 中，这个融合后的序列会被送入 LLM 进行生成。")


# ============================================================================
# 第四节：空间法器驾驭 —— 使用预训练 VLM
# ============================================================================

def experiment_04_pretrained_vlm_inference(device="cpu"):
    """
    实验四：使用预训练多模态大模型进行视觉问答

    "真正的斗宗强者，不会从零修炼空间之力——
     而是驾驭前人已经炼制好的空间法器。
     Qwen2-VL、LLaVA、InternVL……这些都是当世顶级的空间法器。"

    本实验展示如何使用 HuggingFace 上的预训练 VLM 进行推理。
    注意：完整的 VLM 推理需要较大显存，这里提供示例代码和轻量替代方案。
    """
    print("\n" + "=" * 65)
    print("  实验四：空间法器驾驭 —— 预训练 VLM 推理")
    print("=" * 65)

    import torch

    # ------------------------------------------------------------------
    # 方案 A：使用 CLIP 进行"穷人版"视觉问答
    # （无需大显存，用 CLIP 的零样本能力模拟 VQA）
    # ------------------------------------------------------------------
    print("\n[方案 A] 基于 CLIP 的零样本视觉问答（轻量方案）")
    print("-" * 50)

    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import numpy as np

    # 加载 CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # 创建一个包含多种颜色区域的测试图像
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    img_array[:112, :112] = [255, 0, 0]      # 左上：红色
    img_array[:112, 112:] = [0, 0, 255]      # 右上：蓝色
    img_array[112:, :112] = [0, 255, 0]      # 左下：绿色
    img_array[112:, 112:] = [255, 255, 0]    # 右下：黄色
    test_image = Image.fromarray(img_array)

    # 模拟视觉问答：通过设计候选答案，用 CLIP 选择最佳匹配
    questions_and_candidates = [
        {
            "question": "图中包含哪些颜色?",
            "candidates": [
                "an image with red blue green and yellow colors",
                "an image with only black and white colors",
                "an image with purple and orange colors",
            ],
        },
        {
            "question": "这张图是什么类型?",
            "candidates": [
                "a colorful geometric pattern with four quadrants",
                "a photograph of a natural landscape",
                "a portrait of a person",
            ],
        },
        {
            "question": "图的左上角是什么颜色?",
            "candidates": [
                "the top left area is red",
                "the top left area is blue",
                "the top left area is green",
            ],
        },
    ]

    for qa in questions_and_candidates:
        print(f"\n  问题: {qa['question']}")

        inputs = processor(
            text=qa["candidates"],
            images=test_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        best_idx = probs.argmax().item()
        print(f"  最佳答案: {qa['candidates'][best_idx]}")
        print(f"  置信度: {probs[best_idx].item():.4f}")
        print(f"  所有候选得分:")
        for i, (text, prob) in enumerate(zip(qa["candidates"], probs)):
            marker = " <<<" if i == best_idx else ""
            print(f"    [{prob:.4f}] {text}{marker}")

    # ------------------------------------------------------------------
    # 方案 B：完整 VLM 推理代码（需要较大显存）
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("[方案 B] 完整 VLM 推理代码（参考实现）")
    print("-" * 50)

    print("""
  以下代码展示了如何使用 Qwen2-VL 进行真正的视觉问答。
  运行此代码需要至少 16GB 显存（或使用 4-bit 量化降至 ~8GB）。

  ---[ 参考代码 ]---

  from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
  from PIL import Image
  import torch

  # 加载空间法器（Qwen2-VL-7B）
  model = Qwen2VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2-VL-7B-Instruct",
      torch_dtype=torch.float16,
      device_map="auto",
  )
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

  # 准备图像和问题
  image = Image.open("your_image.jpg")
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "image": image},
              {"type": "text", "text": "What do you see in this image?"},
          ],
      }
  ]

  # 推理
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  inputs = processor(
      text=[text], images=[image],
      padding=True, return_tensors="pt"
  ).to(model.device)

  output_ids = model.generate(**inputs, max_new_tokens=256)
  response = processor.batch_decode(
      output_ids[:, inputs.input_ids.shape[1]:],
      skip_special_tokens=True,
  )[0]
  print(response)

  ---[ 代码结束 ]---
    """)

    # ------------------------------------------------------------------
    # 方案 C：使用 4-bit 量化的轻量方案
    # ------------------------------------------------------------------
    print("-" * 50)
    print("[方案 C] 4-bit 量化 VLM 推理代码（节省显存）")
    print("-" * 50)

    print("""
  使用 BitsAndBytes 4-bit 量化，可在 ~8GB 显存上运行 7B VLM：

  ---[ 参考代码 ]---

  from transformers import (
      Qwen2VLForConditionalGeneration,
      AutoProcessor,
      BitsAndBytesConfig,
  )
  import torch

  # 4-bit 量化配置（压缩空间法器以适应较小的丹炉）
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
  )

  model = Qwen2VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2-VL-7B-Instruct",
      quantization_config=bnb_config,
      device_map="auto",
  )
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

  # 其余推理代码与方案 B 相同...

  ---[ 代码结束 ]---

  显存估算:
    FP16 加载: ~14 GB
    4-bit 量化: ~4 GB
    推理时峰值: ~6-8 GB (取决于图像分辨率)
    一块 RTX 4060 (8GB) 即可胜任！
    """)

    print("\n  [空间法器实验完成]")


# ============================================================================
# 第五节：CLIP 特征空间探索 —— 深入空间裂缝
# ============================================================================

def experiment_05_clip_feature_space(device="cpu"):
    """
    实验五：CLIP 特征空间深度探索

    "空间裂缝的内部结构，远比表面看起来更加复杂。
     图像和文本在共享空间中如何分布？
     相似概念之间的距离如何变化？
     让我们深入裂缝内部一探究竟。"

    本实验分析 CLIP 共享特征空间的结构：
    - 图像特征和文本特征的分布
    - 语义相近概念的聚类
    - 跨模态检索能力验证
    """
    print("\n" + "=" * 65)
    print("  实验五：CLIP 特征空间探索 —— 深入空间裂缝")
    print("=" * 65)

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    import numpy as np

    # 加载 CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # ------------------------------------------------------------------
    # 实验 A：文本概念的语义空间分布
    # ------------------------------------------------------------------
    print("\n[实验 A] 文本概念在共享空间中的分布")
    print("-" * 50)

    # 准备语义相关的文本组
    text_groups = {
        "动物": [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a bird",
            "a photo of a fish",
        ],
        "交通工具": [
            "a photo of a car",
            "a photo of a bus",
            "a photo of a bicycle",
            "a photo of an airplane",
        ],
        "自然": [
            "a photo of a mountain",
            "a photo of an ocean",
            "a photo of a forest",
            "a photo of a desert",
        ],
    }

    # 提取所有文本特征
    all_texts = []
    all_labels = []
    for group_name, texts in text_groups.items():
        all_texts.extend(texts)
        all_labels.extend([group_name] * len(texts))

    inputs = processor(text=all_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算组内和组间余弦相似度
    similarity_matrix = torch.mm(text_features, text_features.t())

    print("\n  组内平均相似度（同一类概念间的距离）:")
    group_names = list(text_groups.keys())
    for g_idx, group_name in enumerate(group_names):
        start = g_idx * 4
        end = start + 4
        group_sim = similarity_matrix[start:end, start:end]
        # 排除对角线（自身相似度=1）
        mask = ~torch.eye(4, dtype=torch.bool).to(device)
        avg_sim = group_sim[mask].mean().item()
        print(f"    {group_name}: {avg_sim:.4f}")

    print("\n  组间平均相似度（不同类概念间的距离）:")
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            start_i, end_i = i * 4, (i + 1) * 4
            start_j, end_j = j * 4, (j + 1) * 4
            cross_sim = similarity_matrix[start_i:end_i, start_j:end_j]
            avg_sim = cross_sim.mean().item()
            print(f"    {group_names[i]} ↔ {group_names[j]}: {avg_sim:.4f}")

    # ------------------------------------------------------------------
    # 实验 B：图像到文本的跨模态检索
    # ------------------------------------------------------------------
    print("\n\n[实验 B] 跨模态检索验证")
    print("-" * 50)

    # 创建不同类型的测试图像
    test_images = {}

    # 红色图像 → 应该匹配 "warm" 或 "red" 相关的文本
    red_img = np.full((224, 224, 3), [200, 30, 30], dtype=np.uint8)
    test_images["红色图像"] = Image.fromarray(red_img)

    # 蓝色图像 → 应该匹配 "cool" 或 "blue" 相关的文本
    blue_img = np.full((224, 224, 3), [30, 30, 200], dtype=np.uint8)
    test_images["蓝色图像"] = Image.fromarray(blue_img)

    # 明亮图像 → 应该匹配 "bright" 相关的文本
    bright_img = np.full((224, 224, 3), [240, 240, 240], dtype=np.uint8)
    test_images["明亮图像"] = Image.fromarray(bright_img)

    # 深色图像 → 应该匹配 "dark" 相关的文本
    dark_img = np.full((224, 224, 3), [20, 20, 20], dtype=np.uint8)
    test_images["深色图像"] = Image.fromarray(dark_img)

    retrieval_texts = [
        "a red colored image",
        "a blue colored image",
        "a bright white image",
        "a dark black image",
        "a green colored image",
    ]

    images_list = list(test_images.values())
    image_names = list(test_images.keys())

    inputs = processor(
        text=retrieval_texts,
        images=images_list,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # (num_images, num_texts)
        probs = logits.softmax(dim=1)

    print("\n  跨模态检索结果:")
    for i, img_name in enumerate(image_names):
        best_idx = probs[i].argmax().item()
        print(f"\n    {img_name}:")
        print(f"      最匹配: '{retrieval_texts[best_idx]}' "
              f"(得分: {probs[i][best_idx]:.4f})")

    # ------------------------------------------------------------------
    # 实验 C：特征空间的模态间隙 (Modality Gap)
    # ------------------------------------------------------------------
    print("\n\n[实验 C] 模态间隙分析 (Modality Gap)")
    print("-" * 50)

    # 提取图像和文本特征
    img_inputs = processor(images=images_list, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

    txt_inputs = processor(text=retrieval_texts, return_tensors="pt", padding=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    with torch.no_grad():
        img_features = model.get_image_features(**img_inputs)
        txt_features = model.get_text_features(**txt_inputs)

        # L2 归一化
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

    # 计算模态中心点
    img_center = img_features.mean(dim=0)
    txt_center = txt_features.mean(dim=0)

    # 模态间隙 = 两个模态中心点之间的距离
    modality_gap = (img_center - txt_center).norm().item()
    cosine_gap = torch.dot(img_center, txt_center).item() / (
        img_center.norm().item() * txt_center.norm().item()
    )

    print(f"\n  图像特征中心范数: {img_center.norm().item():.4f}")
    print(f"  文本特征中心范数: {txt_center.norm().item():.4f}")
    print(f"  模态间隙 (L2 距离): {modality_gap:.4f}")
    print(f"  模态间隙 (余弦相似度): {cosine_gap:.4f}")

    # 模态内部方差
    img_var = ((img_features - img_center).norm(dim=-1) ** 2).mean().item()
    txt_var = ((txt_features - txt_center).norm(dim=-1) ** 2).mean().item()
    print(f"\n  图像特征内部方差: {img_var:.4f}")
    print(f"  文本特征内部方差: {txt_var:.4f}")

    print("\n  [分析] CLIP 的特征空间存在'模态间隙'——图像特征和文本特征")
    print("  虽然在同一空间中，但各自占据不同的区域。这正是 LLaVA 需要")
    print("  MLP Projector 来弥合这一间隙的原因。")

    print("\n  [空间裂缝探索完成] CLIP 特征空间的结构已深入分析。")


# ============================================================================
# 主函数：修炼流程编排
# ============================================================================

def main():
    """
    焚诀 · 第七卷 · 空间篇 · 修炼总纲

    "修炼空间之力，需循序渐进：
     先感知空间（视觉编码），
     再撕裂空间（CLIP 对比学习），
     然后搭建桥梁（MLP 投影），
     最终驾驭法器（VLM 推理）。"

    运行方式:
      python 07_多模态模型实战.py          # 运行所有实验
      python 07_多模态模型实战.py 1        # 只运行实验 1
      python 07_多模态模型实战.py 1 3      # 运行实验 1 和 3
    """

    # 检测丹炉
    device = detect_furnace()

    # 解析命令行参数
    if len(sys.argv) > 1:
        experiments = [int(x) for x in sys.argv[1:]]
    else:
        experiments = [1, 2, 3, 4, 5]

    print(f"\n  将执行以下实验: {experiments}")
    print(f"  修炼设备: {device}")

    # 实验调度
    experiment_map = {
        1: ("空间裂缝初探 (CLIP 图文匹配)", experiment_01_clip_image_text_matching),
        2: ("空间感知修炼 (视觉特征提取)", experiment_02_vision_feature_extraction),
        3: ("空间桥梁搭建 (简易多模态管线)", experiment_03_simple_vlm_pipeline),
        4: ("空间法器驾驭 (预训练 VLM 推理)", experiment_04_pretrained_vlm_inference),
        5: ("空间裂缝探索 (CLIP 特征空间)", experiment_05_clip_feature_space),
    }

    for exp_id in experiments:
        if exp_id in experiment_map:
            name, func = experiment_map[exp_id]
            print(f"\n{'#' * 65}")
            print(f"  开始实验 {exp_id}: {name}")
            print(f"{'#' * 65}")
            try:
                func(device)
            except Exception as e:
                print(f"\n  [反噬警告] 实验 {exp_id} 出现异常: {e}")
                print("  可能原因: 模型下载失败 / 显存不足 / 依赖缺失")
                print("  建议: 检查网络连接和丹炉资源后重试")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n  [警告] 未知实验编号: {exp_id}")

    # 修炼总结
    print("\n" + "=" * 65)
    print("  焚诀 · 第七卷 · 空间篇 · 修炼完成")
    print("=" * 65)
    print("""
  恭喜你完成了空间篇的实战修炼！

  你已掌握:
    - CLIP 图文匹配与零样本分类
    - Vision Transformer 特征提取与分析
    - 多模态管线构建（Vision Encoder → MLP → LLM）
    - 预训练 VLM 的使用方法
    - CLIP 特征空间的深度分析

  下一步修炼建议:
    1. 在真实图像数据集上运行 CLIP 零样本分类
    2. 使用 Qwen2-VL 或 LLaVA 进行视觉问答
    3. 尝试用 LoRA 微调一个 VLM 到特定领域
    4. 阅读 LLaVA 和 BLIP-2 的原始论文

  "空间之力已得。维度壁垒已破。
   前方，九转之秘等待着你。"

              —— 焚诀 · 空间篇 · 完 ——
    """)


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    main()
