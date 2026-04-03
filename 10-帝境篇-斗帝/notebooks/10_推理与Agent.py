"""
╔══════════════════════════════════════════════════════════════╗
║           《焚诀》第十卷 · 帝境篇 — 推理与 Agent             ║
║                                                              ║
║  斗帝不依赖外力，他自己就是法则。                                ║
║  本丹方（脚本）演示：                                          ║
║    1. 简易 ReAct Agent — 具备工具调用能力的自主推理体           ║
║    2. Chain-of-Thought 提示 — 让模型展示推理过程               ║
║    3. Self-Consistency — 多路径投票提升可靠性                   ║
║                                                              ║
║  术语对照：                                                    ║
║    GPU = 异火        数据集 = 药材      训练 = 炼丹            ║
║    Loss NaN/OOM = 反噬   模型参数 = 斗气                      ║
║    算法/架构 = 斗技       训练脚本 = 丹方    服务器 = 丹炉      ║
║                                                              ║
║  运行方式:                                                     ║
║    pip install openai    # 需要 OpenAI API key                ║
║    python 10_推理与Agent.py                                   ║
║                                                              ║
║  注意：本脚本可在无 API key 的情况下以 mock 模式运行            ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import re
import os
import math
import random
from collections import Counter
from typing import Optional, Callable
from dataclasses import dataclass, field

# ============================================================
# 第一节：工具定义 — 斗帝之兵器库
# ============================================================
# 斗帝虽已超凡，仍需借天地之力。
# 此处定义 Agent 可调用的"神器"（工具函数）。

@dataclass
class ToolResult:
    """工具调用的返回结果 — 犹如神器反馈的灵力波动"""
    tool_name: str
    input_args: dict
    output: str
    success: bool = True


def tool_calculator(expression: str) -> str:
    """
    计算器工具 — 算道之器
    斗帝虽精通万法，但精确计算仍需专用法器。
    支持基本数学运算。
    """
    try:
        # 安全地评估数学表达式（仅允许数学运算）
        allowed_names = {
            "abs": abs, "round": round,
            "min": min, "max": max,
            "pow": pow, "sqrt": math.sqrt,
            "pi": math.pi, "e": math.e,
            "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败（反噬！）: {str(e)}"


def tool_search(query: str) -> str:
    """
    搜索工具 — 万界之眼
    斗帝可洞察天地万物的信息。
    此处为模拟搜索，实际应用可接入真实搜索 API。
    """
    # 模拟知识库 — 犹如斗帝的万年记忆
    knowledge_base = {
        "transformer": (
            "Transformer 是 Vaswani et al. 2017 年提出的架构，"
            "基于自注意力机制（Self-Attention），已成为 NLP 和多模态 AI 的基石。"
            "核心组件：Multi-Head Attention, Feed-Forward Network, Layer Normalization。"
        ),
        "deepseek-r1": (
            "DeepSeek-R1 是 DeepSeek 于 2025 年 1 月发布的推理模型。"
            "关键创新：通过纯强化学习（无 SFT）训练出 Chain-of-Thought 推理能力。"
            "R1-Zero 展示了推理能力可以从奖励信号中自发涌现。"
            "模型规模：671B 参数（MoE 架构），同时发布了 1.5B-70B 的蒸馏版本。"
        ),
        "scaling laws": (
            "Scaling Laws（Kaplan et al., 2020）发现模型性能与参数量、数据量、计算量之间"
            "存在幂律关系：L(N) ∝ N^(-0.076)。Chinchilla Laws（Hoffmann et al., 2022）进一步"
            "指出参数和数据应同步扩大。当前 GPT-4 级模型约 1-2T 参数（推测）。"
        ),
        "rlhf": (
            "RLHF（Reinforcement Learning from Human Feedback）是当前 LLM 对齐的主流方法。"
            "流程：SFT → 训练奖励模型 → PPO/DPO 优化。"
            "替代方案：DPO（Direct Preference Optimization）、KTO、ORPO 等。"
        ),
        "mcts": (
            "Monte Carlo Tree Search (MCTS) 是一种搜索算法，通过四个阶段运作："
            "Selection → Expansion → Simulation → Backpropagation。"
            "在 AlphaGo 中取得巨大成功，现被应用于 LLM 推理搜索。"
        ),
        "诺贝尔物理学奖 2024": (
            "2024 年诺贝尔物理学奖授予 John Hopfield（91 岁）和 Geoffrey Hinton（76 岁），"
            "以表彰他们在人工神经网络和机器学习领域的基础性发现。"
        ),
        "python 创始人": (
            "Python 由 Guido van Rossum 于 1991 年创建。"
            "他被称为 Python 的 'Benevolent Dictator For Life (BDFL)'。"
        ),
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"搜索结果（万界之眼）：{value}"

    return f"搜索 '{query}' 未找到相关信息。万界之眼尚未收录此知识。"


def tool_code_executor(code: str) -> str:
    """
    代码执行工具 — 造化之笔
    斗帝可令法则具现化——即代码执行。
    仅用于演示，实际生产环境需严格沙箱隔离。
    """
    try:
        # 极其简化的安全执行（仅作演示用途）
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                      "sum": sum, "list": list, "dict": dict,
                                      "str": str, "int": int, "float": float,
                                      "sorted": sorted, "enumerate": enumerate,
                                      "zip": zip, "map": map, "filter": filter}},
             local_vars)
        if "result" in local_vars:
            return f"代码执行成功，result = {local_vars['result']}"
        return "代码执行成功（无返回值）"
    except Exception as e:
        return f"代码执行反噬: {str(e)}"


# ============================================================
# 第二节：ReAct Agent — 斗帝的自主推理体
# ============================================================
# ReAct = Reasoning + Acting
# 斗帝在战斗中交替进行"思考"和"行动"，
# 根据观察结果不断调整策略。

# 所有可用工具的注册表 — 斗帝的纳戒
TOOL_REGISTRY: dict[str, Callable] = {
    "calculator": tool_calculator,
    "search": tool_search,
    "code_executor": tool_code_executor,
}

# 工具描述，供 Agent 了解每件"神器"的功能
TOOL_DESCRIPTIONS = """
可用工具（神器列表）：

1. calculator(expression: str) -> str
   - 功能：计算数学表达式
   - 示例：calculator("3.14 * 2 ** 2")

2. search(query: str) -> str
   - 功能：搜索知识库中的信息
   - 示例：search("transformer 架构")

3. code_executor(code: str) -> str
   - 功能：执行 Python 代码（须在代码中设置 result 变量获取返回值）
   - 示例：code_executor("result = sum(range(100))")
"""


@dataclass
class AgentStep:
    """Agent 的一个推理步骤 — 斗帝出招的一个回合"""
    step_number: int
    thought: str       # 思考：斗帝的内心推演
    action: str        # 行动：调用何种神器
    action_input: str  # 行动输入：神器的使用方式
    observation: str    # 观察：神器返回的结果


@dataclass
class AgentTrace:
    """Agent 的完整推理轨迹 — 斗帝的战斗全记录"""
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0


class ReActAgent:
    """
    ReAct Agent — 斗帝级自主推理体

    核心循环：
    1. Thought（思考）：分析当前状态，决定下一步
    2. Action（行动）：调用工具获取信息
    3. Observation（观察）：接收工具返回的结果
    4. 重复直到可以给出最终答案

    此实现为纯本地模拟版本，不依赖外部 API。
    实际应用中，思考步骤由 LLM 生成。
    """

    def __init__(self, max_steps: int = 5, verbose: bool = True):
        """
        初始化斗帝推理体

        Args:
            max_steps: 最大推理步数（防止无限循环——即"走火入魔"）
            verbose: 是否打印详细推理过程
        """
        self.max_steps = max_steps
        self.verbose = verbose
        self.tools = TOOL_REGISTRY

    def _think(self, question: str, history: list[AgentStep]) -> tuple[str, str, str]:
        """
        模拟 LLM 的思考过程 — 斗帝的内心推演

        在真实系统中，这里会调用 LLM API 来生成思考、行动和输入。
        此处为演示，使用基于规则的模拟。

        Returns:
            (thought, action, action_input) 三元组
        """
        step_num = len(history) + 1

        # ---- 基于规则的推理模拟 ----
        q_lower = question.lower()

        # 若历史步骤已有搜索结果，尝试总结回答
        if history:
            last_obs = history[-1].observation
            # 如果上一步搜索到了有用信息，直接回答
            if "搜索结果" in last_obs and step_num >= 2:
                # 检查是否还需要计算
                if any(kw in question for kw in ["多少", "计算", "总和", "年龄"]):
                    # 需要进一步计算
                    if not any(s.action == "calculator" for s in history):
                        numbers = re.findall(r'\d+', last_obs)
                        if len(numbers) >= 2:
                            expr = " + ".join(numbers[-2:])
                            return (
                                f"从搜索结果中找到了关键数字：{numbers[-2:]}。"
                                f"现在需要计算它们的总和。",
                                "calculator",
                                expr
                            )
                return (
                    "已获得足够信息，可以给出最终答案。",
                    "finish",
                    self._summarize(question, history)
                )

            # 如果上一步是计算，可以总结了
            if "计算结果" in last_obs:
                return (
                    "计算完成，现在可以综合所有信息给出最终答案。",
                    "finish",
                    self._summarize(question, history)
                )

        # ---- 首步：决定使用什么工具 ----
        if any(kw in q_lower for kw in ["计算", "多少", "等于", "加", "减", "乘", "除"]):
            # 数学问题 → 可能先搜索再计算
            if "谁" in question or "什么" in question:
                return (
                    f"问题涉及事实查询和计算，我需要先搜索相关信息。",
                    "search",
                    question
                )
            # 纯计算问题
            numbers = re.findall(r'[\d.]+\s*[\+\-\*\/\%\^]\s*[\d.]+', question)
            if numbers:
                return (
                    "这是一个纯计算问题，直接使用计算器。",
                    "calculator",
                    numbers[0]
                )

        if any(kw in q_lower for kw in ["什么是", "解释", "介绍", "who", "what", "how"]):
            return (
                f"这是一个知识查询问题，我需要搜索相关信息。",
                "search",
                question
            )

        if "代码" in question or "编程" in question or "程序" in question:
            return (
                "用户需要代码相关帮助，让我编写并执行代码。",
                "code_executor",
                'result = "请提供具体的编程需求"'
            )

        # 默认：先搜索
        return (
            f"让我先搜索相关信息来回答这个问题。",
            "search",
            question
        )

    def _summarize(self, question: str, history: list[AgentStep]) -> str:
        """
        综合所有信息生成最终答案 — 斗帝的终极裁决

        Args:
            question: 原始问题
            history: 推理历史

        Returns:
            最终答案字符串
        """
        observations = []
        for step in history:
            if step.observation and "反噬" not in step.observation:
                observations.append(step.observation)

        combined = " | ".join(observations)

        # 简单的答案提取
        if "计算结果" in combined:
            calc_result = re.search(r'计算结果.*?=\s*(.+?)(?:\s|$)', combined)
            if calc_result:
                return f"根据搜索和计算，最终答案是：{calc_result.group(1)}。来源信息：{combined}"

        return f"根据查询结果：{combined}"

    def execute(self, question: str) -> AgentTrace:
        """
        执行 ReAct 推理循环 — 斗帝出手

        这是 Agent 的核心方法。模拟 Thought → Action → Observation 的循环，
        直到得出最终答案或达到最大步数。

        Args:
            question: 用户的问题

        Returns:
            AgentTrace: 完整的推理轨迹
        """
        trace = AgentTrace(question=question)

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"  斗帝推理体启动 — ReAct Agent")
            print(f"  问题：{question}")
            print("=" * 60)

        for step_num in range(1, self.max_steps + 1):
            # 第一阶段：思考（Thought）
            thought, action, action_input = self._think(question, trace.steps)

            if self.verbose:
                print(f"\n--- 第 {step_num} 回合 ---")
                print(f"  💭 思考：{thought}")
                print(f"  ⚔️ 行动：{action}")
                print(f"  📋 输入：{action_input}")

            # 第二阶段：行动（Action）
            if action == "finish":
                # 推理完成，给出最终答案
                trace.final_answer = action_input
                trace.total_steps = step_num
                if self.verbose:
                    print(f"\n{'=' * 60}")
                    print(f"  最终答案：{action_input}")
                    print(f"  总步数：{step_num}")
                    print(f"{'=' * 60}")
                return trace

            # 调用工具
            if action in self.tools:
                observation = self.tools[action](action_input)
            else:
                observation = f"未知工具 '{action}'（反噬：法器不存在！）"

            if self.verbose:
                print(f"  👁️ 观察：{observation}")

            # 记录这一步
            step = AgentStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            )
            trace.steps.append(step)

        # 超过最大步数，强制结束（走火入魔保护）
        trace.final_answer = "推理步数已达上限，未能得出最终答案（推理反噬：走火入魔保护触发）"
        trace.total_steps = self.max_steps

        if self.verbose:
            print(f"\n⚠️ 达到最大步数 {self.max_steps}，强制终止。")

        return trace


# ============================================================
# 第三节：Chain-of-Thought 提示 — 让模型展示推理过程
# ============================================================
# CoT 是推理增强的基石斗技。
# 通过引导模型"出声思考"，大幅提升推理准确率。

class ChainOfThoughtPrompter:
    """
    Chain-of-Thought 提示构建器 — 推理引导术

    支持三种模式：
    1. Zero-shot CoT: 简单地要求模型"逐步思考"
    2. Few-shot CoT: 提供带推理过程的示例
    3. Self-Consistency: 多路径采样 + 投票
    """

    # Zero-shot CoT 的经典后缀 — 一句话激活推理
    ZERO_SHOT_SUFFIX = "让我们一步一步来思考这个问题。\n\n"

    # Few-shot CoT 示例 — 推理的"丹方"示例
    FEW_SHOT_EXAMPLES = [
        {
            "question": "小明有 8 颗灵石，他给了小红 3 颗，然后又从矿洞中挖到了 5 颗。小明现在有几颗灵石？",
            "reasoning": (
                "第一步：小明一开始有 8 颗灵石。\n"
                "第二步：他给了小红 3 颗，所以还剩 8 - 3 = 5 颗。\n"
                "第三步：他又挖到了 5 颗，所以现在有 5 + 5 = 10 颗。\n"
                "所以，小明现在有 10 颗灵石。"
            ),
            "answer": "10"
        },
        {
            "question": "一个炼丹炉每次可以炼制 4 颗丹药。如果需要炼制 20 颗丹药，至少需要炼制几次？",
            "reasoning": (
                "第一步：每次炼制 4 颗丹药。\n"
                "第二步：需要总共 20 颗丹药。\n"
                "第三步：20 ÷ 4 = 5 次。\n"
                "所以，至少需要炼制 5 次。"
            ),
            "answer": "5"
        },
    ]

    def build_zero_shot_prompt(self, question: str) -> str:
        """
        构建 Zero-shot CoT 提示

        仅需在问题后加上"让我们一步一步思考"，
        便可激活模型的推理能力——犹如以一句口诀催动斗技。

        Args:
            question: 用户问题

        Returns:
            构建好的 prompt
        """
        prompt = f"问题：{question}\n\n{self.ZERO_SHOT_SUFFIX}"
        return prompt

    def build_few_shot_prompt(self, question: str) -> str:
        """
        构建 Few-shot CoT 提示

        提供几个带完整推理过程的示例，
        让模型学习推理的"格式"和"风格"——犹如传授斗技的招式。

        Args:
            question: 用户问题

        Returns:
            构建好的 prompt
        """
        prompt = "请按照以下示例的推理格式来回答问题。\n\n"

        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            prompt += f"--- 示例 {i} ---\n"
            prompt += f"问题：{example['question']}\n"
            prompt += f"推理过程：\n{example['reasoning']}\n"
            prompt += f"最终答案：{example['answer']}\n\n"

        prompt += "--- 你的问题 ---\n"
        prompt += f"问题：{question}\n"
        prompt += "推理过程：\n"

        return prompt

    def demonstrate(self, question: str) -> None:
        """
        演示 CoT 提示的构建过程

        Args:
            question: 用户问题
        """
        print("\n" + "=" * 60)
        print("  Chain-of-Thought 推理引导术 — 演示")
        print("=" * 60)

        print("\n[模式 1] Zero-shot CoT:")
        print("-" * 40)
        print(self.build_zero_shot_prompt(question))

        print("\n[模式 2] Few-shot CoT:")
        print("-" * 40)
        print(self.build_few_shot_prompt(question))


# ============================================================
# 第四节：Self-Consistency — 多路径投票
# ============================================================
# 一条推理路径可能有误，但万条路径的共识更可靠。
# 犹如万人共识之道——集众家之长，取最可信之解。

class SelfConsistencyVoter:
    """
    Self-Consistency 投票器 — 万人共识之道

    核心思想：
    1. 对同一问题，采样多条不同的推理路径
    2. 每条路径产生一个最终答案
    3. 多数投票，选择出现次数最多的答案

    这是最简单但出奇有效的推理增强方法之一。
    """

    def __init__(self, n_paths: int = 10):
        """
        Args:
            n_paths: 采样的推理路径数量（越多越可靠，但计算成本越高）
        """
        self.n_paths = n_paths

    def simulate_reasoning_paths(self, question: str) -> list[dict]:
        """
        模拟多条推理路径 — 万种可能的推演

        在真实系统中，这里会用较高 temperature 调用 LLM 多次。
        此处为演示，使用模拟数据。

        Args:
            question: 用户问题

        Returns:
            推理路径列表，每条包含推理过程和最终答案
        """
        # 模拟：大多数路径给出正确答案，少数路径犯错
        # 这模拟了 LLM 在高 temperature 下的行为
        paths = []

        # 模拟问题：一个丹炉有 3 层，每层放 4 份药材，总共需要多少份药材？
        correct_answer = 12
        for i in range(self.n_paths):
            # 80% 概率给出正确推理
            if random.random() < 0.8:
                reasoning = (
                    f"路径 {i+1}：丹炉有 3 层，每层放 4 份药材。"
                    f"总药材 = 3 × 4 = 12 份。"
                )
                answer = correct_answer
            else:
                # 20% 概率犯错
                wrong = random.choice([8, 7, 16, 15])
                reasoning = f"路径 {i+1}：（推理过程中出现了计算错误）得到 {wrong} 份。"
                answer = wrong

            paths.append({
                "path_id": i + 1,
                "reasoning": reasoning,
                "answer": answer
            })

        return paths

    def vote(self, paths: list[dict]) -> dict:
        """
        多数投票 — 万人共识

        Args:
            paths: 推理路径列表

        Returns:
            投票结果，包含最终答案和置信度
        """
        answers = [p["answer"] for p in paths]
        counter = Counter(answers)
        total = len(answers)

        # 获取得票最多的答案
        most_common = counter.most_common()
        winner = most_common[0]

        result = {
            "final_answer": winner[0],
            "vote_count": winner[1],
            "confidence": winner[1] / total,
            "total_paths": total,
            "vote_distribution": dict(counter),
            "all_candidates": most_common
        }

        return result

    def demonstrate(self, question: str) -> None:
        """
        演示 Self-Consistency 的完整流程

        Args:
            question: 用户问题
        """
        print("\n" + "=" * 60)
        print("  Self-Consistency 万人共识之道 — 演示")
        print("=" * 60)
        print(f"\n问题：{question}")
        print(f"采样路径数：{self.n_paths}")
        print("-" * 40)

        # 采样多条推理路径
        paths = self.simulate_reasoning_paths(question)

        # 打印每条路径
        for p in paths:
            status = "✅" if p["answer"] == 12 else "❌"
            print(f"  {status} {p['reasoning']} → 答案: {p['answer']}")

        # 投票
        result = self.vote(paths)

        print(f"\n{'=' * 40}")
        print(f"  投票结果：")
        print(f"  最终答案：{result['final_answer']}")
        print(f"  置信度：{result['confidence']:.1%} ({result['vote_count']}/{result['total_paths']})")
        print(f"  票数分布：{result['vote_distribution']}")

        if result['confidence'] >= 0.7:
            print(f"  判定：高置信度 ✅ — 斗帝之眼确认此答案可信。")
        elif result['confidence'] >= 0.5:
            print(f"  判定：中等置信度 ⚠️ — 需更多推理路径验证。")
        else:
            print(f"  判定：低置信度 ❌ — 答案不可靠，建议增加采样数。")


# ============================================================
# 第五节：简易 Best-of-N 评估 — 择优而选
# ============================================================
# Best-of-N：生成多个回答，选出最优者。
# 犹如炼丹百次，择其品质最高者。

class BestOfNSelector:
    """
    Best-of-N 选择器 — 百炼择优

    核心思想：
    1. 生成 N 个候选回答
    2. 用评分函数对每个回答打分
    3. 选择得分最高的回答
    """

    @staticmethod
    def score_reasoning(reasoning: str) -> float:
        """
        推理质量评分函数 — 简易版 Process Reward Model

        在真实系统中，这里应使用训练好的 PRM。
        此处基于启发式规则进行评分。

        评分维度：
        1. 步骤明确性：是否有清晰的分步标记
        2. 逻辑连贯性：是否使用逻辑连接词
        3. 数值计算：是否包含具体数字和运算
        4. 结论明确：是否有明确的最终答案

        Args:
            reasoning: 推理过程文本

        Returns:
            0-1 之间的评分
        """
        score = 0.0
        max_score = 4.0

        # 维度 1：步骤明确性（是否有 "第一步"、"第二步" 等标记）
        step_markers = re.findall(r'第[一二三四五六七八九十\d]+步', reasoning)
        if len(step_markers) >= 2:
            score += 1.0
        elif len(step_markers) >= 1:
            score += 0.5

        # 维度 2：逻辑连贯性（逻辑连接词的使用）
        logic_words = ['因为', '所以', '因此', '由此', '首先', '然后', '最后', '综上']
        logic_count = sum(1 for w in logic_words if w in reasoning)
        score += min(logic_count / 3, 1.0)

        # 维度 3：包含具体数字和运算
        has_numbers = bool(re.findall(r'\d+', reasoning))
        has_operations = bool(re.findall(r'[+\-×÷=]', reasoning))
        if has_numbers:
            score += 0.5
        if has_operations:
            score += 0.5

        # 维度 4：有明确的最终答案
        if any(kw in reasoning for kw in ['答案是', '结果是', '所以', '最终']):
            score += 1.0

        return score / max_score

    def select_best(self, candidates: list[str]) -> tuple[str, float, list[float]]:
        """
        从候选回答中选择最优 — 百炼择优

        Args:
            candidates: 候选推理过程列表

        Returns:
            (最优推理, 最优分数, 所有分数)
        """
        scores = [self.score_reasoning(c) for c in candidates]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best_idx], scores[best_idx], scores

    def demonstrate(self) -> None:
        """演示 Best-of-N 选择过程"""
        print("\n" + "=" * 60)
        print("  Best-of-N 百炼择优 — 演示")
        print("=" * 60)

        # 模拟几个质量不同的推理过程
        candidates = [
            "3 加 4 等于 12。答案是 12。",  # 计算错误
            (
                "第一步：识别数据——丹炉有 3 层，每层 4 份药材。\n"
                "第二步：因为每层放 4 份，所以总共需要 3 × 4 = 12 份。\n"
                "第三步：因此，最终答案是 12 份药材。"
            ),  # 完整清晰
            "3 × 4 = 12",  # 太简洁
            (
                "首先，丹炉有 3 层。然后，每层需要 4 份药材。\n"
                "所以总数 = 3 × 4 = 12。答案是 12 份。"
            ),  # 较好但不够完整
        ]

        print(f"\n候选推理数量：{len(candidates)}\n")
        best, best_score, all_scores = self.select_best(candidates)

        for i, (c, s) in enumerate(zip(candidates, all_scores)):
            marker = " ★ 最优" if i == all_scores.index(best_score) else ""
            print(f"  候选 {i+1} (评分: {s:.2f}){marker}:")
            for line in c.split('\n'):
                print(f"    {line}")
            print()

        print(f"  最优推理评分：{best_score:.2f}")
        print(f"  评分分布：{[f'{s:.2f}' for s in all_scores]}")


# ============================================================
# 第六节：主程序 — 斗帝试炼场
# ============================================================

def main():
    """
    主入口 — 斗帝试炼场

    依次演示：
    1. ReAct Agent 自主推理
    2. Chain-of-Thought 提示构建
    3. Self-Consistency 多路径投票
    4. Best-of-N 择优选择
    """
    print("╔" + "═" * 58 + "╗")
    print("║" + "《焚诀》第十卷 · 帝境篇 — 斗帝试炼场".center(38) + "║")
    print("╚" + "═" * 58 + "╝")

    # --- 试炼一：ReAct Agent ---
    print("\n\n" + "▓" * 60)
    print("  试炼一：ReAct Agent — 斗帝的自主推理")
    print("▓" * 60)

    agent = ReActAgent(max_steps=5, verbose=True)

    # 问题 1：知识查询
    agent.execute("什么是 Transformer 架构？")

    # 问题 2：需要搜索 + 计算
    agent.execute("2024年诺贝尔物理学奖得主的年龄总和是多少？")

    # --- 试炼二：Chain-of-Thought ---
    print("\n\n" + "▓" * 60)
    print("  试炼二：Chain-of-Thought — 推理引导术")
    print("▓" * 60)

    cot = ChainOfThoughtPrompter()
    cot.demonstrate("一位炼丹师有 15 株千年灵芝，他用了 7 株炼制丹药，"
                    "然后弟子又送来了 4 株。请问炼丹师现在有几株千年灵芝？")

    # --- 试炼三：Self-Consistency ---
    print("\n\n" + "▓" * 60)
    print("  试炼三：Self-Consistency — 万人共识之道")
    print("▓" * 60)

    random.seed(42)  # 固定随机种子以便复现
    voter = SelfConsistencyVoter(n_paths=15)
    voter.demonstrate("一个丹炉有 3 层，每层放 4 份药材，总共需要多少份药材？")

    # --- 试炼四：Best-of-N ---
    print("\n\n" + "▓" * 60)
    print("  试炼四：Best-of-N — 百炼择优")
    print("▓" * 60)

    selector = BestOfNSelector()
    selector.demonstrate()

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("  斗帝试炼完毕")
    print("=" * 60)
    print("""
  本丹方演示了斗帝境界的核心斗技：

  1. ReAct Agent：交替思考与行动，自主调用工具解决问题
     → 这是 AI Agent 的基础框架

  2. Chain-of-Thought：引导模型展示推理过程
     → Zero-shot ("让我们一步一步思考") 和 Few-shot (提供示例)

  3. Self-Consistency：采样多条推理路径，投票选最佳答案
     → 用计算量换取准确率的核心思路

  4. Best-of-N：生成多个候选，用评分函数择优
     → Process Reward Model 的简化版本

  在真实的斗帝修炼中，这些斗技会结合 LLM API 使用，
  产生远超本演示的强大效果。

  修炼建议：
  - 将本脚本中的模拟函数替换为真实的 LLM API 调用
  - 在 Self-Consistency 中尝试不同的 temperature 和采样数
  - 实现真正的 PRM 评分函数替代简单的启发式规则
  - 尝试 MCTS + PRM 的组合，实现更强大的推理搜索
  - 构建多 Agent 协作系统，让多个 Agent 分工合作
  """)


if __name__ == "__main__":
    main()