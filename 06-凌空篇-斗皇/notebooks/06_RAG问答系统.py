"""
╔══════════════════════════════════════════════════════════════════════╗
║                  《焚诀》第六卷 · 凌空篇 · 实战丹方                    ║
║                                                                      ║
║              万物追溯之术 — RAG 检索增强生成问答系统                     ║
║                                                                      ║
║  "即便是斗皇级强者，也不可能记住这世间所有的功法典籍。                    ║
║   而万物追溯之术，使修炼者能在需要时瞬间翻阅万卷藏书，                   ║
║   引经据典，对答如流。"                                                ║
║                                                                      ║
║  本丹方构建一套完整的 RAG 系统:                                        ║
║  1. 文档加载与分块（药材切割）                                          ║
║  2. Embedding 向量化（药性提取）                                       ║
║  3. FAISS 向量存储（藏经阁建设）                                       ║
║  4. 混合检索 + Re-ranking（精准寻宝）                                  ║
║  5. LLM 生成回答（引经据典）                                           ║
║                                                                      ║
║  依赖安装:                                                            ║
║  pip install sentence-transformers faiss-cpu rank-bm25 jieba         ║
║  pip install transformers torch numpy                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

# ============================================================
# 日志配置 — 丹房炼制记录
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAG-万物追溯")


# ============================================================
# 第一层：文档加载与分块 — 药材采集与切割
# "百斤矿石，淘洗之后得精华一两，方可入炉。"
# ============================================================

@dataclass
class DocumentChunk:
    """
    文档块：药材切片的最小单位。
    每一个 chunk 携带原始文本和来源信息，
    如同每一份药材切片都贴有产地标签。
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            # 以内容哈希作为唯一标识，防止重复入库
            self.chunk_id = hashlib.md5(
                self.text.encode("utf-8")
            ).hexdigest()[:12]


class DocumentLoader:
    """
    文档加载器：从不同来源采集药材（原始文本）。
    支持纯文本文件和目录批量加载。
    如同炼丹师在山间采药，逐一验收入库。
    """

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """加载纯文本文件。"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def load_directory(
        dir_path: str,
        extensions: list[str] = None,
    ) -> list[dict]:
        """
        批量加载目录下所有文件。
        如同包下一座药山，逐一采集所有药材。
        """
        if extensions is None:
            extensions = [".txt", ".md"]

        documents = []
        dir_path = Path(dir_path)

        if not dir_path.exists():
            logger.warning(f"[采药失败] 目录不存在: {dir_path}")
            return documents

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                try:
                    text = DocumentLoader.load_text_file(str(file_path))
                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": file_path.name,
                            "path": str(file_path),
                        })
                        logger.info(f"[采药成功] {file_path.name} ({len(text)} 字)")
                except Exception as e:
                    logger.error(f"[采药失败] {file_path.name}: {e}")

        logger.info(f"[采药汇总] 共采集 {len(documents)} 份药材")
        return documents


class TextChunker:
    """
    文档分块器：将长文本切割为适合检索的小块。

    分块策略：递归字符分块 —— 优先按段落、句子、词语的顺序切割。
    如同按药材的天然节理切割，保持每段的完整性。

    参数说明:
    - chunk_size: 每个 chunk 的最大字符数（药材切片的厚度）
    - chunk_overlap: 相邻 chunk 的重叠字符数（切口的衔接部分）
      重叠是为了防止关键信息刚好落在切割边界上。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 中文优化的分隔符优先级：段落 → 换行 → 句号 → 其他标点 → 空格
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", " "]

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """按指定分隔符切割文本，保留分隔符。"""
        if separator == "":
            return list(text)  # 逐字符切割（最后的手段）

        parts = text.split(separator)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            else:
                if part:
                    result.append(part)
        return result

    def _merge_chunks(self, pieces: list[str]) -> list[str]:
        """
        将小片段合并为不超过 chunk_size 的块。
        如同将切好的药材片按分量装入药袋。
        """
        chunks = []
        current_chunk = ""

        for piece in pieces:
            # 如果当前块加上这个片段不超限，就合并
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果单个片段就超长，直接作为一个块（不再细分）
                if len(piece) > self.chunk_size:
                    chunks.append(piece[:self.chunk_size].strip())
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        为相邻 chunk 添加重叠部分。
        如同药材切片之间留有衔接，确保连续信息不断裂。
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            # 取前一个 chunk 的末尾作为当前 chunk 的开头
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append(prev_tail + chunks[i])

        return overlapped

    def split(self, text: str) -> list[str]:
        """
        递归分块主方法。
        尝试从最高优先级的分隔符开始切割，
        如果切出的块仍然太大，递归使用下一级分隔符。
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # 尝试每个分隔符
        for separator in self.separators:
            pieces = self._split_by_separator(text, separator)
            if len(pieces) > 1:
                # 用这个分隔符能切开，合并为合适大小的块
                chunks = self._merge_chunks(pieces)
                # 添加重叠
                chunks = self._add_overlap(chunks)
                return [c for c in chunks if c.strip()]

        # 所有分隔符都无法切割（极端情况），强制按字符数切割
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def chunk_document(
        self,
        text: str,
        metadata: dict = None,
    ) -> list[DocumentChunk]:
        """
        将一份完整文档切分为 DocumentChunk 列表。
        如同将一整株药材按标准规格切片、编号。
        """
        if metadata is None:
            metadata = {}

        raw_chunks = self.split(text)
        doc_chunks = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            }
            doc_chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=chunk_meta,
            ))

        return doc_chunks


# ============================================================
# 第二层：Embedding 向量化 — 药性提取
# "将文字化为数字向量，如同将药材的药性提取为可度量的数值。"
# ============================================================

class EmbeddingModel:
    """
    Embedding 模型封装：将文本转化为稠密向量。

    向量化是 RAG 的基础——只有将文本映射到向量空间，
    才能用数学方法衡量文本之间的语义距离。

    如同炼丹师以灵识感知药材的药性，
    Embedding 模型以神经网络感知文本的语义。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化 Embedding 模型。
        默认使用 bge-small-zh —— 体积小、速度快，适合学习和原型开发。
        生产环境推荐 bge-large-zh-v1.5 或 bge-m3。
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[向量化] 正在加载 Embedding 模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"[向量化] 模型加载完成，向量维度: {self.dimension}")
        except ImportError:
            logger.error("[反噬] 缺少 sentence-transformers！请执行: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"[反噬] 模型加载失败: {e}")
            raise

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        将文本列表批量转化为向量。

        参数:
        - texts: 待向量化的文本列表
        - batch_size: 批处理大小（影响显存使用和速度）
        - normalize: 是否归一化（归一化后内积 = 余弦相似度）

        返回: shape=(len(texts), dimension) 的 numpy 数组
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """向量化单条文本。"""
        return self.encode([text], normalize=normalize)[0]


# ============================================================
# 第三层：向量存储 — 藏经阁建设
# "轻便快捷的私人藏书架，以 FAISS 为基石。"
# ============================================================

class FAISSVectorStore:
    """
    基于 FAISS 的向量存储与检索。

    FAISS (Facebook AI Similarity Search) 是目前最高效的
    向量相似度搜索库之一。我们使用 IndexFlatIP（内积索引）
    配合归一化向量，等价于余弦相似度搜索。

    如同修炼者在洞府中建立的私人藏经阁，
    每一卷典籍都以独特的灵气印记（向量）标注，
    查找时以灵识（查询向量）扫描，瞬间定位。
    """

    def __init__(self, dimension: int):
        """
        初始化 FAISS 索引。
        dimension: 向量维度，必须与 Embedding 模型的输出维度一致。
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.error("[反噬] 缺少 faiss！请执行: pip install faiss-cpu")
            raise

        self.dimension = dimension
        # IndexFlatIP: 精确内积搜索（归一化后等价于余弦相似度）
        # 适合中小规模数据（<100万）
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

        logger.info(f"[藏经阁] FAISS 索引初始化完成，维度: {dimension}")

    @property
    def size(self) -> int:
        """当前存储的向量数量。"""
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ):
        """
        批量添加向量和对应的文档块。
        如同将一批切好的药材逐一归档入库。
        """
        assert len(embeddings) == len(chunks), \
            f"向量数量({len(embeddings)})与文档块数量({len(chunks)})不匹配！"

        # FAISS 要求 float32 且做好归一化
        vectors = np.array(embeddings, dtype=np.float32)
        self.faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        logger.info(f"[入库] 新增 {len(chunks)} 个文档块，总计: {self.size}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        向量相似度搜索：给定查询向量，返回最相似的 Top-K 文档块。

        如同以灵识在藏经阁中搜索，
        灵气印记最匹配的典籍会首先浮现。
        """
        if self.size == 0:
            logger.warning("[搜索] 藏经阁为空，无法检索！")
            return []

        # 准备查询向量
        query = np.array([query_vector], dtype=np.float32)
        self.faiss.normalize_L2(query)

        # FAISS 搜索
        actual_k = min(top_k, self.size)
        scores, indices = self.index.search(query, actual_k)

        # 组装结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": float(score),
                })

        return results

    def save(self, dir_path: str):
        """
        持久化存储：将索引和文档数据保存到磁盘。
        如同将藏经阁的索引目录刻录在玉简中，便于日后恢复。
        """
        os.makedirs(dir_path, exist_ok=True)

        # 保存 FAISS 索引
        index_path = os.path.join(dir_path, "index.faiss")
        self.faiss.write_index(self.index, index_path)

        # 保存文档数据
        docs_data = []
        for chunk in self.chunks:
            docs_data.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
            })
        docs_path = os.path.join(dir_path, "documents.json")
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[存储] 藏经阁已保存到: {dir_path}")

    def load(self, dir_path: str):
        """从磁盘恢复索引和文档数据。"""
        index_path = os.path.join(dir_path, "index.faiss")
        docs_path = os.path.join(dir_path, "documents.json")

        self.index = self.faiss.read_index(index_path)

        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        self.chunks = []
        for doc in docs_data:
            self.chunks.append(DocumentChunk(
                text=doc["text"],
                metadata=doc["metadata"],
                chunk_id=doc["chunk_id"],
            ))

        logger.info(f"[恢复] 藏经阁已从 {dir_path} 恢复，共 {self.size} 条记录")


# ============================================================
# 第四层：BM25 关键词检索 — 肉眼搜索
# "向量检索以灵识感知语义，BM25 以肉眼匹配关键词。"
# ============================================================

class BM25Retriever:
    """
    BM25 关键词检索器。

    BM25 是传统信息检索中最经典的算法，
    擅长精确匹配关键词和专有名词。

    与向量检索的区别:
    - 向量检索: "机器学习" 能匹配 "AI" "人工智能"（语义相似）
    - BM25: "机器学习" 只能匹配包含 "机器" 或 "学习" 的文档（词汇匹配）

    两者互补，如同以灵识和肉眼双重搜索。
    """

    def __init__(self, chunks: list[DocumentChunk]):
        """
        初始化 BM25 索引。
        需要对中文文本先进行分词（中文没有天然的空格分隔）。
        """
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            self.jieba = jieba
        except ImportError:
            logger.error("[反噬] 缺少依赖！请执行: pip install rank-bm25 jieba")
            raise

        self.chunks = chunks

        # 对所有文档进行分词
        logger.info("[BM25] 正在构建关键词索引（分词中）...")
        tokenized_docs = []
        for chunk in chunks:
            tokens = list(jieba.cut(chunk.text))
            tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"[BM25] 关键词索引构建完成，共 {len(chunks)} 个文档块")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 关键词搜索。
        """
        # 对查询也进行分词
        tokenized_query = list(self.jieba.cut(query))

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有匹配的结果
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "bm25_score": float(scores[idx]),
                })

        return results


# ============================================================
# 第五层：混合检索 + Re-ranking — 精准寻宝
# "粗筛以灵识和肉眼双重搜索，精排以交叉编码器逐一鉴定。"
# ============================================================

class HybridRetriever:
    """
    混合检索器：融合向量检索（Dense）与 BM25 检索（Sparse）。

    原理：
    1. 向量检索擅长语义匹配（理解同义词、上下义词）
    2. BM25 擅长精确匹配（专有名词、缩写、代码关键词）
    3. 用 alpha 参数控制两者的权重比例
    4. 最终用 Re-ranker（交叉编码器）精排 Top-K

    如同先以灵识扫描大范围，再以肉眼细查，
    最后请大师逐一鉴定，确保万无一失。
    """

    def __init__(
        self,
        embed_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        reranker_model: str = None,
    ):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.reranker = None

        # 加载 Re-ranker（可选）
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"[精排器] 正在加载 Re-ranker: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
                logger.info("[精排器] Re-ranker 加载完成")
            except Exception as e:
                logger.warning(f"[精排器] Re-ranker 加载失败，将跳过精排: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        alpha: float = 0.6,
        use_reranking: bool = True,
    ) -> list[dict]:
        """
        完整的混合检索流程:
        1. 向量检索 Top-initial_k
        2. BM25 检索 Top-initial_k
        3. 分数融合（RRF 或加权融合）
        4. Re-ranking 精排 Top-K

        参数:
        - query: 用户查询
        - top_k: 最终返回的文档数
        - initial_k: 初始检索的候选文档数
        - alpha: 向量检索权重 (0~1)，1-alpha 为 BM25 权重
        - use_reranking: 是否使用 Re-ranker 精排
        """
        logger.info(f"[检索] 查询: '{query[:50]}...' (alpha={alpha}, top_k={top_k})")

        # === 第一阶段：双路检索（粗筛）===

        # 1a. 向量检索
        query_embedding = self.embed_model.encode_single(query)
        dense_results = self.vector_store.search(query_embedding, top_k=initial_k)

        # 1b. BM25 检索
        sparse_results = self.bm25.search(query, top_k=initial_k)

        # === 第二阶段：分数融合 ===

        # 用 chunk_id 作为 key 合并两路结果
        score_map: dict[str, dict] = {}

        # 归一化向量检索分数到 [0, 1]
        if dense_results:
            dense_scores = [r["score"] for r in dense_results]
            d_min, d_max = min(dense_scores), max(dense_scores)
            d_range = d_max - d_min if d_max > d_min else 1.0

            for r in dense_results:
                cid = r["chunk_id"]
                norm_score = (r["score"] - d_min) / d_range
                score_map[cid] = {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "dense_score": norm_score,
                    "sparse_score": 0.0,
                }

        # 归一化 BM25 分数到 [0, 1]
        if sparse_results:
            sparse_scores = [r["bm25_score"] for r in sparse_results]
            s_min, s_max = min(sparse_scores), max(sparse_scores)
            s_range = s_max - s_min if s_max > s_min else 1.0

            for r in sparse_results:
                cid = r["chunk_id"]
                norm_score = (r["bm25_score"] - s_min) / s_range
                if cid in score_map:
                    score_map[cid]["sparse_score"] = norm_score
                else:
                    score_map[cid] = {
                        "text": r["text"],
                        "metadata": r["metadata"],
                        "dense_score": 0.0,
                        "sparse_score": norm_score,
                    }

        # 计算加权融合分数
        fused_results = []
        for cid, data in score_map.items():
            hybrid_score = alpha * data["dense_score"] + (1 - alpha) * data["sparse_score"]
            fused_results.append({
                "chunk_id": cid,
                "text": data["text"],
                "metadata": data["metadata"],
                "hybrid_score": hybrid_score,
                "dense_score": data["dense_score"],
                "sparse_score": data["sparse_score"],
            })

        # 按融合分数排序
        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 取候选集用于精排
        candidates = fused_results[:initial_k]

        logger.info(
            f"[检索] 向量命中: {len(dense_results)}, "
            f"BM25命中: {len(sparse_results)}, "
            f"融合后: {len(candidates)} 个候选"
        )

        # === 第三阶段：Re-ranking 精排 ===

        if use_reranking and self.reranker and candidates:
            logger.info("[精排] 正在用 Cross-Encoder 重排序...")

            # 构建 (query, document) 对
            pairs = [(query, c["text"]) for c in candidates]

            # Cross-Encoder 打分
            rerank_scores = self.reranker.predict(pairs)

            # 更新分数并重排
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info("[精排] 重排序完成")

        # 返回 Top-K
        final_results = candidates[:top_k]

        for i, r in enumerate(final_results, 1):
            score_key = "rerank_score" if "rerank_score" in r else "hybrid_score"
            logger.info(
                f"  [{i}] {score_key}={r[score_key]:.4f} | "
                f"{r['text'][:60]}..."
            )

        return final_results


# ============================================================
# 第六层：LLM 生成回答 — 引经据典
# "检索到的典籍不是答案本身，而是回答的依据。"
# ============================================================

# RAG Prompt 模板：引导 LLM 基于检索内容回答
RAG_SYSTEM_PROMPT = """你是一个专业的AI助手。请严格基于以下参考文档回答用户问题。

重要规则：
1. 只使用参考文档中的信息来回答，不要编造不在文档中的内容
2. 如果参考文档中没有相关信息，请明确说"根据现有资料，我无法回答这个问题"
3. 引用信息时，请标注来源，如 [来源: 文档名]
4. 回答要简洁、准确、有条理"""

RAG_USER_PROMPT_TEMPLATE = """参考文档：
{context}

用户问题：{question}

请基于以上参考文档回答用户问题："""


def format_retrieved_context(
    results: list[dict],
    max_total_chars: int = 3000,
) -> str:
    """
    将检索结果格式化为 LLM 可理解的上下文。
    需要控制总长度，防止超出上下文窗口。
    如同从藏经阁取出的典籍需要摘录关键段落，不能把整座阁搬给别人看。
    """
    context_parts = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        text = result["text"]
        source = result.get("metadata", {}).get("source", "未知来源")
        score_key = "rerank_score" if "rerank_score" in result else "hybrid_score"
        score = result.get(score_key, result.get("score", 0))

        # 检查是否超出长度限制
        if total_chars + len(text) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining > 100:
                text = text[:remaining] + "...(截断)"
            else:
                break

        header = f"[文档{i}] (来源: {source}, 相关度: {score:.3f})"
        context_parts.append(f"{header}\n{text}")
        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts)


def generate_answer_local(
    question: str,
    context: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    使用本地模型生成回答（无需 API）。
    适合在自己的丹炉（服务器）上运行。

    注意: 需要足够的 GPU 显存来加载模型。
    1.5B 模型约需 3GB 显存，7B 模型约需 14GB 显存。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("[反噬] 缺少 transformers/torch！")
        raise

    logger.info(f"[生成] 正在加载本地模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 构建消息
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )},
    ]

    # 生成
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # 解码，只取新生成的部分
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    return answer.strip()


def generate_answer_mock(question: str, context: str) -> str:
    """
    模拟生成回答（用于无 GPU 环境的演示）。
    实际项目中应替换为真实的 LLM 调用。
    """
    # 从上下文中提取关键句子作为"回答"
    sentences = re.split(r'[。！？\n]', context)
    relevant = [s.strip() for s in sentences if len(s.strip()) > 20][:3]

    if relevant:
        answer = f"根据检索到的相关文档，针对您的问题「{question}」，以下是相关信息：\n\n"
        for i, sent in enumerate(relevant, 1):
            answer += f"{i}. {sent}。\n"
        answer += "\n（注：此为演示模式，实际部署时将使用 LLM 生成更完整的回答）"
    else:
        answer = "根据现有资料，我无法找到与您问题直接相关的信息。"

    return answer


# ============================================================
# 顶层：RAG 系统总成 — 万物追溯完整丹方
# "药材采集、切割入库、灵识检索、引经据典——一气呵成。"
# ============================================================

class RAGSystem:
    """
    完整的 RAG 问答系统。

    整合文档加载、分块、向量化、存储、检索、生成的完整流水线。
    如同一座功能完备的工业丹房，从药材入库到丹药出炉，全流程自动化。
    """

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-small-zh-v1.5",
        reranker_model_name: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        logger.info("=" * 60)
        logger.info("  万物追溯 RAG 系统 — 初始化中")
        logger.info("=" * 60)

        # 初始化各组件
        self.embed_model = EmbeddingModel(embed_model_name)
        self.vector_store = FAISSVectorStore(self.embed_model.dimension)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.bm25 = None  # 在文档入库后初始化
        self.retriever = None  # 在文档入库后初始化
        self.reranker_model_name = reranker_model_name
        self.all_chunks: list[DocumentChunk] = []

        logger.info("[系统] RAG 系统初始化完成，等待药材入库...")

    def ingest_texts(self, texts: list[dict]):
        """
        批量摄入文本数据。

        参数:
        - texts: [{"text": "...", "source": "文件名", ...}, ...]

        如同一次性采集大量药材入库。
        """
        logger.info(f"[摄入] 开始处理 {len(texts)} 份文档...")

        all_new_chunks = []

        for doc in texts:
            text = doc["text"]
            metadata = {k: v for k, v in doc.items() if k != "text"}

            # 分块
            chunks = self.chunker.chunk_document(text, metadata)
            all_new_chunks.extend(chunks)

        if not all_new_chunks:
            logger.warning("[摄入] 没有生成任何文档块！")
            return

        # 向量化
        logger.info(f"[向量化] 正在对 {len(all_new_chunks)} 个块进行 Embedding...")
        texts_to_embed = [c.text for c in all_new_chunks]
        embeddings = self.embed_model.encode(texts_to_embed, show_progress=True)

        # 存入向量库
        self.vector_store.add(embeddings, all_new_chunks)
        self.all_chunks.extend(all_new_chunks)

        # 重建 BM25 索引（需要包含所有已入库的文档）
        logger.info("[BM25] 重建关键词索引...")
        self.bm25 = BM25Retriever(self.all_chunks)

        # 重建混合检索器
        self.retriever = HybridRetriever(
            embed_model=self.embed_model,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25,
            reranker_model=self.reranker_model_name,
        )

        logger.info(f"[摄入完成] 总计 {len(self.all_chunks)} 个文档块已入库")

    def ingest_directory(self, dir_path: str, extensions: list[str] = None):
        """从目录批量摄入文档。"""
        documents = DocumentLoader.load_directory(dir_path, extensions)
        if documents:
            self.ingest_texts(documents)

    def query(
        self,
        question: str,
        top_k: int = 5,
        alpha: float = 0.6,
        use_reranking: bool = True,
        use_local_llm: bool = False,
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> dict:
        """
        问答接口：输入问题，返回 RAG 增强的回答。

        这是整个系统的核心入口，如同丹房的出丹口——
        投入问题（药材），经过检索（炼制），产出回答（丹药）。
        """
        if self.retriever is None:
            return {
                "answer": "系统尚未加载任何文档，请先摄入文档。",
                "sources": [],
                "question": question,
            }

        logger.info("=" * 60)
        logger.info(f"[问答] 问题: {question}")
        logger.info("=" * 60)

        # 1. 检索相关文档
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            alpha=alpha,
            use_reranking=use_reranking,
        )

        if not retrieved:
            return {
                "answer": "未找到相关文档，无法回答此问题。",
                "sources": [],
                "question": question,
            }

        # 2. 格式化上下文
        context = format_retrieved_context(retrieved)

        # 3. 生成回答
        if use_local_llm:
            answer = generate_answer_local(question, context, llm_model)
        else:
            answer = generate_answer_mock(question, context)

        # 4. 组装返回结果
        sources = []
        for r in retrieved:
            sources.append({
                "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                "source": r.get("metadata", {}).get("source", "unknown"),
                "score": r.get("rerank_score", r.get("hybrid_score", 0)),
            })

        result = {
            "answer": answer,
            "sources": sources,
            "question": question,
            "num_retrieved": len(retrieved),
        }

        logger.info(f"[回答] {answer[:200]}...")
        return result


# ============================================================
# 演示入口 — 丹方试炼
# ============================================================

def create_demo_documents() -> list[dict]:
    """
    创建演示用的文档数据。
    如同准备一批示范药材，用于测试丹方的正确性。

    这些文档涵盖了 AI/ML 的基础知识，可以用来测试 RAG 系统的检索和回答能力。
    """
    documents = [
        {
            "text": (
                "Transformer 架构由 Vaswani 等人在 2017 年的论文"
                "《Attention Is All You Need》中提出。它完全基于自注意力机制"
                "（Self-Attention），摒弃了传统的循环神经网络（RNN）和卷积神经网络"
                "（CNN）结构。Transformer 的核心创新在于多头注意力机制（Multi-Head "
                "Attention），它允许模型同时关注输入序列的不同位置和不同的表示子空间。"
                "这种架构极大地提高了并行计算能力，使得训练大规模语言模型成为可能。"
            ),
            "source": "transformer_basics.txt",
        },
        {
            "text": (
                "大语言模型（Large Language Model, LLM）是基于 Transformer 架构"
                "的超大规模神经网络模型。典型代表包括 GPT 系列（OpenAI）、LLaMA 系列"
                "（Meta）、Qwen 系列（阿里巴巴）等。这些模型通常包含数十亿到数千亿个"
                "参数，通过在海量文本数据上进行预训练来学习语言的统计规律和世界知识。"
                "预训练完成后，通过监督微调（SFT）和人类反馈强化学习（RLHF）等技术，"
                "使模型能够按照人类指令进行对话和任务执行。"
            ),
            "source": "llm_overview.txt",
        },
        {
            "text": (
                "检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合信息"
                "检索和文本生成的技术框架。RAG 的核心思想是：在大语言模型生成回答之前，"
                "先从外部知识库中检索与用户问题相关的文档片段，然后将这些检索到的内容"
                "作为上下文提供给模型。这种方法能有效缓解模型的幻觉问题（Hallucination），"
                "同时解决知识截止（Knowledge Cutoff）的限制。RAG 系统的典型架构包括三个"
                "阶段：检索（Retrieve）、增强（Augment）和生成（Generate）。"
            ),
            "source": "rag_introduction.txt",
        },
        {
            "text": (
                "向量数据库是专门用于存储和检索高维向量的数据库系统。在 RAG 系统中，"
                "文本首先通过 Embedding 模型转换为稠密向量，然后存储在向量数据库中。"
                "查询时，用户的问题也被转换为向量，通过相似度搜索（如余弦相似度或内积）"
                "找到最相关的文档。常见的向量数据库包括 FAISS（Facebook）、Milvus、"
                "Qdrant、Chroma、Weaviate 和 Pinecone 等。其中 FAISS 是最轻量级的"
                "选择，适合中小规模场景；Milvus 支持分布式部署，适合大规模生产环境。"
            ),
            "source": "vector_database.txt",
        },
        {
            "text": (
                "监督微调（Supervised Fine-Tuning, SFT）是将预训练语言模型转化为"
                "能够按指令对话的关键步骤。SFT 使用指令-回答对（instruction-response "
                "pairs）作为训练数据，教会模型理解用户意图并给出格式化的回答。"
                "SFT 的数据格式通常采用 system/user/assistant 三角色结构，"
                "训练时只在 assistant 的回复上计算损失（Loss Masking），"
                "避免模型学习如何提问而非如何回答。关键超参数包括：学习率通常设为 "
                "1e-5 到 5e-5，训练 2-3 个 epoch，warmup 比例约 3%。"
            ),
            "source": "sft_guide.txt",
        },
        {
            "text": (
                "数据清洗是机器学习项目中最重要但最容易被忽视的环节。在工业界，"
                "有一个广泛共识：数据工作占整个项目工作量的 80% 以上。高质量的训练数据"
                "远比大量低质量数据更有价值。数据清洗的典型流程包括：去重（使用 MinHash "
                "或 SimHash 进行模糊去重）、语言检测（fastText 语言识别）、质量过滤"
                "（基于困惑度或规则的过滤）、隐私脱敏（去除手机号、身份证号等 PII 信息）、"
                "以及毒性过滤（去除有害、暴力、歧视性内容）。"
            ),
            "source": "data_cleaning.txt",
        },
        {
            "text": (
                "LoRA（Low-Rank Adaptation）是一种参数高效微调（PEFT）方法，"
                "由 Hu 等人在 2021 年提出。LoRA 的核心思想是：在模型的权重矩阵旁边"
                "添加两个低秩矩阵 A 和 B（秩 r 远小于原矩阵维度），训练时只更新这两个"
                "小矩阵，冻结原始模型权重。这使得可训练参数量大幅减少（通常只有全量微调"
                "的 0.1%-1%），显存需求也相应降低。QLoRA 在此基础上进一步引入 4-bit 量化，"
                "使得在单张消费级 GPU（如 RTX 4090 24GB）上就能微调 7B 甚至 13B 的模型。"
            ),
            "source": "lora_peft.txt",
        },
        {
            "text": (
                "混合精度训练（Mixed Precision Training）是一种同时使用 FP16/BF16 "
                "和 FP32 数据类型的训练技术。核心思想是：前向传播和反向传播使用低精度"
                "（FP16/BF16）以加速计算和减少显存，而参数更新和梯度累积使用 FP32 "
                "以保持数值稳定性。BF16（Brain Floating Point 16）相比 FP16 拥有"
                "更大的指数范围，能更好地避免梯度溢出问题，是目前大模型训练的首选精度。"
                "配合梯度缩放（Gradient Scaling），混合精度训练可以将显存使用减半，"
                "训练速度提升 2-3 倍。"
            ),
            "source": "mixed_precision.txt",
        },
    ]

    return documents


def run_demo():
    """
    运行 RAG 系统演示。

    这是整个丹方的试炼流程：
    1. 初始化系统（点燃丹炉）
    2. 摄入文档（投入药材）
    3. 提出问题（开始炼制）
    4. 获取回答（丹药出炉）
    """
    print("\n" + "=" * 70)
    print("  《焚诀》第六卷 · 凌空篇 · 万物追溯 RAG 系统 · 丹方试炼")
    print("=" * 70 + "\n")

    # ---- 第一步：初始化系统 ----
    # 注意: 首次运行会下载 Embedding 模型（约 100MB）
    # 如果网络不通，可以提前下载模型到本地
    print("[第一步] 点燃丹炉 — 初始化 RAG 系统...\n")

    rag = RAGSystem(
        embed_model_name="BAAI/bge-small-zh-v1.5",  # 小模型，适合演示
        reranker_model_name=None,  # 不使用 Re-ranker（减少依赖）
        chunk_size=300,
        chunk_overlap=30,
    )

    # ---- 第二步：摄入文档 ----
    print("\n[第二步] 投入药材 — 摄入文档...\n")

    demo_docs = create_demo_documents()
    rag.ingest_texts(demo_docs)

    # ---- 第三步：提问与回答 ----
    print("\n[第三步] 开始炼制 — 问答测试...\n")

    test_questions = [
        "什么是 RAG？它解决了什么问题？",
        "如何进行监督微调（SFT）？关键超参数有哪些？",
        "LoRA 和 QLoRA 有什么区别？",
        "数据清洗包括哪些步骤？",
        "有哪些常见的向量数据库？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"  问题 {i}: {question}")
        print(f"{'─' * 60}\n")

        result = rag.query(
            question=question,
            top_k=3,
            alpha=0.6,
            use_reranking=False,  # 演示模式不使用 Re-ranker
        )

        print(f"  回答:\n")
        # 按行缩进输出
        for line in result["answer"].split("\n"):
            print(f"    {line}")

        print(f"\n  参考来源:")
        for j, src in enumerate(result["sources"], 1):
            print(f"    [{j}] {src['source']} (相关度: {src['score']:.3f})")

    # ---- 完成 ----
    print(f"\n{'=' * 70}")
    print("  丹方试炼完成！万物追溯之术，引经据典，对答如流。")
    print(f"{'=' * 70}\n")

    return rag  # 返回系统实例，可继续交互


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 运行演示
    rag_system = run_demo()

    # 交互式问答（可选）
    print("\n进入交互模式（输入 'quit' 退出）:\n")
    while True:
        try:
            question = input("你的问题> ").strip()
            if question.lower() in ("quit", "exit", "q", "退出"):
                print("\n凌空篇修炼结束，后会有期！")
                break
            if not question:
                continue

            result = rag_system.query(question, top_k=3)
            print(f"\n回答:\n{result['answer']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n凌空篇修炼结束，后会有期！")
            break
