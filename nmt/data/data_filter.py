"""
高质量数据筛选模块

功能说明：
    基于 LaBSE（Language-agnostic BERT Sentence Embedding）模型进行双语句对
    语义相似度计算，筛选高质量翻译样本。

主要特性：
    - 语义相似度评分：使用 LaBSE 计算中英句对的语义相似度
    - 长度比例过滤：过滤长度比例异常的样本
    - 重复 N-gram 检测：过滤含有过多重复内容的样本
    - 领域相关性评估：（可选）与目标领域词汇分布对齐
    - 批量处理：支持大规模数据集的高效处理

依赖：
    - sentence-transformers: LaBSE 模型加载
    - torch: 张量计算
    - numpy: 数值计算

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import re

import numpy as np
import torch
from tqdm import tqdm

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers 未安装，LaBSE 功能将被禁用")


# ====================================
# 常量定义
# ====================================

# 默认 LaBSE 模型名称
DEFAULT_LABSE_MODEL = "sentence-transformers/LaBSE"

# 语义相似度阈值
DEFAULT_MIN_SIMILARITY = 0.7

# 长度比例范围（中文字符数 / 英文字符数）
DEFAULT_MIN_RATIO = 0.25  # 1:4
DEFAULT_MAX_RATIO = 0.5   # 1:2

# N-gram 重复阈值（重复率超过此值视为低质量）
DEFAULT_MAX_NGRAM_REPEAT_RATIO = 0.3

# 批处理大小（根据显存调整）
DEFAULT_BATCH_SIZE = 64


@dataclass
class FilterStats:
    """
    筛选统计信息数据类
    
    属性：
        total_samples: 输入样本总数
        similarity_filtered: 相似度过滤数量
        ratio_filtered: 长度比例过滤数量
        ngram_filtered: N-gram 重复过滤数量
        domain_filtered: 领域相关性过滤数量
        final_samples: 最终保留数量
    """
    total_samples: int = 0
    similarity_filtered: int = 0
    ratio_filtered: int = 0
    ngram_filtered: int = 0
    domain_filtered: int = 0
    final_samples: int = 0
    
    # 相似度分布统计
    similarity_scores: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "输入样本数": self.total_samples,
            "相似度过滤": self.similarity_filtered,
            "比例过滤": self.ratio_filtered,
            "N-gram重复过滤": self.ngram_filtered,
            "领域过滤": self.domain_filtered,
            "最终保留": self.final_samples,
        }
    
    def __str__(self) -> str:
        """格式化输出统计信息"""
        lines = [
            "=" * 50,
            "数据筛选统计报告",
            "=" * 50,
        ]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value:,}")
        
        # 计算保留率
        if self.total_samples > 0:
            retention_rate = self.final_samples / self.total_samples * 100
            lines.append(f"  保留率: {retention_rate:.2f}%")
        
        # 相似度分布
        if self.similarity_scores:
            scores = np.array(self.similarity_scores)
            lines.append(f"\n  相似度统计:")
            lines.append(f"    平均值: {np.mean(scores):.4f}")
            lines.append(f"    中位数: {np.median(scores):.4f}")
            lines.append(f"    标准差: {np.std(scores):.4f}")
            lines.append(f"    最小值: {np.min(scores):.4f}")
            lines.append(f"    最大值: {np.max(scores):.4f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class FilteredSample:
    """
    筛选后的样本数据类
    
    属性：
        chinese: 中文文本
        english: 英文文本
        similarity_score: 语义相似度分数
        length_ratio: 长度比例
        source: 数据来源（可选）
        difficulty: 难度等级（用于课程学习）
    """
    chinese: str
    english: str
    similarity_score: float
    length_ratio: float
    source: Optional[str] = None
    difficulty: Optional[str] = None  # easy, medium, hard
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "chinese": self.chinese,
            "english": self.english,
            "similarity_score": round(self.similarity_score, 4),
            "length_ratio": round(self.length_ratio, 4),
        }
        if self.source:
            result["source"] = self.source
        if self.difficulty:
            result["difficulty"] = self.difficulty
        return result


class DataFilter:
    """
    高质量数据筛选器
    
    功能说明：
        基于多种指标筛选高质量翻译样本：
        - LaBSE 语义相似度
        - 长度比例合理性
        - N-gram 重复检测
        - 领域相关性（可选）
    
    参数：
        model_name: LaBSE 模型名称或路径
        min_similarity: 最小语义相似度阈值
        min_ratio: 最小长度比例（中/英）
        max_ratio: 最大长度比例
        max_ngram_repeat: 最大 N-gram 重复率
        batch_size: 批处理大小
        device: 计算设备
        logger: 日志记录器
        
    示例：
        >>> filter = DataFilter(min_similarity=0.7)
        >>> filtered_samples = filter.filter_dataset(samples)
        >>> print(filter.stats)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_LABSE_MODEL,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_ratio: float = DEFAULT_MIN_RATIO,
        max_ratio: float = DEFAULT_MAX_RATIO,
        max_ngram_repeat: float = DEFAULT_MAX_NGRAM_REPEAT_RATIO,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据筛选器
        
        参数：
            model_name: LaBSE 模型名称或路径
            min_similarity: 最小语义相似度
            min_ratio: 最小长度比例
            max_ratio: 最大长度比例
            max_ngram_repeat: 最大 N-gram 重复率
            batch_size: 批处理大小
            device: 计算设备 ('cuda' 或 'cpu')
            logger: 日志记录器
        """
        self.model_name = model_name
        self.min_similarity = min_similarity
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_ngram_repeat = max_ngram_repeat
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        # 统计信息
        self.stats = FilterStats()
        
        # 确定计算设备
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 加载 LaBSE 模型
        self.model = None
        self._load_model()
        
        self.logger.info(f"数据筛选器初始化完成")
        self.logger.info(f"  计算设备: {self.device}")
        self.logger.info(f"  最小相似度: {min_similarity}")
        self.logger.info(f"  长度比例范围: {min_ratio} - {max_ratio}")
    
    def _load_model(self) -> None:
        """
        加载 LaBSE 模型
        
        LaBSE（Language-agnostic BERT Sentence Embedding）是一个多语言
        句子编码模型，支持 109 种语言，能够将不同语言的相似句子映射到
        相近的向量空间位置。
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence-transformers 未安装，跳过模型加载")
            return
        
        try:
            self.logger.info(f"正在加载 LaBSE 模型: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self.logger.info("LaBSE 模型加载完成")
        except Exception as e:
            self.logger.error(f"LaBSE 模型加载失败: {e}")
            self.model = None
    
    def compute_similarity(
        self,
        chinese_texts: List[str],
        english_texts: List[str]
    ) -> np.ndarray:
        """
        批量计算中英句对的语义相似度
        
        使用 LaBSE 模型将中英文本编码为向量，然后计算余弦相似度。
        
        参数：
            chinese_texts: 中文文本列表
            english_texts: 英文文本列表
            
        返回：
            np.ndarray: 相似度分数数组
        """
        if self.model is None:
            # 模型未加载时返回默认值
            self.logger.warning("模型未加载，使用默认相似度 0.8")
            return np.full(len(chinese_texts), 0.8)
        
        # 编码中文文本
        zh_embeddings = self.model.encode(
            chinese_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 归一化
        )
        
        # 编码英文文本
        en_embeddings = self.model.encode(
            english_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 计算余弦相似度（由于已归一化，直接点积即可）
        similarities = np.sum(zh_embeddings * en_embeddings, axis=1)
        
        return similarities
    
    def compute_length_ratio(self, chinese: str, english: str) -> float:
        """
        计算中英文长度比例
        
        中文通常比英文字符数少（中文一个字 ≈ 英文 2-4 个单词），
        合理的比例范围大约在 0.25-0.5 之间。
        
        参数：
            chinese: 中文文本
            english: 英文文本
            
        返回：
            float: 长度比例（中文字符数 / 英文字符数）
        """
        zh_len = len(chinese)
        en_len = len(english)
        
        if en_len == 0:
            return 0.0
        
        return zh_len / en_len
    
    def check_ngram_repeat(
        self,
        text: str,
        n: int = 3
    ) -> float:
        """
        检测文本中的 N-gram 重复率
        
        高重复率可能表示文本质量低（如重复短语、模板文本等）。
        
        参数：
            text: 输入文本
            n: N-gram 的 N 值
            
        返回：
            float: 重复率（重复 N-gram 数 / 总 N-gram 数）
        """
        # 分词（简单按字符或空格分割）
        if self._is_mostly_chinese(text):
            # 中文按字符分割
            tokens = list(text.replace(' ', ''))
        else:
            # 英文按空格分割
            tokens = text.split()
        
        if len(tokens) < n:
            return 0.0
        
        # 生成 N-gram
        ngrams = [
            tuple(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        ]
        
        if not ngrams:
            return 0.0
        
        # 统计重复
        ngram_counts = Counter(ngrams)
        total = len(ngrams)
        unique = len(ngram_counts)
        
        # 重复率 = 1 - 唯一比例
        repeat_ratio = 1 - (unique / total)
        
        return repeat_ratio
    
    def _is_mostly_chinese(self, text: str) -> bool:
        """
        判断文本是否主要是中文
        
        参数：
            text: 输入文本
            
        返回：
            bool: 是否主要是中文
        """
        chinese_chars = sum(
            1 for char in text
            if '\u4e00' <= char <= '\u9fff'
        )
        return chinese_chars > len(text) * 0.3
    
    def classify_difficulty(
        self,
        chinese: str,
        english: str,
        similarity: float
    ) -> str:
        """
        根据样本特征分类难度等级（用于课程学习）
        
        难度分级依据：
        - easy: 短句（<30 tokens）+ 高相似度（≥0.85）
        - medium: 中等长度（30-80 tokens）+ 标准相似度（0.7-0.85）
        - hard: 长句（>80 tokens）或较低相似度
        
        参数：
            chinese: 中文文本
            english: 英文文本
            similarity: 语义相似度
            
        返回：
            str: 难度等级 ('easy', 'medium', 'hard')
        """
        # 估算 token 数量（简化：中文字符数 + 英文单词数）
        zh_tokens = len(chinese)
        en_tokens = len(english.split())
        avg_tokens = (zh_tokens + en_tokens) / 2
        
        # 根据长度和相似度分级
        if avg_tokens < 30 and similarity >= 0.85:
            return "easy"
        elif avg_tokens < 80 and similarity >= 0.7:
            return "medium"
        else:
            return "hard"
    
    def filter_sample(
        self,
        chinese: str,
        english: str,
        precomputed_similarity: Optional[float] = None,
        source: Optional[str] = None
    ) -> Optional[FilteredSample]:
        """
        筛选单个样本
        
        参数：
            chinese: 中文文本
            english: 英文文本
            precomputed_similarity: 预计算的相似度（批量处理时使用）
            source: 数据来源
            
        返回：
            Optional[FilteredSample]: 筛选后的样本，不通过则返回 None
        """
        # 计算长度比例
        length_ratio = self.compute_length_ratio(chinese, english)
        
        # 长度比例过滤
        if length_ratio < self.min_ratio or length_ratio > self.max_ratio:
            self.stats.ratio_filtered += 1
            return None
        
        # N-gram 重复检测（中英都检查）
        zh_repeat = self.check_ngram_repeat(chinese)
        en_repeat = self.check_ngram_repeat(english)
        if zh_repeat > self.max_ngram_repeat or en_repeat > self.max_ngram_repeat:
            self.stats.ngram_filtered += 1
            return None
        
        # 语义相似度（使用预计算值或实时计算）
        if precomputed_similarity is not None:
            similarity = precomputed_similarity
        else:
            similarities = self.compute_similarity([chinese], [english])
            similarity = similarities[0]
        
        # 记录相似度
        self.stats.similarity_scores.append(similarity)
        
        # 相似度过滤
        if similarity < self.min_similarity:
            self.stats.similarity_filtered += 1
            return None
        
        # 分类难度
        difficulty = self.classify_difficulty(chinese, english, similarity)
        
        return FilteredSample(
            chinese=chinese,
            english=english,
            similarity_score=float(similarity),
            length_ratio=length_ratio,
            source=source,
            difficulty=difficulty
        )
    
    def filter_dataset(
        self,
        samples: List[Dict[str, str]],
        zh_key: str = "chinese",
        en_key: str = "english",
        show_progress: bool = True
    ) -> List[FilteredSample]:
        """
        批量筛选数据集
        
        使用批处理方式计算语义相似度以提高效率。
        
        参数：
            samples: 输入样本列表
            zh_key: 中文字段名
            en_key: 英文字段名
            show_progress: 是否显示进度条
            
        返回：
            List[FilteredSample]: 筛选后的样本列表
        """
        # 重置统计
        self.stats = FilterStats()
        self.stats.total_samples = len(samples)
        
        filtered_samples = []
        
        # 分批处理
        num_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(
            range(num_batches),
            desc="数据筛选",
            disable=not show_progress
        ):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(samples))
            batch = samples[start_idx:end_idx]
            
            # 提取文本
            chinese_texts = [s.get(zh_key, "") for s in batch]
            english_texts = [s.get(en_key, "") for s in batch]
            sources = [s.get("source", None) for s in batch]
            
            # 批量计算相似度
            similarities = self.compute_similarity(chinese_texts, english_texts)
            
            # 逐个筛选
            for i, (zh, en, sim, src) in enumerate(
                zip(chinese_texts, english_texts, similarities, sources)
            ):
                result = self.filter_sample(
                    chinese=zh,
                    english=en,
                    precomputed_similarity=sim,
                    source=src
                )
                if result is not None:
                    filtered_samples.append(result)
        
        # 更新统计
        self.stats.final_samples = len(filtered_samples)
        
        self.logger.info(str(self.stats))
        
        return filtered_samples
    
    def filter_dataset_streaming(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        zh_key: str = "chinese",
        en_key: str = "english",
        target_samples: Optional[int] = None
    ) -> FilterStats:
        """
        流式筛选大规模数据集
        
        适用于无法一次性加载到内存的大规模数据集。
        
        参数：
            input_path: 输入文件路径（JSONL 格式）
            output_path: 输出文件路径
            zh_key: 中文字段名
            en_key: 英文字段名
            target_samples: 目标样本数量（达到后停止）
            
        返回：
            FilterStats: 筛选统计信息
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 重置统计
        self.stats = FilterStats()
        
        self.logger.info(f"开始流式筛选: {input_path}")
        
        # 批量缓冲
        batch_buffer: List[Dict] = []
        
        with open(input_path, 'r', encoding='utf-8') as in_file, \
             open(output_path, 'w', encoding='utf-8') as out_file:
            
            for line in tqdm(in_file, desc="流式筛选"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    batch_buffer.append(sample)
                    self.stats.total_samples += 1
                    
                    # 批量处理
                    if len(batch_buffer) >= self.batch_size:
                        self._process_batch(
                            batch_buffer, out_file, zh_key, en_key
                        )
                        batch_buffer.clear()
                    
                    # 检查目标数量
                    if target_samples and self.stats.final_samples >= target_samples:
                        self.logger.info(f"已达到目标样本数: {target_samples}")
                        break
                        
                except json.JSONDecodeError:
                    continue
            
            # 处理剩余样本
            if batch_buffer:
                self._process_batch(batch_buffer, out_file, zh_key, en_key)
        
        self.logger.info(str(self.stats))
        return self.stats
    
    def _process_batch(
        self,
        batch: List[Dict],
        out_file,
        zh_key: str,
        en_key: str
    ) -> None:
        """
        处理一个批次并写入输出
        
        参数：
            batch: 样本批次
            out_file: 输出文件句柄
            zh_key: 中文字段名
            en_key: 英文字段名
        """
        # 提取文本
        chinese_texts = [s.get(zh_key, "") for s in batch]
        english_texts = [s.get(en_key, "") for s in batch]
        sources = [s.get("source", None) for s in batch]
        
        # 批量计算相似度
        similarities = self.compute_similarity(chinese_texts, english_texts)
        
        # 逐个筛选并写入
        for zh, en, sim, src in zip(
            chinese_texts, english_texts, similarities, sources
        ):
            result = self.filter_sample(
                chinese=zh,
                english=en,
                precomputed_similarity=sim,
                source=src
            )
            if result is not None:
                self.stats.final_samples += 1
                out_file.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + '\n'
                )
    
    def get_difficulty_distribution(
        self,
        samples: List[FilteredSample]
    ) -> Dict[str, int]:
        """
        统计样本难度分布
        
        参数：
            samples: 筛选后的样本列表
            
        返回：
            Dict[str, int]: 各难度等级的样本数量
        """
        distribution = {"easy": 0, "medium": 0, "hard": 0}
        for sample in samples:
            if sample.difficulty:
                distribution[sample.difficulty] += 1
        return distribution


def compute_labse_similarity_batch(
    chinese_texts: List[str],
    english_texts: List[str],
    model_path: str = DEFAULT_LABSE_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None
) -> np.ndarray:
    """
    便捷函数：批量计算中英句对的 LaBSE 语义相似度
    
    参数：
        chinese_texts: 中文文本列表
        english_texts: 英文文本列表
        model_path: LaBSE 模型路径
        batch_size: 批处理大小
        device: 计算设备
        
    返回：
        np.ndarray: 相似度分数数组
        
    示例：
        >>> zh = ["你好", "谢谢"]
        >>> en = ["Hello", "Thank you"]
        >>> scores = compute_labse_similarity_batch(zh, en)
        >>> print(scores)  # [0.95, 0.92]
    """
    # 创建临时筛选器
    filter_obj = DataFilter(
        model_name=model_path,
        batch_size=batch_size,
        device=device
    )
    
    return filter_obj.compute_similarity(chinese_texts, english_texts)


# ====================================
# 命令行接口
# ====================================

def main():
    """
    命令行入口函数
    
    使用方式：
        python data_filter.py --input cleaned_data.jsonl --output filtered_data.jsonl
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="高质量翻译数据筛选工具（基于 LaBSE）"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入文件路径（JSONL 格式）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LABSE_MODEL,
        help=f"LaBSE 模型路径（默认: {DEFAULT_LABSE_MODEL}）"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"最小语义相似度（默认: {DEFAULT_MIN_SIMILARITY}）"
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=DEFAULT_MIN_RATIO,
        help=f"最小长度比例（默认: {DEFAULT_MIN_RATIO}）"
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help=f"最大长度比例（默认: {DEFAULT_MAX_RATIO}）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批处理大小（默认: {DEFAULT_BATCH_SIZE}）"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="目标样本数量（可选）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（cuda/cpu，默认自动检测）"
    )
    parser.add_argument(
        "--zh-key",
        type=str,
        default="chinese",
        help="中文字段名（默认: chinese）"
    )
    parser.add_argument(
        "--en-key",
        type=str,
        default="english",
        help="英文字段名（默认: english）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 创建筛选器
    filter_obj = DataFilter(
        model_name=args.model,
        min_similarity=args.min_similarity,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 执行流式筛选
    stats = filter_obj.filter_dataset_streaming(
        input_path=args.input,
        output_path=args.output,
        zh_key=args.zh_key,
        en_key=args.en_key,
        target_samples=args.target_samples
    )
    
    print(stats)


if __name__ == "__main__":
    main()
