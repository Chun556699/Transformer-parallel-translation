"""
课程学习模块

功能说明：
    实现基于难度分级的课程学习（Curriculum Learning）策略：
    - 根据训练进度动态调整样本难度
    - 从简单样本开始，逐步引入困难样本
    - 加速模型收敛，提升长句翻译质量

难度分级依据：
    - 句子长度（token 数）
    - 对齐置信度（LaBSE 语义相似度）
    - 词汇稀有度（OOV 比例）

训练阶段：
    - 阶段一（前 30% epoch）：短句 + 高置信度
    - 阶段二（中 40% epoch）：中等长度 + 标准样本
    - 阶段三（后 30% epoch）：长句 + 困难样本

依赖：
    - torch: PyTorch 采样器
    - numpy: 数值计算

作者：NMT Project
版本：1.0.0
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset

# ====================================
# 常量定义
# ====================================

# 难度等级
class DifficultyLevel(Enum):
    """难度等级枚举"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# 默认课程学习配置
DEFAULT_CURRICULUM_CONFIG = {
    "easy": {
        "max_length": 30,        # 最大 token 数
        "min_confidence": 0.85,  # 最小对齐置信度
        "epoch_ratio": 0.3,      # 占总 epoch 的比例
    },
    "medium": {
        "max_length": 80,
        "min_confidence": 0.7,
        "epoch_ratio": 0.4,
    },
    "hard": {
        "max_length": 512,
        "min_confidence": 0.5,
        "epoch_ratio": 0.3,
    }
}


@dataclass
class CurriculumStage:
    """
    课程学习阶段配置
    
    属性：
        name: 阶段名称
        max_length: 最大序列长度
        min_confidence: 最小置信度
        epoch_ratio: epoch 比例
        sample_indices: 该阶段包含的样本索引
    """
    name: str
    max_length: int
    min_confidence: float
    epoch_ratio: float
    sample_indices: List[int] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"CurriculumStage(name={self.name}, "
            f"max_length={self.max_length}, "
            f"min_confidence={self.min_confidence}, "
            f"samples={len(self.sample_indices)})"
        )


@dataclass
class CurriculumStats:
    """
    课程学习统计信息
    
    属性：
        easy_samples: 简单样本数
        medium_samples: 中等样本数
        hard_samples: 困难样本数
        current_stage: 当前阶段
        current_epoch: 当前 epoch
    """
    easy_samples: int = 0
    medium_samples: int = 0
    hard_samples: int = 0
    current_stage: str = "easy"
    current_epoch: int = 0
    
    def __str__(self) -> str:
        return (
            f"课程学习统计:\n"
            f"  简单样本: {self.easy_samples:,}\n"
            f"  中等样本: {self.medium_samples:,}\n"
            f"  困难样本: {self.hard_samples:,}\n"
            f"  当前阶段: {self.current_stage}\n"
            f"  当前epoch: {self.current_epoch}"
        )


class CurriculumSampler(Sampler):
    """
    课程学习采样器
    
    功能说明：
        根据训练进度动态调整采样策略：
        - 早期训练阶段：只采样简单样本
        - 中期训练阶段：采样简单 + 中等样本
        - 后期训练阶段：采样全部样本
    
    参数：
        dataset: 数据集实例
        total_epochs: 总训练轮数
        curriculum_config: 课程配置
        shuffle: 是否打乱
        seed: 随机种子
        logger: 日志记录器
        
    示例：
        >>> sampler = CurriculumSampler(
        ...     dataset=train_dataset,
        ...     total_epochs=10,
        ...     curriculum_config=DEFAULT_CURRICULUM_CONFIG
        ... )
        >>> sampler.set_epoch(0)  # 设置当前 epoch
        >>> for idx in sampler:
        ...     sample = dataset[idx]
    """
    
    def __init__(
        self,
        dataset: Dataset,
        total_epochs: int,
        curriculum_config: Optional[Dict] = None,
        shuffle: bool = True,
        seed: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化课程学习采样器
        
        参数：
            dataset: 数据集
            total_epochs: 总 epoch 数
            curriculum_config: 课程配置
            shuffle: 是否打乱
            seed: 随机种子
            logger: 日志记录器
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.curriculum_config = curriculum_config or DEFAULT_CURRICULUM_CONFIG
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        
        # 当前 epoch
        self.current_epoch = 0
        
        # 统计信息
        self.stats = CurriculumStats()
        
        # 构建难度分级
        self._build_curriculum_stages()
        
        self.logger.info(f"课程学习采样器初始化完成")
        self.logger.info(str(self.stats))
    
    def _build_curriculum_stages(self) -> None:
        """
        根据样本特征构建难度分级
        
        遍历数据集，根据长度和置信度将样本分配到不同阶段。
        优先使用数据中已有的 difficulty 标签。
        """
        # 初始化阶段
        self.stages = {
            "easy": CurriculumStage(
                name="easy",
                **self.curriculum_config["easy"]
            ),
            "medium": CurriculumStage(
                name="medium",
                **self.curriculum_config["medium"]
            ),
            "hard": CurriculumStage(
                name="hard",
                **self.curriculum_config["hard"]
            )
        }
        
        # 分配样本
        for idx in range(len(self.dataset)):
            # 获取样本信息
            try:
                sample = self.dataset.get_raw_sample(idx)
            except AttributeError:
                # 如果数据集没有 get_raw_sample 方法，跳过
                sample = {}
            
            # 获取难度信息 - 优先使用已有的标签
            difficulty = sample.get("difficulty", None)
            
            if difficulty and difficulty in ["easy", "medium", "hard"]:
                # 如果样本已有有效的难度标签，直接使用
                self.stages[difficulty].sample_indices.append(idx)
            else:
                # 根据长度推断难度 - 使用与抽样一致的逻辑
                zh_text = sample.get("chinese", "")
                en_text = sample.get("english", "")
                zh_len = len(zh_text)
                en_len = len(en_text.split())
                avg_length = (zh_len + en_len) / 2  # 使用平均长度，与抽样一致
                length_ratio = zh_len / en_len if en_len > 0 else 1.0
                
                # 分级逻辑 - 与 high_quality_sampler.py 保持一致
                # Easy: 短句（中等条件）
                if avg_length < 25 and 0.5 <= length_ratio <= 2.0:
                    self.stages["easy"].sample_indices.append(idx)
                # Hard: 长句 或 长度比例异常
                elif avg_length >= 100 or length_ratio < 0.3 or length_ratio > 3.0:
                    self.stages["hard"].sample_indices.append(idx)
                # Medium: 其他情况
                else:
                    self.stages["medium"].sample_indices.append(idx)
        
        # 更新统计
        self.stats.easy_samples = len(self.stages["easy"].sample_indices)
        self.stats.medium_samples = len(self.stages["medium"].sample_indices)
        self.stats.hard_samples = len(self.stages["hard"].sample_indices)
        
        self.logger.info(f"难度分级完成:")
        for stage_name, stage in self.stages.items():
            self.logger.info(f"  {stage_name}: {len(stage.sample_indices):,} 样本")
    
    def _get_current_stage(self) -> str:
        """
        根据当前 epoch 确定训练阶段
        
        返回：
            str: 当前阶段名称
        """
        progress = self.current_epoch / self.total_epochs
        
        easy_end = self.curriculum_config["easy"]["epoch_ratio"]
        medium_end = easy_end + self.curriculum_config["medium"]["epoch_ratio"]
        
        if progress < easy_end:
            return "easy"
        elif progress < medium_end:
            return "medium"
        else:
            return "hard"
    
    def _get_active_indices(self) -> List[int]:
        """
        获取当前阶段的活跃样本索引
        
        渐进式策略：
        - easy 阶段：只使用简单样本
        - medium 阶段：使用简单 + 中等样本
        - hard 阶段：使用全部样本
        
        自动扩展策略：
        - 如果当前阶段没有样本，自动扩展到下一阶段
        - 确保始终有样本可用于训练
        
        返回：
            List[int]: 活跃样本索引列表
        """
        stage = self._get_current_stage()
        self.stats.current_stage = stage
        
        # 根据阶段获取样本索引，如果当前阶段没有样本则扩展到下一阶段
        if stage == "easy":
            indices = self.stages["easy"].sample_indices.copy()
            # 如果 easy 没有样本，扩展到 medium
            if not indices:
                self.logger.warning("easy 阶段没有样本，自动扩展到 medium 阶段")
                indices = self.stages["medium"].sample_indices.copy()
                self.stats.current_stage = "medium"
            # 如果 medium 也没有，扩展到 hard
            if not indices:
                self.logger.warning("medium 阶段也没有样本，自动扩展到 hard 阶段")
                indices = (
                    self.stages["medium"].sample_indices +
                    self.stages["hard"].sample_indices
                )
                self.stats.current_stage = "hard"
        elif stage == "medium":
            indices = (
                self.stages["easy"].sample_indices +
                self.stages["medium"].sample_indices
            )
            # 如果没有样本，扩展到 hard
            if not indices:
                self.logger.warning("easy + medium 阶段没有样本，自动扩展到 hard 阶段")
                indices = (
                    self.stages["easy"].sample_indices +
                    self.stages["medium"].sample_indices +
                    self.stages["hard"].sample_indices
                )
                self.stats.current_stage = "hard"
        else:
            indices = (
                self.stages["easy"].sample_indices +
                self.stages["medium"].sample_indices +
                self.stages["hard"].sample_indices
            )
        
        # 最终检查：如果仍然没有样本，使用全部数据
        if not indices:
            self.logger.warning("所有阶段都没有样本，使用数据集全部索引")
            indices = list(range(len(self.dataset)))
        
        return indices
    
    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch
        
        参数：
            epoch: epoch 编号
        """
        self.current_epoch = epoch
        self.stats.current_epoch = epoch
        
        stage = self._get_current_stage()
        active_count = len(self._get_active_indices())
        
        self.logger.info(
            f"Epoch {epoch}/{self.total_epochs}: "
            f"阶段={stage}, 活跃样本={active_count:,}"
        )
    
    def __iter__(self) -> Iterator[int]:
        """
        迭代生成样本索引
        
        返回：
            Iterator[int]: 样本索引迭代器
        """
        indices = self._get_active_indices()
        
        if self.shuffle:
            # 使用 epoch 相关的随机种子
            rng = random.Random(self.seed + self.current_epoch)
            rng.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """返回当前阶段的样本数量"""
        return len(self._get_active_indices())


class DynamicCurriculumSampler(Sampler):
    """
    动态课程学习采样器
    
    功能说明：
        根据模型在各类样本上的表现动态调整采样策略：
        - 追踪各难度级别的损失
        - 优先采样损失较高的类别
        - 实现自适应的课程学习
    
    参数：
        dataset: 数据集
        batch_size: 批次大小
        initial_temperature: 初始温度（控制探索/利用平衡）
        logger: 日志记录器
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        initial_temperature: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化动态课程学习采样器
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = initial_temperature
        self.logger = logger or logging.getLogger(__name__)
        
        # 各难度级别的统计
        self.difficulty_stats = {
            "easy": {"count": 0, "total_loss": 0.0, "avg_loss": 0.0},
            "medium": {"count": 0, "total_loss": 0.0, "avg_loss": 0.0},
            "hard": {"count": 0, "total_loss": 0.0, "avg_loss": 0.0},
        }
        
        # 索引分组
        self._build_difficulty_groups()
    
    def _build_difficulty_groups(self) -> None:
        """构建难度分组"""
        self.difficulty_indices = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset.get_raw_sample(idx)
                difficulty = sample.get("difficulty", "medium")
                self.difficulty_indices[difficulty].append(idx)
            except Exception:
                self.difficulty_indices["medium"].append(idx)
    
    def update_loss(self, difficulty: str, loss: float) -> None:
        """
        更新指定难度级别的损失统计
        
        参数：
            difficulty: 难度级别
            loss: 损失值
        """
        if difficulty in self.difficulty_stats:
            stats = self.difficulty_stats[difficulty]
            stats["count"] += 1
            stats["total_loss"] += loss
            stats["avg_loss"] = stats["total_loss"] / stats["count"]
    
    def get_sampling_weights(self) -> Dict[str, float]:
        """
        计算各难度级别的采样权重
        
        基于损失的 softmax 分布，损失越高的类别采样概率越大。
        
        返回：
            Dict[str, float]: 采样权重
        """
        losses = {
            k: v["avg_loss"] if v["avg_loss"] > 0 else 1.0
            for k, v in self.difficulty_stats.items()
        }
        
        # Softmax 计算
        max_loss = max(losses.values())
        exp_losses = {
            k: np.exp((v - max_loss) / self.temperature)
            for k, v in losses.items()
        }
        total = sum(exp_losses.values())
        
        weights = {k: v / total for k, v in exp_losses.items()}
        return weights
    
    def __iter__(self) -> Iterator[int]:
        """迭代生成样本索引"""
        weights = self.get_sampling_weights()
        
        # 计算各难度级别的采样数量
        total = len(self.dataset)
        sample_counts = {
            k: int(total * w)
            for k, w in weights.items()
        }
        
        # 采样
        indices = []
        for difficulty, count in sample_counts.items():
            available = self.difficulty_indices[difficulty]
            if available:
                sampled = random.choices(available, k=min(count, len(available)))
                indices.extend(sampled)
        
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.dataset)


def curriculum_sampling(
    dataset: Dataset,
    epoch: int,
    total_epochs: int,
    config: Optional[Dict] = None
) -> List[int]:
    """
    基于训练进度的课程学习采样函数
    
    这是计划中提到的核心函数，根据训练进度选择不同难度的样本。
    
    参数：
        dataset: 数据集
        epoch: 当前 epoch
        total_epochs: 总 epoch 数
        config: 课程配置
        
    返回：
        List[int]: 采样的样本索引
        
    示例：
        >>> indices = curriculum_sampling(dataset, epoch=0, total_epochs=10)
        >>> for idx in indices:
        ...     sample = dataset[idx]
    """
    config = config or DEFAULT_CURRICULUM_CONFIG
    progress = epoch / total_epochs
    
    # 分类样本
    easy_indices = []
    medium_indices = []
    hard_indices = []
    
    for idx in range(len(dataset)):
        try:
            sample = dataset.get_raw_sample(idx)
        except AttributeError:
            sample = {}
        
        difficulty = sample.get("difficulty", None)
        
        if difficulty == "easy":
            easy_indices.append(idx)
        elif difficulty == "medium":
            medium_indices.append(idx)
        elif difficulty == "hard":
            hard_indices.append(idx)
        else:
            # 根据长度推断
            src_text = sample.get("chinese", sample.get("english", ""))
            length = len(src_text)
            confidence = sample.get("similarity_score", 0.8)
            
            if length <= config["easy"]["max_length"] and \
               confidence >= config["easy"]["min_confidence"]:
                easy_indices.append(idx)
            elif length <= config["medium"]["max_length"] and \
                 confidence >= config["medium"]["min_confidence"]:
                medium_indices.append(idx)
            else:
                hard_indices.append(idx)
    
    # 根据进度选择样本
    easy_end = config["easy"]["epoch_ratio"]
    medium_end = easy_end + config["medium"]["epoch_ratio"]
    
    if progress < easy_end:
        # 前 30% epoch：短句 + 高置信度对齐
        return easy_indices
    elif progress < medium_end:
        # 中 40% epoch：中等长度 + 标准样本
        return easy_indices + medium_indices
    else:
        # 后 30% epoch：长句 + 困难样本（全量数据）
        return easy_indices + medium_indices + hard_indices


def filter_by_difficulty(
    dataset: Dataset,
    max_len: int = 30,
    min_confidence: float = 0.85
) -> List[int]:
    """
    根据难度条件过滤样本
    
    参数：
        dataset: 数据集
        max_len: 最大长度
        min_confidence: 最小置信度
        
    返回：
        List[int]: 符合条件的样本索引
    """
    filtered_indices = []
    
    for idx in range(len(dataset)):
        try:
            sample = dataset.get_raw_sample(idx)
        except AttributeError:
            continue
        
        # 获取长度
        src_text = sample.get("chinese", sample.get("english", ""))
        length = len(src_text)
        
        # 获取置信度
        confidence = sample.get("similarity_score", 0.8)
        
        # 过滤
        if length <= max_len and confidence >= min_confidence:
            filtered_indices.append(idx)
    
    return filtered_indices


def compute_sample_difficulty(
    src_text: str,
    tgt_text: str,
    similarity_score: float = 0.8,
    vocab: Optional[set] = None
) -> Dict[str, Any]:
    """
    计算单个样本的难度指标
    
    参数：
        src_text: 源语言文本
        tgt_text: 目标语言文本
        similarity_score: 语义相似度分数
        vocab: 词表（用于计算 OOV 比例）
        
    返回：
        Dict: 难度指标
    """
    # 长度特征
    src_length = len(src_text)
    tgt_length = len(tgt_text)
    
    # 长度比例
    length_ratio = src_length / tgt_length if tgt_length > 0 else 0
    
    # OOV 比例（如果提供词表）
    oov_ratio = 0.0
    if vocab:
        tokens = list(src_text)  # 简单按字符分词
        oov_count = sum(1 for t in tokens if t not in vocab)
        oov_ratio = oov_count / len(tokens) if tokens else 0
    
    # 综合难度分数（0-1，越高越难）
    # 权重：长度 40%，置信度 40%，OOV 20%
    length_score = min(src_length / 100, 1.0)  # 100 字符以上为最难
    confidence_score = 1 - similarity_score  # 置信度越低越难
    
    difficulty_score = (
        0.4 * length_score +
        0.4 * confidence_score +
        0.2 * oov_ratio
    )
    
    # 分级
    if difficulty_score < 0.3:
        difficulty_level = "easy"
    elif difficulty_score < 0.6:
        difficulty_level = "medium"
    else:
        difficulty_level = "hard"
    
    return {
        "src_length": src_length,
        "tgt_length": tgt_length,
        "length_ratio": length_ratio,
        "similarity_score": similarity_score,
        "oov_ratio": oov_ratio,
        "difficulty_score": difficulty_score,
        "difficulty_level": difficulty_level,
    }


# ====================================
# 命令行接口
# ====================================

def main():
    """
    命令行入口函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="课程学习工具"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="分析数据集难度分布（JSONL 文件路径）"
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        # 分析难度分布
        import json
        from collections import Counter
        
        difficulties = Counter()
        
        with open(args.analyze, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        difficulty = sample.get("difficulty", "unknown")
                        difficulties[difficulty] += 1
                    except json.JSONDecodeError:
                        continue
        
        print("\n难度分布:")
        for diff, count in difficulties.most_common():
            print(f"  {diff}: {count:,}")


if __name__ == "__main__":
    main()
