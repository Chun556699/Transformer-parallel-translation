"""
数据集构建模块

功能说明：
    提供翻译数据集的构建与加载功能，包括：
    - PyTorch Dataset 封装
    - 动态长度分桶（Bucket Batching）
    - 训练/验证/测试集划分
    - 流式数据加载（大规模数据）
    - 数据增强接口

依赖：
    - torch: PyTorch 数据集基类
    - transformers: 分词器
    - tqdm: 进度条

作者：NMT Project
版本：1.0.0
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

# 导入本地模块
try:
    from .tokenizer import TranslationTokenizer, BilingualTokenizer
except ImportError:
    # 直接运行时的导入
    from tokenizer import TranslationTokenizer, BilingualTokenizer


# ====================================
# 常量定义
# ====================================

# 默认数据集划分比例
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_VAL_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05

# 默认批次大小
DEFAULT_BATCH_SIZE = 32

# 默认最大序列长度
DEFAULT_MAX_LENGTH = 512

# 长度分桶边界
DEFAULT_BUCKET_BOUNDARIES = [32, 64, 128, 256, 512]


@dataclass
class DatasetStats:
    """
    数据集统计信息
    
    属性：
        total_samples: 总样本数
        train_samples: 训练集样本数
        val_samples: 验证集样本数
        test_samples: 测试集样本数
        avg_src_length: 平均源语言长度
        avg_tgt_length: 平均目标语言长度
        max_src_length: 最大源语言长度
        max_tgt_length: 最大目标语言长度
    """
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    avg_src_length: float = 0.0
    avg_tgt_length: float = 0.0
    max_src_length: int = 0
    max_tgt_length: int = 0
    
    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"数据集统计:\n"
            f"  总样本数: {self.total_samples:,}\n"
            f"  训练集: {self.train_samples:,}\n"
            f"  验证集: {self.val_samples:,}\n"
            f"  测试集: {self.test_samples:,}\n"
            f"  平均源语言长度: {self.avg_src_length:.1f}\n"
            f"  平均目标语言长度: {self.avg_tgt_length:.1f}\n"
            f"  最大源语言长度: {self.max_src_length}\n"
            f"  最大目标语言长度: {self.max_tgt_length}"
        )


class TranslationDataset(Dataset):
    """
    翻译数据集类
    
    功能说明：
        封装中英翻译数据，支持：
        - 从 JSONL 文件加载
        - 从内存列表加载
        - 动态分词与编码
        - 数据增强（可选）
    
    参数：
        data: 数据列表或文件路径
        tokenizer: 分词器实例
        src_key: 源语言字段名
        tgt_key: 目标语言字段名
        max_length: 最大序列长度
        max_target_length: 目标语言最大长度
        direction: 翻译方向
        augment_fn: 数据增强函数（可选）
        lazy_load: 是否延迟加载（大数据集）
        
    示例：
        >>> dataset = TranslationDataset(
        ...     data="data/train.jsonl",
        ...     tokenizer=tokenizer,
        ...     src_key="chinese",
        ...     tgt_key="english",
        ...     direction="zh2en"
        ... )
        >>> sample = dataset[0]
        >>> print(sample["input_ids"].shape)
    """
    
    def __init__(
        self,
        data: Union[str, Path, List[Dict[str, str]]],
        tokenizer: TranslationTokenizer,
        src_key: str = "chinese",
        tgt_key: str = "english",
        max_length: int = DEFAULT_MAX_LENGTH,
        max_target_length: Optional[int] = None,
        direction: str = "zh2en",
        augment_fn: Optional[Callable] = None,
        lazy_load: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据集
        
        参数：
            data: 数据（文件路径或列表）
            tokenizer: 分词器
            src_key: 源语言字段名
            tgt_key: 目标语言字段名
            max_length: 最大序列长度
            max_target_length: 目标语言最大长度
            direction: 翻译方向
            augment_fn: 数据增强函数
            lazy_load: 是否延迟加载
            logger: 日志记录器
        """
        self.tokenizer = tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_length = max_length
        self.max_target_length = max_target_length or max_length
        self.direction = direction
        self.augment_fn = augment_fn
        self.lazy_load = lazy_load
        self.logger = logger or logging.getLogger(__name__)
        
        # 加载数据
        if isinstance(data, (str, Path)):
            self.data_path = Path(data)
            if lazy_load:
                # 延迟加载模式：只读取索引
                self.samples = None
                self._build_index()
            else:
                # 立即加载模式
                self.samples = self._load_data(data)
        else:
            # 内存数据
            self.data_path = None
            self.samples = data
        
        self.logger.info(f"数据集初始化完成: {len(self)} 样本")
    
    def _load_data(self, data_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        从文件加载数据
        
        参数：
            data_path: 数据文件路径
            
        返回：
            List[Dict]: 样本列表
        """
        data_path = Path(data_path)
        samples = []
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        self.logger.info(f"正在加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载数据"):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _build_index(self) -> None:
        """
        构建数据索引（延迟加载模式）
        
        记录每行在文件中的字节偏移量，以便按需读取。
        """
        self._line_offsets = []
        
        with open(self.data_path, 'rb') as f:
            offset = 0
            for line in f:
                self._line_offsets.append(offset)
                offset += len(line)
        
        self._total_samples = len(self._line_offsets)
        self.logger.info(f"索引构建完成: {self._total_samples} 行")
    
    def _read_line(self, index: int) -> Dict[str, str]:
        """
        读取指定行（延迟加载模式）
        
        参数：
            index: 行索引
            
        返回：
            Dict: 样本数据
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            f.seek(self._line_offsets[index])
            line = f.readline().strip()
            return json.loads(line)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.lazy_load:
            return self._total_samples
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        参数：
            index: 样本索引
            
        返回：
            Dict[str, torch.Tensor]: 编码后的样本
        """
        # 获取原始样本
        if self.lazy_load:
            sample = self._read_line(index)
        else:
            sample = self.samples[index]
        
        # 提取源语言和目标语言文本
        src_text = sample.get(self.src_key, "")
        tgt_text = sample.get(self.tgt_key, "")
        
        # 数据增强（如果启用）
        if self.augment_fn is not None:
            src_text, tgt_text = self.augment_fn(src_text, tgt_text)
        
        # 准备翻译批次
        encoded = self.tokenizer.prepare_translation_batch(
            src_texts=[src_text],
            tgt_texts=[tgt_text],
            max_length=self.max_length,
            max_target_length=self.max_target_length
        )
        
        # 移除批次维度
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        
        if "labels" in encoded:
            result["labels"] = encoded["labels"].squeeze(0)
        
        # 添加元数据（用于调试）
        result["src_text"] = src_text
        result["tgt_text"] = tgt_text
        
        # 如果有难度信息，也包含进去
        if "difficulty" in sample:
            result["difficulty"] = sample["difficulty"]
        
        return result
    
    def get_raw_sample(self, index: int) -> Dict[str, str]:
        """
        获取原始样本（未编码）
        
        参数：
            index: 样本索引
            
        返回：
            Dict: 原始样本数据
        """
        if self.lazy_load:
            return self._read_line(index)
        return self.samples[index]
    
    def get_lengths(self) -> List[int]:
        """
        获取所有样本的源语言长度
        
        用于长度分桶采样。
        
        返回：
            List[int]: 长度列表
        """
        lengths = []
        
        for i in range(len(self)):
            sample = self.get_raw_sample(i)
            src_text = sample.get(self.src_key, "")
            lengths.append(len(src_text))
        
        return lengths
    
    def get_statistics(self) -> DatasetStats:
        """
        计算数据集统计信息
        
        返回：
            DatasetStats: 统计信息
        """
        stats = DatasetStats(total_samples=len(self))
        
        src_lengths = []
        tgt_lengths = []
        
        for i in range(len(self)):
            sample = self.get_raw_sample(i)
            src_text = sample.get(self.src_key, "")
            tgt_text = sample.get(self.tgt_key, "")
            
            src_lengths.append(len(src_text))
            tgt_lengths.append(len(tgt_text))
        
        if src_lengths:
            stats.avg_src_length = sum(src_lengths) / len(src_lengths)
            stats.max_src_length = max(src_lengths)
        
        if tgt_lengths:
            stats.avg_tgt_length = sum(tgt_lengths) / len(tgt_lengths)
            stats.max_tgt_length = max(tgt_lengths)
        
        return stats


class BucketSampler(Sampler):
    """
    长度分桶采样器
    
    功能说明：
        根据序列长度对样本进行分桶，使同一批次内的样本长度接近，
        减少 padding 浪费，提高训练效率。
    
    参数：
        dataset: 数据集实例
        batch_size: 批次大小
        bucket_boundaries: 分桶边界
        shuffle: 是否打乱
        drop_last: 是否丢弃最后不完整批次
        
    示例：
        >>> sampler = BucketSampler(dataset, batch_size=32)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset: TranslationDataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        bucket_boundaries: List[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        初始化分桶采样器
        
        参数：
            dataset: 数据集
            batch_size: 批次大小
            bucket_boundaries: 分桶边界
            shuffle: 是否打乱
            drop_last: 是否丢弃最后批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries or DEFAULT_BUCKET_BOUNDARIES
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 构建分桶
        self._build_buckets()
    
    def _build_buckets(self) -> None:
        """
        根据长度将样本分配到不同桶中
        """
        # 获取所有样本的长度
        lengths = self.dataset.get_lengths()
        
        # 初始化桶
        num_buckets = len(self.bucket_boundaries) + 1
        self.buckets = [[] for _ in range(num_buckets)]
        
        # 分配样本到桶
        for idx, length in enumerate(lengths):
            bucket_idx = self._get_bucket_index(length)
            self.buckets[bucket_idx].append(idx)
    
    def _get_bucket_index(self, length: int) -> int:
        """
        根据长度确定桶索引
        
        参数：
            length: 序列长度
            
        返回：
            int: 桶索引
        """
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        迭代生成批次索引
        
        返回：
            Iterator[List[int]]: 批次索引迭代器
        """
        batches = []
        
        # 对每个桶内的样本进行分批
        for bucket in self.buckets:
            if not bucket:
                continue
            
            # 打乱桶内顺序
            bucket_copy = bucket.copy()
            if self.shuffle:
                random.shuffle(bucket_copy)
            
            # 分批
            for i in range(0, len(bucket_copy), self.batch_size):
                batch = bucket_copy[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # 打乱批次顺序
        if self.shuffle:
            random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self) -> int:
        """返回批次数量"""
        total_batches = 0
        for bucket in self.buckets:
            if self.drop_last:
                total_batches += len(bucket) // self.batch_size
            else:
                total_batches += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total_batches


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    批次数据整理函数
    
    将多个样本整理为批次张量。
    
    参数：
        batch: 样本列表
        
    返回：
        Dict[str, torch.Tensor]: 批次数据
    """
    # 提取各字段
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    # 处理 labels
    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        result["labels"] = labels
    
    return result


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_bucket_sampler: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    创建数据加载器
    
    参数：
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        use_bucket_sampler: 是否使用分桶采样
        drop_last: 是否丢弃最后批次
        
    返回：
        DataLoader: 数据加载器
    """
    if use_bucket_sampler:
        # 使用分桶采样器
        batch_sampler = BucketSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
    else:
        # 普通采样
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )


def split_dataset(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    shuffle: bool = True
) -> Dict[str, int]:
    """
    划分数据集为训练/验证/测试集
    
    参数：
        data_path: 输入数据路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        shuffle: 是否打乱
        
    返回：
        Dict[str, int]: 各集合的样本数量
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"比例之和应为 1.0，当前为 {total_ratio}")
    
    # 加载所有数据
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载数据"):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    
    # 计算划分点
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # 保存数据
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    
    counts = {}
    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        counts[split_name] = len(split_data)
        logging.info(f"{split_name}: {len(split_data)} 样本 -> {output_path}")
    
    return counts


def load_datasets(
    data_dir: Union[str, Path],
    tokenizer: TranslationTokenizer,
    src_key: str = "chinese",
    tgt_key: str = "english",
    max_length: int = DEFAULT_MAX_LENGTH,
    direction: str = "zh2en"
) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    加载训练/验证/测试数据集
    
    参数：
        data_dir: 数据目录
        tokenizer: 分词器
        src_key: 源语言字段名
        tgt_key: 目标语言字段名
        max_length: 最大序列长度
        direction: 翻译方向
        
    返回：
        Tuple: (训练集, 验证集, 测试集)
    """
    data_dir = Path(data_dir)
    
    # 加载训练集
    train_dataset = TranslationDataset(
        data=data_dir / "train.jsonl",
        tokenizer=tokenizer,
        src_key=src_key,
        tgt_key=tgt_key,
        max_length=max_length,
        direction=direction
    )
    
    # 加载验证集
    val_dataset = TranslationDataset(
        data=data_dir / "val.jsonl",
        tokenizer=tokenizer,
        src_key=src_key,
        tgt_key=tgt_key,
        max_length=max_length,
        direction=direction
    )
    
    # 加载测试集
    test_dataset = TranslationDataset(
        data=data_dir / "test.jsonl",
        tokenizer=tokenizer,
        src_key=src_key,
        tgt_key=tgt_key,
        max_length=max_length,
        direction=direction
    )
    
    return train_dataset, val_dataset, test_dataset


# ====================================
# 数据增强函数
# ====================================

def augment_swap_words(
    src_text: str,
    tgt_text: str,
    swap_prob: float = 0.1
) -> Tuple[str, str]:
    """
    随机交换相邻词（简单数据增强）
    
    参数：
        src_text: 源语言文本
        tgt_text: 目标语言文本
        swap_prob: 交换概率
        
    返回：
        Tuple[str, str]: 增强后的文本对
    """
    # 只对源语言进行增强
    if random.random() > swap_prob:
        return src_text, tgt_text
    
    words = src_text.split()
    if len(words) < 2:
        return src_text, tgt_text
    
    # 随机选择一个位置进行交换
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    
    return ' '.join(words), tgt_text


def augment_dropout_words(
    src_text: str,
    tgt_text: str,
    drop_prob: float = 0.1
) -> Tuple[str, str]:
    """
    随机删除词（简单数据增强）
    
    参数：
        src_text: 源语言文本
        tgt_text: 目标语言文本
        drop_prob: 删除概率
        
    返回：
        Tuple[str, str]: 增强后的文本对
    """
    words = src_text.split()
    if len(words) < 3:
        return src_text, tgt_text
    
    # 随机删除词
    new_words = [w for w in words if random.random() > drop_prob]
    
    # 确保至少保留一个词
    if not new_words:
        new_words = [random.choice(words)]
    
    return ' '.join(new_words), tgt_text


# ====================================
# 命令行接口
# ====================================

def main():
    """
    命令行入口函数
    
    使用方式：
        python dataset.py --input filtered_data.jsonl --output data/splits --split
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="翻译数据集构建工具"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入数据路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="划分数据集"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"训练集比例（默认: {DEFAULT_TRAIN_RATIO}）"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"验证集比例（默认: {DEFAULT_VAL_RATIO}）"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"测试集比例（默认: {DEFAULT_TEST_RATIO}）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="计算数据集统计信息"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.split:
        # 划分数据集
        counts = split_dataset(
            data_path=args.input,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print("\n数据集划分完成:")
        for split_name, count in counts.items():
            print(f"  {split_name}: {count:,} 样本")
    
    if args.stats:
        # 需要分词器才能计算统计信息
        print("统计功能需要加载分词器，请通过 Python API 使用")


if __name__ == "__main__":
    main()
