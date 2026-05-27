#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量数据抽样模块

功能说明：
    从大规模数据集中抽取具有代表性的高质量子集，确保：
    1. 保持原数据集的语义分布特征
    2.维Helsinki-NLP模型的原有性能
    3. 提供分层抽样和难度平衡策略
    4.支持多维度质量评估

核心策略：
    - 分层抽样：按难度等级、领域、长度等维度分层
    - 语义代表性：保持原数据的语义分布
    -阈值：LaBSE相似度≥0.75（比标准更高）
    -多样性保证：避免过度集中某一类型

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import random
import numpy as np

from tqdm import tqdm
import torch

#尝导入导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers 未安装，LaBSE功能将被禁用")

# ====================================
#常量定义
# ====================================

# 默认参数
DEFAULT_TARGET_SAMPLES = 300000  #目标样本数
DEFAULT_QUALITY_THRESHOLD = 0.75  #高质量阈值（比标准0.7更高）
DEFAULT_MAX_LENGTH = 128  # 最大长度限制

# 分层维度
STRATIFICATION_DIMENSIONS = [
    'difficulty',      #等级
    'length_group',   #长度分组
    'domain_type',    #类型（可选）
]

#等级定义
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']
LENGTH_GROUPS = ['short', 'medium', 'long']

# ====================================
# 数据类定义
# ====================================

@dataclass
class SampleMetadata:
    """样本元数据"""
    id: str
    chinese: str
    english: str
    similarity_score: float
    difficulty: str
    length_group: str
    domain_type: str = "general"
    source_dataset: str = "translation2019zh"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'chinese': self.chinese,
            'english': self.english,
            'similarity_score': self.similarity_score,
            'difficulty': self.difficulty,
            'length_group': self.length_group,
            'domain_type': self.domain_type,
            'source_dataset': self.source_dataset,
        }

@dataclass
class SamplingStats:
    """抽样统计信息"""
    total_input: int = 0
    quality_filtered: int = 0
    stratified_selected: int = 0
    final_samples: int = 0
    
    #各分布统计
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[str, int] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_input': self.total_input,
            'quality_filtered': self.quality_filtered,
            'stratified_selected': self.stratified_selected,
            'final_samples': self.final_samples,
            'difficulty_distribution': self.difficulty_distribution,
            'length_distribution': self.length_distribution,
            'domain_distribution': self.domain_distribution,
        }

# ====================================
#高质量数据抽样器
# ====================================

class HighQualitySampler:
    """
   高质量数据抽样器
    
   功能：
        从大规模数据集中抽取保持语义分布的高质量子集
    """
    
    def __init__(
        self,
        target_samples: int = DEFAULT_TARGET_SAMPLES,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        labse_model: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        初始化抽样器
        
        参数：
            target_samples:目标样本数量
            quality_threshold:阈值（LaBSE相似度）
            labse_model: LaBSE模型路径
            device:计算设备
            seed:随种子
        """
        self.target_samples = target_samples
        self.quality_threshold = quality_threshold
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化LaBSE模型
        self.labse_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and labse_model:
            try:
                device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
                self.labse_model = SentenceTransformer(
                    labse_model or "sentence-transformers/LaBSE",
                    device=device
                )
                self.logger.info(f"LaBSE模型已加载到 {device}")
            except Exception as e:
                self.logger.warning(f"LaBSE模型加载失败: {e}")
    
    def compute_sample_metadata(
        self,
        samples: List[Dict[str, str]]
    ) -> List[SampleMetadata]:
        """
       计算样本元数据（难度、长度分组等）
        
        参数：
            samples: 输入样本列表
            
        返回：
            List[SampleMetadata]:数据的样本列表
        """
        self.logger.info("计算样本元数据...")
        
        #批量计算相似度
        if self.labse_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            chinese_texts = [s['chinese'] for s in samples]
            english_texts = [s['english'] for s in samples]
            
            # 分批处理避免内存溢出
            batch_size = 64
            similarities = []
            
            for i in tqdm(range(0, len(samples), batch_size), desc="计算相似度"):
                batch_zh = chinese_texts[i:i+batch_size]
                batch_en = english_texts[i:i+batch_size]
                
                #编码
                zh_embeddings = self.labse_model.encode(
                    batch_zh, show_progress_bar=False, 
                    convert_to_tensor=True, normalize_embeddings=True
                )
                en_embeddings = self.labse_model.encode(
                    batch_en, show_progress_bar=False,
                    convert_to_tensor=True, normalize_embeddings=True
                )
                
                #计算余弦相似度
                batch_similarities = torch.cosine_similarity(zh_embeddings, en_embeddings, dim=1)
                similarities.extend(batch_similarities.cpu().numpy())
        else:
            # 如果没有LaBSE，使用简单的长度比例作为替代
            similarities = []
            for s in samples:
                zh_len = len(s['chinese'])
                en_len = len(s['english'].split())
                #简单的长度比例相似度估计
                ratio = min(zh_len, en_len) / max(zh_len, en_len) if max(zh_len, en_len) > 0 else 0
                similarities.append(ratio)
        
        #构建元数据
        metadata_samples = []
        for i, sample in enumerate(samples):
            #计算长度分组
            zh_len = len(sample['chinese'])
            en_len = len(sample['english'].split())
            avg_length = (zh_len + en_len) / 2
            
            if avg_length < 30:
                length_group = 'short'
            elif avg_length < 80:
                length_group = 'medium'
            else:
                length_group = 'long'
            
            #难度分类（与WMT18测试集分布对齐：easy~33%, medium~64%, hard~3%）
            # 综合考虑：长度、相似度、以及中英文长度比例
            length_ratio = zh_len / en_len if en_len > 0 else 1.0
            
            # Easy: 短句（中等条件）
            if avg_length < 25 and 0.5 <= length_ratio <= 2.0:
                difficulty = 'easy'
            # Hard: 长句 或 长度比例异常
            elif avg_length >= 100 or length_ratio < 0.3 or length_ratio > 3.0:
                difficulty = 'hard'
            # Medium: 其他情况
            else:
                difficulty = 'medium'
            
            metadata_samples.append(SampleMetadata(
                id=f"sample_{i:08d}",
                chinese=sample['chinese'],
                english=sample['english'],
                similarity_score=float(similarities[i]),
                difficulty=difficulty,
                length_group=length_group,
                domain_type="general"  #可扩展为具体领域分类
            ))
        
        return metadata_samples
    
    def stratified_sampling(
        self,
        samples: List[SampleMetadata]
    ) -> List[SampleMetadata]:
        """
        分层抽样
        
        参数：
            samples:数据的样本列表
            
        返回：
            List[SampleMetadata]:抽后的样本列表
        """
        self.logger.info("执行分层抽样...")
        
        # 按维度分组
        stratified_groups = defaultdict(list)
        
        for sample in samples:
            #组多个维度作为分层键
            group_key = f"{sample.difficulty}_{sample.length_group}"
            stratified_groups[group_key].append(sample)
        
        #计算每层的样本数量（按比例分配）
        total_groups = len(stratified_groups)
        samples_per_group = max(1, self.target_samples // len(stratified_groups))
        
        selected_samples = []
        
        for group_key, group_samples in stratified_groups.items():
            #按质量排序（相似度降序）
            group_samples.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # 选择该组的样本数量
            num_to_select = min(samples_per_group, len(group_samples))
            selected_samples.extend(group_samples[:num_to_select])
            
            self.logger.debug(f"组 {group_key}: 选择 {num_to_select}/{len(group_samples)}样")
        
        # 如果选多了，随机抽样到目标数量
        if len(selected_samples) > self.target_samples:
            selected_samples = random.sample(selected_samples, self.target_samples)
        
        # 如果选少了，从剩余样本中补充
        if len(selected_samples) < self.target_samples:
            existing_ids = {s.id for s in selected_samples}
            remaining = [s for s in samples if s.id not in existing_ids]
            needed = min(len(remaining), self.target_samples - len(selected_samples))
            if needed > 0:
                remaining.sort(key=lambda x: x.similarity_score, reverse=True)
                selected_samples.extend(remaining[:needed])
        
        self.logger.info(f"分层抽样完成: {len(selected_samples)}样本")
        return selected_samples
    
    def diversity_enhancement(
        self,
        samples: List[SampleMetadata]
    ) -> List[SampleMetadata]:
        """
       多样性增强（避免重复和过度集中）
        
        参数：
            samples: 输入样本列表
            
        返回：
            List[SampleMetadata]:多样性的样本列表
        """
        self.logger.info("执行多样性增强...")
        
        # 按相似度分桶，避免选择过于相似的样本
        similarity_buckets = defaultdict(list)
        bucket_width = 0.05  # 5%相似度为一个桶
        
        for sample in samples:
            bucket_key = int(sample.similarity_score / bucket_width)
            similarity_buckets[bucket_key].append(sample)
        
        # 从每个桶中选择样本，避免过度集中
        enhanced_samples = []
        max_per_bucket = max(1, len(samples) // len(similarity_buckets))
        
        for bucket_samples in similarity_buckets.values():
            if len(bucket_samples) <= max_per_bucket:
                enhanced_samples.extend(bucket_samples)
            else:
                # 从桶中随机选择
                enhanced_samples.extend(random.sample(bucket_samples, max_per_bucket))
        
        # 如果样本不足目标数量，从所有样本中补充
        if len(enhanced_samples) < self.target_samples and len(samples) >= self.target_samples:
            existing_ids = {s.id for s in enhanced_samples}
            remaining = [s for s in samples if s.id not in existing_ids]
            needed = min(len(remaining), self.target_samples - len(enhanced_samples))
            enhanced_samples.extend(random.sample(remaining, needed))
        
        self.logger.info(f"多样性增强完成: {len(enhanced_samples)} 样本")
        return enhanced_samples
    
    def sample_high_quality_subset(
        self,
        input_path: str,
        output_path: str,
        zh_key: str = "chinese",
        en_key: str = "english"
    ) -> SamplingStats:
        """
        从大规模数据集抽取高质量子集
        
        参数：
            input_path: 输入数据文件路径
            output_path: 输出文件路径
            zh_key: 中文字段名
            en_key:英文字段名
            
        返回：
            SamplingStats:抽统计信息
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = SamplingStats()
        
        self.logger.info("=" * 60)
        self.logger.info("高质量数据抽样流程")
        self.logger.info("=" * 60)
        self.logger.info(f"输入文件: {input_path}")
        self.logger.info(f"输出文件: {output_path}")
        self.logger.info(f"目标样本数: {self.target_samples}")
        self.logger.info(f"质量阈值: {self.quality_threshold}")
        self.logger.info("=" * 60)
        
        # 1. 加载数据
        self.logger.info("1. 加载数据...")
        samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                try:
                    data = json.loads(line.strip())
                    samples.append({
                        'chinese': data[zh_key],
                        'english': data[en_key]
                    })
                except Exception as e:
                    self.logger.warning(f"解析行失败: {e}")
                    continue
        
        stats.total_input = len(samples)
        self.logger.info(f"加载样本数: {stats.total_input}")
        
        # 2.质量过滤
        self.logger.info("2.质过滤...")
        quality_filtered_samples = []
        
        for sample in tqdm(samples, desc="质量过滤"):
            #检查度检查
            if (len(sample['chinese']) > DEFAULT_MAX_LENGTH or 
                len(sample['english'].split()) > DEFAULT_MAX_LENGTH):
                continue
            
            #基本质量检查
            if (len(sample['chinese'].strip()) < 3 or 
                len(sample['english'].strip()) < 3):
                continue
            
            quality_filtered_samples.append(sample)
        
        stats.quality_filtered = len(quality_filtered_samples)
        self.logger.info(f"质量过滤后: {stats.quality_filtered}样本")
        
        # 3. 计算元数据
        self.logger.info("3. 计算样本元数据...")
        metadata_samples = self.compute_sample_metadata(quality_filtered_samples)
        
        # 4.高质量筛选
        self.logger.info("4.高质量筛选...")
        high_quality_samples = [
            meta for meta in metadata_samples 
            if meta.similarity_score >= self.quality_threshold
        ]
        
        self.logger.info(f"高质量样本: {len(high_quality_samples)}")
        
        if len(high_quality_samples) < self.target_samples:
            self.logger.warning(
                f"高质量样本不足目标数量 ({len(high_quality_samples)} < {self.target_samples})，"
                f"将使用所有高质量样本"
            )
            self.target_samples = len(high_quality_samples)
        
        # 5. 分层抽样
        self.logger.info("5. 分层抽样...")
        stratified_samples = self.stratified_sampling(high_quality_samples)
        stats.stratified_selected = len(stratified_samples)
        
        # 6.多样性增强
        self.logger.info("6.多性样性增强...")
        final_samples = self.diversity_enhancement(stratified_samples)
        stats.final_samples = len(final_samples)
        
        # 7.统计分布
        self.logger.info("7. 生成统计信息...")
        stats.difficulty_distribution = Counter(s.difficulty for s in final_samples)
        stats.length_distribution = Counter(s.length_group for s in final_samples)
        stats.domain_distribution = Counter(s.domain_type for s in final_samples)
        
        # 8. 保存结果
        self.logger.info("8. 保存结果...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in tqdm(final_samples, desc="保存样本"):
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        # 保存统计信息
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 输出总结
        self.logger.info("=" * 60)
        self.logger.info("抽样完成统计")
        self.logger.info("=" * 60)
        self.logger.info(f"输入样本: {stats.total_input:,}")
        self.logger.info(f"质量过滤: {stats.quality_filtered:,}")
        self.logger.info(f"高质量样本: {len(high_quality_samples):,}")
        self.logger.info(f"分层选择: {stats.stratified_selected:,}")
        self.logger.info(f"最终输出: {stats.final_samples:,}")
        self.logger.info("")
        self.logger.info("分布统计:")
        self.logger.info(f" 分布: {dict(stats.difficulty_distribution)}")
        self.logger.info(f"  长度分布: {dict(stats.length_distribution)}")
        self.logger.info(f" 分布: {dict(stats.domain_distribution)}")
        self.logger.info("=" * 60)
        
        return stats

# ====================================
# 便捷函数
# ====================================

def sample_translation_dataset(
    input_path: str,
    output_path: str,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    **kwargs
) -> SamplingStats:
    """
    便捷函数：抽样翻译数据集
    
    参数：
        input_path: 输入路径
        output_path: 输出路径
        target_samples:目标样本数
        quality_threshold:质阈值
        **kwargs:其他参数
        
    返回：
        SamplingStats:抽统计
    """
    sampler = HighQualitySampler(
        target_samples=target_samples,
        quality_threshold=quality_threshold,
        **kwargs
    )
    
    return sampler.sample_high_quality_subset(input_path, output_path)

# ====================================
#命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="高质量翻译数据抽样工具"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入数据文件路径（JSONL格式）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=DEFAULT_TARGET_SAMPLES,
        help=f"目标样本数量（默认: {DEFAULT_TARGET_SAMPLES:,}）"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLD,
        help=f"质量阈值（默认: {DEFAULT_QUALITY_THRESHOLD}）"
    )
    parser.add_argument(
        "--labse-model",
        type=str,
        default=None,
        help="LaBSE模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（cuda/cpu）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
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
    
    #配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    #执行抽样
    sampler = HighQualitySampler(
        target_samples=args.target_samples,
        quality_threshold=args.quality_threshold,
        labse_model=args.labse_model,
        device=args.device,
        seed=args.seed
    )
    
    stats = sampler.sample_high_quality_subset(
        input_path=args.input,
        output_path=args.output,
        zh_key=args.zh_key,
        en_key=args.en_key
    )
    
    return 0

if __name__ == "__main__":
    exit(main())