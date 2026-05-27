"""
数据处理模块

功能说明：
    提供翻译数据的完整处理流水线，包括：
    - 数据清洗与预处理
    - 高质量数据筛选（LaBSE）
    - 分词与编码
    - 数据集构建与加载
    - 课程学习难度分级
    - 联合BPE预处理（WMT18策略）
    - 反向翻译数据增强（WMT18策略）

模块组成：
    - cleaner: 数据清洗
    - data_filter: 高质量筛选
    - tokenizer: 分词器封装
    - dataset: 数据集类
    - curriculum: 课程学习
    - bpe_processor: 联合BPE（WMT18）
    - back_translation: 反向翻译（WMT18）

作者：NMT Project
版本：2.0.0
"""

from .cleaner import DataCleaner, CleaningStats, SamplePair, load_translation2019zh
from .data_filter import DataFilter, FilterStats, FilteredSample, compute_labse_similarity_batch
from .high_quality_sampler import HighQualitySampler, SampleMetadata, SamplingStats, sample_translation_dataset
from .tokenizer import TranslationTokenizer, BilingualTokenizer, analyze_tokenization
from .dataset import (
    TranslationDataset,
    BucketSampler,
    create_dataloader,
    split_dataset,
    load_datasets,
    collate_fn,
    DatasetStats,
)
from .curriculum import (
    CurriculumSampler,
    DynamicCurriculumSampler,
    curriculum_sampling,
    filter_by_difficulty,
    compute_sample_difficulty,
    CurriculumStage,
    CurriculumStats,
    DifficultyLevel,
    DEFAULT_CURRICULUM_CONFIG,
)

# WMT18策略模块
from .bpe_processor import (
    JointBPEProcessor,
    BPEConfig,
    BPEStats,
    WMT18Preprocessor,
    train_joint_bpe,
    create_wmt18_preprocessor,
)
from .back_translation import (
    BackTranslator,
    BackTranslationConfig,
    SyntheticSample,
    DataMixer,
    IterativeBackTranslation,
    create_back_translator,
    augment_with_back_translation,
)

__all__ = [
    # 数据清洗
    "DataCleaner",
    "CleaningStats",
    "SamplePair",
    "load_translation2019zh",
    
    # 数据筛选
    "DataFilter",
    "FilterStats",
    "FilteredSample",
    "compute_labse_similarity_batch",
    
    # 高质量抽样
    "HighQualitySampler",
    "SampleMetadata",
    "SamplingStats",
    "sample_translation_dataset",
    
    # 分词器
    "TranslationTokenizer",
    "BilingualTokenizer",
    "analyze_tokenization",
    
    # 数据集
    "TranslationDataset",
    "BucketSampler",
    "create_dataloader",
    "split_dataset",
    "load_datasets",
    "collate_fn",
    "DatasetStats",
    
    # 课程学习
    "CurriculumSampler",
    "DynamicCurriculumSampler",
    "curriculum_sampling",
    "filter_by_difficulty",
    "compute_sample_difficulty",
    "CurriculumStage",
    "CurriculumStats",
    "DifficultyLevel",
    "DEFAULT_CURRICULUM_CONFIG",
    
    # WMT18策略 - 联合BPE
    "JointBPEProcessor",
    "BPEConfig",
    "BPEStats",
    "WMT18Preprocessor",
    "train_joint_bpe",
    "create_wmt18_preprocessor",
    
    # WMT18策略 - 反向翻译
    "BackTranslator",
    "BackTranslationConfig",
    "SyntheticSample",
    "DataMixer",
    "IterativeBackTranslation",
    "create_back_translator",
    "augment_with_back_translation",
]
