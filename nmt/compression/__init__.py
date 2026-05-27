"""
模型压缩模块

功能说明：
    提供模型压缩相关功能，包括：
    - 结构化剪枝（FFN + 注意力头）
    - 动态量化（INT8/INT4）
    - 压缩统计与分析

压缩目标：
    - 原始模型：~300MB
    - 压缩后：~100MB/方向
    - BLEU 下降：<0.3

作者：NMT Project
版本：1.0.0
"""

from .pruning import (
    StructuredPruner,
    PruningConfig,
    PruningStats,
    prune_model,
    iterative_prune_and_finetune,
)

from .quantization import (
    DynamicQuantizer,
    INT4Quantizer,
    QuantizationConfig,
    QuantizationStats,
    quantize_model,
    create_calibration_dataset,
    save_quantized_model,
)

__all__ = [
    # 剪枝
    "StructuredPruner",
    "PruningConfig",
    "PruningStats",
    "prune_model",
    "iterative_prune_and_finetune",
    
    # 量化
    "DynamicQuantizer",
    "INT4Quantizer",
    "QuantizationConfig",
    "QuantizationStats",
    "quantize_model",
    "create_calibration_dataset",
    "save_quantized_model",
]
