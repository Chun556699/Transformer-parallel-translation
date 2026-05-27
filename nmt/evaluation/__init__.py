"""
评估模块

功能说明：
    提供翻译质量评估功能，包括：
    - 多元评估指标（BLEU, COMET, BERTScore, chrF++, TER）
    - 显著性检验
    - 评估报告生成

评测目标：
    - BLEU ≥ 30.0
    - COMET ≥ 0.80

作者：NMT Project
版本：1.0.0
"""

from .metrics import (
    MultiMetricEvaluator,
    EvaluationResult,
    evaluate_translation,
    compute_significance,
)

__all__ = [
    "MultiMetricEvaluator",
    "EvaluationResult",
    "evaluate_translation",
    "compute_significance",
]
