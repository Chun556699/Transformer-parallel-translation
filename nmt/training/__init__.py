"""
训练模块

功能说明：
    提供模型训练相关功能，包括：
    - 翻译模型训练器
    - Hugging Face Trainer 封装
    - 训练状态管理
    - 训练入口脚本
    - WMT18冠军策略：Label Smoothing、Noam调度器、检查点平均、模型集成

作者：NMT Project
版本：2.0.0
"""

from .trainer import (
    TranslationTrainer,
    HuggingFaceTrainer,
    TrainingState,
    create_trainer,
)

# WMT18冠军策略
from .wmt18_strategies import (
    LabelSmoothingLoss,
    NoamScheduler,
    get_noam_scheduler,
    LargeBatchOptimizer,
    CheckpointAverager,
    ModelEnsemble,
    WMT18TrainingConfig,
    create_wmt18_training_components,
)

__all__ = [
    # 基础训练
    "TranslationTrainer",
    "HuggingFaceTrainer",
    "TrainingState",
    "create_trainer",
    
    # WMT18策略
    "LabelSmoothingLoss",
    "NoamScheduler",
    "get_noam_scheduler",
    "LargeBatchOptimizer",
    "CheckpointAverager",
    "ModelEnsemble",
    "WMT18TrainingConfig",
    "create_wmt18_training_components",
]
