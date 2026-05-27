"""
模型定义模块

功能说明：
    提供模型配置和管理功能，包括：
    - 模型配置数据类
    - 训练配置数据类
    - 模型加载与保存
    - 混合精度配置

作者：NMT Project
版本：1.0.0
"""

from .config import (
    ModelConfig,
    TrainingConfig,
    OptimizerConfig,
    NMTModelManager,
    load_config_from_yaml,
    create_training_config_from_yaml,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "NMTModelManager",
    "load_config_from_yaml",
    "create_training_config_from_yaml",
]
