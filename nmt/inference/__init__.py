"""
推理引擎模块

功能说明：
    提供高性能推理引擎，包括：
    - Prompt Cache 机制
    - 动态批处理
    - 多精度支持

作者：NMT Project
版本：1.0.0
"""

from .prompt_cache import (
    PromptCache,
    CacheConfig,
    CacheStats,
    CacheEntry,
    LRUCache,
    cached_translate,
)

__all__ = [
    # Prompt Cache
    "PromptCache",
    "CacheConfig",
    "CacheStats",
    "CacheEntry",
    "LRUCache",
    "cached_translate",
]
