"""
Prompt Cache 模块

功能说明：
    实现 KV Cache 复用机制，加速重复/相似模式的翻译：
    - LRU 缓存策略
    - 前缀树快速查找
    - 可配置缓存大小
    - 内存管理

加速场景：
    - 重复模式翻译（如文档模板）
    - 相似句式的批量翻译
    - 术语词典查询

预期效果：
    - 重复模式加速 3-5x

依赖：
    - torch: 张量缓存
    - collections: LRU 缓存

作者：NMT Project
版本：1.0.0
"""

import os
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import threading

import torch
import numpy as np


# ====================================
# 常量定义
# ====================================

# 默认缓存大小（MB）
DEFAULT_CACHE_SIZE_MB = 512

# 默认最大缓存条目数
DEFAULT_MAX_ENTRIES = 10000

# 缓存键前缀长度
DEFAULT_PREFIX_LENGTH = 50


@dataclass
class CacheConfig:
    """
    缓存配置数据类
    
    属性：
        max_size_mb: 最大缓存大小（MB）
        max_entries: 最大缓存条目数
        prefix_length: 前缀匹配长度
        ttl_seconds: 缓存过期时间（秒），0 表示不过期
        enable_compression: 是否启用压缩
    """
    max_size_mb: int = DEFAULT_CACHE_SIZE_MB
    max_entries: int = DEFAULT_MAX_ENTRIES
    prefix_length: int = DEFAULT_PREFIX_LENGTH
    ttl_seconds: int = 0
    enable_compression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_size_mb": self.max_size_mb,
            "max_entries": self.max_entries,
            "prefix_length": self.prefix_length,
            "ttl_seconds": self.ttl_seconds,
            "enable_compression": self.enable_compression,
        }


@dataclass
class CacheStats:
    """
    缓存统计信息
    
    属性：
        hits: 缓存命中次数
        misses: 缓存未命中次数
        hit_rate: 命中率
        total_entries: 当前缓存条目数
        total_size_mb: 当前缓存大小（MB）
        evictions: 驱逐次数
    """
    hits: int = 0
    misses: int = 0
    total_entries: int = 0
    total_size_mb: float = 0.0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """计算命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"缓存统计:\n"
            f"  命中次数: {self.hits}\n"
            f"  未命中次数: {self.misses}\n"
            f"  命中率: {self.hit_rate:.2%}\n"
            f"  缓存条目: {self.total_entries}\n"
            f"  缓存大小: {self.total_size_mb:.2f} MB\n"
            f"  驱逐次数: {self.evictions}"
        )


@dataclass
class CacheEntry:
    """
    缓存条目数据类
    
    属性：
        key: 缓存键
        value: 缓存值（翻译结果）
        kv_cache: KV 缓存张量（可选）
        timestamp: 创建时间戳
        access_count: 访问次数
        size_bytes: 条目大小
    """
    key: str
    value: str
    kv_cache: Optional[Dict[str, torch.Tensor]] = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self) -> None:
        """更新访问信息"""
        self.access_count += 1
        self.timestamp = time.time()


class LRUCache:
    """
    LRU 缓存实现
    
    功能说明：
        基于 OrderedDict 实现的 LRU（最近最少使用）缓存：
        - O(1) 查找和插入
        - 自动驱逐最旧条目
        - 线程安全
    
    参数：
        max_entries: 最大条目数
        max_size_mb: 最大缓存大小
        
    示例：
        >>> cache = LRUCache(max_entries=1000)
        >>> cache.put("key", "value")
        >>> result = cache.get("key")
    """
    
    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_size_mb: int = DEFAULT_CACHE_SIZE_MB
    ):
        """初始化 LRU 缓存"""
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        
        self.stats = CacheStats()
    
    def _compute_entry_size(self, entry: CacheEntry) -> int:
        """
        计算缓存条目大小
        
        参数：
            entry: 缓存条目
            
        返回：
            int: 大小（字节）
        """
        size = len(entry.key.encode('utf-8'))
        size += len(entry.value.encode('utf-8'))
        
        if entry.kv_cache:
            for tensor in entry.kv_cache.values():
                size += tensor.nelement() * tensor.element_size()
        
        return size
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        获取缓存条目
        
        参数：
            key: 缓存键
            
        返回：
            Optional[CacheEntry]: 缓存条目
        """
        with self._lock:
            if key in self._cache:
                # 命中：移动到末尾
                entry = self._cache.pop(key)
                entry.update_access()
                self._cache[key] = entry
                self.stats.hits += 1
                return entry
            else:
                self.stats.misses += 1
                return None
    
    def put(
        self,
        key: str,
        value: str,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        添加缓存条目
        
        参数：
            key: 缓存键
            value: 缓存值
            kv_cache: KV 缓存
        """
        with self._lock:
            # 如果键已存在，先删除
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_size_bytes -= old_entry.size_bytes
            
            # 创建新条目
            entry = CacheEntry(
                key=key,
                value=value,
                kv_cache=kv_cache
            )
            entry.size_bytes = self._compute_entry_size(entry)
            
            # 检查是否需要驱逐
            while (
                len(self._cache) >= self.max_entries or
                self._current_size_bytes + entry.size_bytes > self.max_size_bytes
            ):
                if not self._cache:
                    break
                self._evict_oldest()
            
            # 添加新条目
            self._cache[key] = entry
            self._current_size_bytes += entry.size_bytes
            self.stats.total_entries = len(self._cache)
            self.stats.total_size_mb = self._current_size_bytes / (1024 * 1024)
    
    def _evict_oldest(self) -> None:
        """驱逐最旧的条目"""
        if self._cache:
            _, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
    
    def contains(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache
    
    def remove(self, key: str) -> bool:
        """
        删除缓存条目
        
        参数：
            key: 缓存键
            
        返回：
            bool: 是否删除成功
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_size_bytes -= entry.size_bytes
                self.stats.total_entries = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self.stats = CacheStats()


class TrieNode:
    """
    前缀树节点
    
    用于快速匹配具有公共前缀的字符串。
    """
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.cache_key: Optional[str] = None


class PrefixTrie:
    """
    前缀树实现
    
    功能说明：
        用于快速匹配具有公共前缀的翻译请求。
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, text: str, cache_key: str) -> None:
        """
        插入文本
        
        参数：
            text: 文本
            cache_key: 缓存键
        """
        node = self.root
        for char in text:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.cache_key = cache_key
    
    def search_prefix(self, text: str) -> Optional[str]:
        """
        搜索最长匹配前缀
        
        参数：
            text: 输入文本
            
        返回：
            Optional[str]: 匹配的缓存键
        """
        node = self.root
        last_match_key = None
        
        for char in text:
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_end:
                last_match_key = node.cache_key
        
        return last_match_key
    
    def remove(self, text: str) -> bool:
        """
        从前缀树中删除
        
        参数：
            text: 文本
            
        返回：
            bool: 是否删除成功
        """
        # 简化实现：标记为非结束节点
        node = self.root
        for char in text:
            if char not in node.children:
                return False
            node = node.children[char]
        
        if node.is_end:
            node.is_end = False
            node.cache_key = None
            return True
        return False


class PromptCache:
    """
    Prompt Cache 管理器
    
    功能说明：
        管理翻译请求的缓存，支持：
        - 精确匹配缓存
        - 前缀匹配缓存
        - KV Cache 复用
    
    参数：
        config: 缓存配置
        logger: 日志记录器
        
    示例：
        >>> cache = PromptCache()
        >>> cache.put("你好", "Hello")
        >>> result = cache.get("你好")
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """初始化 Prompt Cache"""
        self.config = config or CacheConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 精确匹配缓存
        self._exact_cache = LRUCache(
            max_entries=self.config.max_entries,
            max_size_mb=self.config.max_size_mb
        )
        
        # 前缀树（用于前缀匹配）
        self._prefix_trie = PrefixTrie()
        
        self.logger.info("Prompt Cache 初始化完成")
        self.logger.info(f"  最大缓存: {self.config.max_size_mb} MB")
        self.logger.info(f"  最大条目: {self.config.max_entries}")
    
    def _compute_key(self, text: str) -> str:
        """
        计算缓存键
        
        参数：
            text: 输入文本
            
        返回：
            str: 缓存键（MD5 哈希）
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(
        self,
        text: str,
        use_prefix_match: bool = True
    ) -> Optional[str]:
        """
        获取缓存的翻译结果
        
        参数：
            text: 输入文本
            use_prefix_match: 是否使用前缀匹配
            
        返回：
            Optional[str]: 缓存的翻译结果
        """
        key = self._compute_key(text)
        
        # 尝试精确匹配
        entry = self._exact_cache.get(key)
        if entry:
            return entry.value
        
        # 尝试前缀匹配
        if use_prefix_match:
            prefix = text[:self.config.prefix_length]
            prefix_key = self._prefix_trie.search_prefix(prefix)
            if prefix_key:
                entry = self._exact_cache.get(prefix_key)
                if entry:
                    return entry.value
        
        return None
    
    def get_with_kv_cache(
        self,
        text: str
    ) -> Tuple[Optional[str], Optional[Dict[str, torch.Tensor]]]:
        """
        获取缓存的翻译结果和 KV Cache
        
        参数：
            text: 输入文本
            
        返回：
            Tuple: (翻译结果, KV Cache)
        """
        key = self._compute_key(text)
        entry = self._exact_cache.get(key)
        
        if entry:
            return entry.value, entry.kv_cache
        
        return None, None
    
    def put(
        self,
        text: str,
        translation: str,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        添加翻译结果到缓存
        
        参数：
            text: 输入文本
            translation: 翻译结果
            kv_cache: KV Cache（可选）
        """
        key = self._compute_key(text)
        
        # 添加到精确缓存
        self._exact_cache.put(key, translation, kv_cache)
        
        # 添加到前缀树
        prefix = text[:self.config.prefix_length]
        self._prefix_trie.insert(prefix, key)
    
    def invalidate(self, text: str) -> bool:
        """
        使缓存条目失效
        
        参数：
            text: 输入文本
            
        返回：
            bool: 是否成功
        """
        key = self._compute_key(text)
        
        # 从精确缓存删除
        success = self._exact_cache.remove(key)
        
        # 从前缀树删除
        prefix = text[:self.config.prefix_length]
        self._prefix_trie.remove(prefix)
        
        return success
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._exact_cache.clear()
        self._prefix_trie = PrefixTrie()
        self.logger.info("缓存已清空")
    
    @property
    def stats(self) -> CacheStats:
        """获取缓存统计"""
        return self._exact_cache.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息字典"""
        stats = self._exact_cache.stats
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "total_entries": stats.total_entries,
            "total_size_mb": stats.total_size_mb,
            "evictions": stats.evictions,
        }


def cached_translate(
    translate_fn,
    cache: PromptCache,
    text: str,
    **kwargs
) -> str:
    """
    带缓存的翻译函数
    
    参数：
        translate_fn: 翻译函数
        cache: Prompt Cache
        text: 输入文本
        **kwargs: 翻译函数的其他参数
        
    返回：
        str: 翻译结果
    """
    # 尝试从缓存获取
    cached_result = cache.get(text)
    if cached_result:
        return cached_result
    
    # 执行翻译
    result = translate_fn(text, **kwargs)
    
    # 缓存结果
    cache.put(text, result)
    
    return result


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prompt Cache 工具"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行测试"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_CACHE_SIZE_MB,
        help=f"最大缓存大小 MB（默认: {DEFAULT_CACHE_SIZE_MB}）"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.test:
        # 简单测试
        cache = PromptCache(CacheConfig(max_size_mb=args.max_size))
        
        # 添加缓存
        cache.put("你好", "Hello")
        cache.put("世界", "World")
        cache.put("你好世界", "Hello World")
        
        # 测试获取
        print(f"'你好' -> {cache.get('你好')}")
        print(f"'世界' -> {cache.get('世界')}")
        print(f"'你好世界' -> {cache.get('你好世界')}")
        print(f"'不存在' -> {cache.get('不存在')}")
        
        # 打印统计
        print("\n" + str(cache.stats))


if __name__ == "__main__":
    main()
