#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消费级显卡适配模块

功能说明：
    - 自动检测 GPU 硬件规格（VRAM、计算能力）
    - 根据 VRAM 自动选择最优精度（FP16/INT8/INT4）
    - 动态显存管理和层卸载机制
    - 多平台 GPU 支持（NVIDIA/AMD/Intel）
    - 并发控制和资源监控

适配策略：
    - VRAM ≥ 12GB: FP16 全精度
    - VRAM ≥ 8GB:  INT8 量化
    - VRAM ≥ 6GB:  INT4 量化
    - VRAM < 6GB:  INT4 + CPU 卸载

依赖：
    - torch >= 2.0
    - pynvml (NVIDIA 显卡监控)
    - psutil (系统资源监控)

作者：NMT 翻译系统
"""

import os
import sys
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import torch

# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

# VRAM 阈值（GB）
VRAM_THRESHOLD_FP16 = 12.0   # FP16 全精度阈值
VRAM_THRESHOLD_INT8 = 8.0    # INT8 量化阈值
VRAM_THRESHOLD_INT4 = 6.0    # INT4 量化阈值

# 显存预留（用于其他进程）
VRAM_RESERVE_RATIO = 0.1     # 预留 10% 显存

# 最大并发请求数（不同 VRAM 等级）
MAX_CONCURRENT_REQUESTS = {
    'high': 16,     # VRAM ≥ 12GB
    'medium': 8,    # VRAM ≥ 8GB
    'low': 4,       # VRAM ≥ 6GB
    'minimal': 2,   # VRAM < 6GB
}

# 支持的 GPU 供应商
class GPUVendor(Enum):
    """GPU 供应商枚举"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"

# 精度模式
class PrecisionMode(Enum):
    """推理精度模式"""
    FP32 = "fp32"           # 全精度（主要用于 CPU）
    FP16 = "fp16"           # 半精度
    BF16 = "bf16"           # BF16（Ampere+ 架构）
    INT8 = "int8"           # INT8 量化
    INT4 = "int4"           # INT4 量化
    INT4_OFFLOAD = "int4_offload"  # INT4 + CPU 卸载


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class GPUInfo:
    """
    GPU 硬件信息
    
    属性：
        device_id: GPU 设备 ID
        name: GPU 名称
        vendor: GPU 供应商
        total_memory: 总显存（GB）
        free_memory: 可用显存（GB）
        compute_capability: 计算能力（NVIDIA 专用）
        driver_version: 驱动版本
        cuda_version: CUDA 版本（NVIDIA 专用）
        is_available: 是否可用
    """
    device_id: int = 0
    name: str = "Unknown"
    vendor: GPUVendor = GPUVendor.UNKNOWN
    total_memory: float = 0.0
    free_memory: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)
    driver_version: str = ""
    cuda_version: str = ""
    is_available: bool = False
    
    @property
    def available_memory(self) -> float:
        """获取实际可用显存（扣除预留）"""
        return self.free_memory * (1 - VRAM_RESERVE_RATIO)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"GPU({self.name}, "
            f"VRAM={self.total_memory:.1f}GB, "
            f"Free={self.free_memory:.1f}GB, "
            f"CC={self.compute_capability[0]}.{self.compute_capability[1]})"
        )


@dataclass
class AdapterConfig:
    """
    显卡适配配置
    
    属性：
        precision_mode: 推理精度模式
        max_batch_size: 最大批处理大小
        max_concurrent: 最大并发请求数
        enable_offload: 是否启用 CPU 卸载
        offload_layers: 卸载到 CPU 的层数
        memory_efficient: 是否启用内存优化模式
        use_flash_attention: 是否使用 Flash Attention
    """
    precision_mode: PrecisionMode = PrecisionMode.FP16
    max_batch_size: int = 32
    max_concurrent: int = 8
    enable_offload: bool = False
    offload_layers: int = 0
    memory_efficient: bool = False
    use_flash_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'precision_mode': self.precision_mode.value,
            'max_batch_size': self.max_batch_size,
            'max_concurrent': self.max_concurrent,
            'enable_offload': self.enable_offload,
            'offload_layers': self.offload_layers,
            'memory_efficient': self.memory_efficient,
            'use_flash_attention': self.use_flash_attention,
        }


# ============================================================================
# GPU 检测器
# ============================================================================

class GPUDetector:
    """
    GPU 硬件检测器
    
    功能：
        - 检测系统中的 GPU 设备
        - 获取 GPU 详细规格
        - 监控 GPU 使用状态
    
    示例：
        >>> detector = GPUDetector()
        >>> gpu_info = detector.detect_primary_gpu()
        >>> print(gpu_info)
        GPU(NVIDIA GeForce RTX 5090, VRAM=32.0GB, Free=28.5GB, CC=12.0)
    """
    
    def __init__(self):
        """初始化 GPU 检测器"""
        self._nvidia_available = self._check_nvidia()
        self._pynvml_initialized = False
        
    def _check_nvidia(self) -> bool:
        """检查 NVIDIA GPU 是否可用"""
        return torch.cuda.is_available()
    
    def _init_pynvml(self) -> bool:
        """初始化 pynvml 库"""
        if self._pynvml_initialized:
            return True
            
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_initialized = True
            return True
        except ImportError:
            logger.warning("pynvml 未安装，部分 GPU 监控功能不可用")
            return False
        except Exception as e:
            logger.warning(f"pynvml 初始化失败: {e}")
            return False
    
    def detect_all_gpus(self) -> List[GPUInfo]:
        """
        检测所有可用 GPU
        
        返回：
            List[GPUInfo]: GPU 信息列表
        """
        gpus = []
        
        # 检测 NVIDIA GPU
        if self._nvidia_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                gpu_info = self._get_nvidia_gpu_info(i)
                if gpu_info.is_available:
                    gpus.append(gpu_info)
        
        # TODO: 添加 AMD ROCm 和 Intel oneAPI 支持
        
        return gpus
    
    def detect_primary_gpu(self) -> Optional[GPUInfo]:
        """
        检测主 GPU（默认使用 CUDA_VISIBLE_DEVICES 中的第一个）
        
        返回：
            Optional[GPUInfo]: 主 GPU 信息，无可用 GPU 时返回 None
        """
        gpus = self.detect_all_gpus()
        return gpus[0] if gpus else None
    
    def _get_nvidia_gpu_info(self, device_id: int) -> GPUInfo:
        """
        获取 NVIDIA GPU 详细信息
        
        参数：
            device_id: GPU 设备 ID
            
        返回：
            GPUInfo: GPU 信息
        """
        try:
            # 基础信息（通过 PyTorch 获取）
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory / (1024 ** 3)  # 转换为 GB
            
            # 计算能力
            compute_capability = (props.major, props.minor)
            
            # 获取空闲显存
            torch.cuda.set_device(device_id)
            free_memory, _ = torch.cuda.mem_get_info(device_id)
            free_memory_gb = free_memory / (1024 ** 3)
            
            # 获取驱动和 CUDA 版本
            driver_version = ""
            cuda_version = f"{torch.version.cuda}" if torch.version.cuda else ""
            
            # 尝试通过 pynvml 获取更详细信息
            if self._init_pynvml():
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                except Exception as e:
                    logger.debug(f"pynvml 获取信息失败: {e}")
            
            return GPUInfo(
                device_id=device_id,
                name=props.name,
                vendor=GPUVendor.NVIDIA,
                total_memory=total_memory,
                free_memory=free_memory_gb,
                compute_capability=compute_capability,
                driver_version=driver_version,
                cuda_version=cuda_version,
                is_available=True
            )
            
        except Exception as e:
            logger.error(f"获取 GPU {device_id} 信息失败: {e}")
            return GPUInfo(device_id=device_id, is_available=False)
    
    def get_memory_usage(self, device_id: int = 0) -> Dict[str, float]:
        """
        获取 GPU 显存使用情况
        
        参数：
            device_id: GPU 设备 ID
            
        返回：
            Dict: 显存使用信息（GB）
        """
        if not self._nvidia_available:
            return {'total': 0, 'used': 0, 'free': 0}
        
        try:
            torch.cuda.set_device(device_id)
            free, total = torch.cuda.mem_get_info(device_id)
            total_gb = total / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            used_gb = total_gb - free_gb
            
            return {
                'total': round(total_gb, 2),
                'used': round(used_gb, 2),
                'free': round(free_gb, 2),
                'usage_percent': round((used_gb / total_gb) * 100, 1)
            }
        except Exception as e:
            logger.error(f"获取显存使用情况失败: {e}")
            return {'total': 0, 'used': 0, 'free': 0, 'usage_percent': 0}


# ============================================================================
# 精度选择器
# ============================================================================

class PrecisionSelector:
    """
    自动精度选择器
    
    功能：
        - 根据 GPU 规格自动选择最优精度
        - 支持手动覆盖精度设置
        - 提供精度配置建议
    
    示例：
        >>> selector = PrecisionSelector()
        >>> config = selector.select_config(gpu_info)
        >>> print(config.precision_mode)
        PrecisionMode.FP16
    """
    
    def __init__(self, force_precision: Optional[str] = None):
        """
        初始化精度选择器
        
        参数：
            force_precision: 强制使用的精度模式（可选）
        """
        self.force_precision = force_precision
        self._detector = GPUDetector()
    
    def select_precision(self, vram_gb: float) -> PrecisionMode:
        """
        根据 VRAM 选择精度模式
        
        参数：
            vram_gb: 可用显存（GB）
            
        返回：
            PrecisionMode: 推荐的精度模式
        """
        # 如果强制指定精度，直接返回
        if self.force_precision:
            try:
                return PrecisionMode(self.force_precision)
            except ValueError:
                logger.warning(f"无效的精度模式: {self.force_precision}")
        
        # 根据 VRAM 自动选择
        if vram_gb >= VRAM_THRESHOLD_FP16:
            return PrecisionMode.FP16
        elif vram_gb >= VRAM_THRESHOLD_INT8:
            return PrecisionMode.INT8
        elif vram_gb >= VRAM_THRESHOLD_INT4:
            return PrecisionMode.INT4
        else:
            return PrecisionMode.INT4_OFFLOAD
    
    def select_config(self, gpu_info: Optional[GPUInfo] = None) -> AdapterConfig:
        """
        根据 GPU 信息生成完整适配配置
        
        参数：
            gpu_info: GPU 信息（可选，为空时自动检测）
            
        返回：
            AdapterConfig: 适配配置
        """
        # 自动检测 GPU
        if gpu_info is None:
            gpu_info = self._detector.detect_primary_gpu()
        
        # 无 GPU 时使用 CPU 配置
        if gpu_info is None or not gpu_info.is_available:
            logger.info("未检测到可用 GPU，使用 CPU 模式")
            return AdapterConfig(
                precision_mode=PrecisionMode.FP32,
                max_batch_size=8,
                max_concurrent=2,
                enable_offload=False,
                memory_efficient=True
            )
        
        # 获取可用显存
        available_vram = gpu_info.available_memory
        logger.info(f"检测到 GPU: {gpu_info.name}, 可用显存: {available_vram:.1f}GB")
        
        # 选择精度模式
        precision_mode = self.select_precision(available_vram)
        
        # 确定并发级别
        if available_vram >= VRAM_THRESHOLD_FP16:
            concurrent_level = 'high'
            max_batch = 64
        elif available_vram >= VRAM_THRESHOLD_INT8:
            concurrent_level = 'medium'
            max_batch = 32
        elif available_vram >= VRAM_THRESHOLD_INT4:
            concurrent_level = 'low'
            max_batch = 16
        else:
            concurrent_level = 'minimal'
            max_batch = 8
        
        max_concurrent = MAX_CONCURRENT_REQUESTS[concurrent_level]
        
        # 确定是否需要卸载
        enable_offload = precision_mode == PrecisionMode.INT4_OFFLOAD
        offload_layers = 3 if enable_offload else 0  # 卸载 3 层到 CPU
        
        # 是否启用 Flash Attention（Ampere+ 架构支持）
        use_flash_attention = (
            gpu_info.vendor == GPUVendor.NVIDIA and
            gpu_info.compute_capability[0] >= 8
        )
        
        config = AdapterConfig(
            precision_mode=precision_mode,
            max_batch_size=max_batch,
            max_concurrent=max_concurrent,
            enable_offload=enable_offload,
            offload_layers=offload_layers,
            memory_efficient=available_vram < VRAM_THRESHOLD_INT8,
            use_flash_attention=use_flash_attention
        )
        
        logger.info(f"适配配置: {config.to_dict()}")
        return config
    
    def get_recommendation(self, gpu_info: Optional[GPUInfo] = None) -> str:
        """
        获取配置建议文本
        
        参数：
            gpu_info: GPU 信息
            
        返回：
            str: 配置建议
        """
        if gpu_info is None:
            gpu_info = self._detector.detect_primary_gpu()
        
        config = self.select_config(gpu_info)
        
        recommendations = []
        recommendations.append(f"推荐精度: {config.precision_mode.value}")
        recommendations.append(f"最大批大小: {config.max_batch_size}")
        recommendations.append(f"最大并发: {config.max_concurrent}")
        
        if config.enable_offload:
            recommendations.append(f"CPU 卸载: 启用 ({config.offload_layers} 层)")
        
        if config.use_flash_attention:
            recommendations.append("Flash Attention: 启用")
        
        return "\n".join(recommendations)


# ============================================================================
# 显存管理器
# ============================================================================

class MemoryManager:
    """
    GPU 显存管理器
    
    功能：
        - 动态显存分配和释放
        - 显存使用监控
        - 内存泄漏检测
        - 自动垃圾回收
    
    示例：
        >>> manager = MemoryManager()
        >>> manager.optimize_memory()
        >>> stats = manager.get_stats()
    """
    
    def __init__(self, device_id: int = 0):
        """
        初始化显存管理器
        
        参数：
            device_id: GPU 设备 ID
        """
        self.device_id = device_id
        self._detector = GPUDetector()
        self._lock = threading.Lock()
        
    def optimize_memory(self) -> None:
        """
        优化显存使用
        
        执行操作：
            - 清空 CUDA 缓存
            - 触发 Python 垃圾回收
            - 同步 CUDA 流
        """
        with self._lock:
            try:
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.device_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                logger.debug("显存优化完成")
            except Exception as e:
                logger.error(f"显存优化失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取显存统计信息
        
        返回：
            Dict: 显存统计
        """
        memory_usage = self._detector.get_memory_usage(self.device_id)
        
        # 获取 PyTorch 分配的显存
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        else:
            allocated = 0
            reserved = 0
        
        return {
            **memory_usage,
            'pytorch_allocated': round(allocated, 2),
            'pytorch_reserved': round(reserved, 2),
        }
    
    def check_memory_available(self, required_gb: float) -> bool:
        """
        检查是否有足够显存
        
        参数：
            required_gb: 需要的显存（GB）
            
        返回：
            bool: 是否有足够显存
        """
        stats = self.get_stats()
        return stats.get('free', 0) >= required_gb
    
    def estimate_batch_size(self, 
                           sample_memory_mb: float,
                           target_utilization: float = 0.8) -> int:
        """
        估算最优批大小
        
        参数：
            sample_memory_mb: 单个样本占用显存（MB）
            target_utilization: 目标显存利用率
            
        返回：
            int: 推荐的批大小
        """
        stats = self.get_stats()
        free_mb = stats.get('free', 0) * 1024  # 转换为 MB
        target_mb = free_mb * target_utilization
        
        batch_size = int(target_mb / sample_memory_mb) if sample_memory_mb > 0 else 1
        return max(1, batch_size)


# ============================================================================
# 并发控制器
# ============================================================================

class ConcurrencyController:
    """
    并发请求控制器
    
    功能：
        - 限制同时进行的推理请求数
        - 动态调整并发数
        - 请求队列管理
    
    示例：
        >>> controller = ConcurrencyController(max_concurrent=8)
        >>> async with controller.acquire():
        >>>     result = await model.translate(text)
    """
    
    def __init__(self, max_concurrent: int = 8):
        """
        初始化并发控制器
        
        参数：
            max_concurrent: 最大并发数
        """
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'rejected_requests': 0,
            'peak_concurrent': 0,
        }
    
    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        获取执行许可
        
        参数：
            blocking: 是否阻塞等待
            timeout: 等待超时时间（秒）
            
        返回：
            bool: 是否成功获取
        """
        acquired = self._semaphore.acquire(blocking=blocking, timeout=timeout)
        
        with self._lock:
            self._stats['total_requests'] += 1
            if acquired:
                self._active_count += 1
                self._stats['peak_concurrent'] = max(
                    self._stats['peak_concurrent'],
                    self._active_count
                )
            else:
                self._stats['rejected_requests'] += 1
        
        return acquired
    
    def release(self) -> None:
        """释放执行许可"""
        self._semaphore.release()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
    
    @property
    def active_count(self) -> int:
        """当前活跃请求数"""
        return self._active_count
    
    @property
    def available_slots(self) -> int:
        """可用槽位数"""
        return self.max_concurrent - self._active_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats,
                'current_active': self._active_count,
                'max_concurrent': self.max_concurrent,
            }


# ============================================================================
# 主适配器类
# ============================================================================

class GPUAdapter:
    """
    GPU 适配器（主类）
    
    功能：
        - 集成 GPU 检测、精度选择、显存管理
        - 提供统一的配置接口
        - 自动适配不同级别的 GPU
    
    示例：
        >>> adapter = GPUAdapter()
        >>> config = adapter.get_config()
        >>> print(f"使用精度: {config.precision_mode.value}")
        
        >>> # 获取 PyTorch 数据类型
        >>> dtype = adapter.get_torch_dtype()
        >>> model = model.to(dtype)
    """
    
    _instance: Optional['GPUAdapter'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, force_precision: Optional[str] = None):
        """
        初始化 GPU 适配器
        
        参数：
            force_precision: 强制使用的精度模式（可选）
        """
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._detector = GPUDetector()
        self._selector = PrecisionSelector(force_precision)
        self._gpu_info: Optional[GPUInfo] = None
        self._config: Optional[AdapterConfig] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._concurrency_controller: Optional[ConcurrencyController] = None
        
        # 执行初始化
        self._initialize()
        self._initialized = True
    
    def _initialize(self) -> None:
        """执行初始化"""
        # 检测 GPU
        self._gpu_info = self._detector.detect_primary_gpu()
        
        # 生成配置
        self._config = self._selector.select_config(self._gpu_info)
        
        # 初始化管理器
        device_id = self._gpu_info.device_id if self._gpu_info else 0
        self._memory_manager = MemoryManager(device_id)
        self._concurrency_controller = ConcurrencyController(
            self._config.max_concurrent
        )
        
        # 输出配置信息
        self._log_config()
    
    def _log_config(self) -> None:
        """输出配置信息"""
        logger.info("=" * 60)
        logger.info("GPU 适配器配置")
        logger.info("=" * 60)
        
        if self._gpu_info and self._gpu_info.is_available:
            logger.info(f"GPU: {self._gpu_info.name}")
            logger.info(f"显存: {self._gpu_info.total_memory:.1f}GB "
                       f"(可用: {self._gpu_info.free_memory:.1f}GB)")
            logger.info(f"计算能力: {self._gpu_info.compute_capability[0]}."
                       f"{self._gpu_info.compute_capability[1]}")
        else:
            logger.info("GPU: 不可用（使用 CPU 模式）")
        
        logger.info(f"精度模式: {self._config.precision_mode.value}")
        logger.info(f"最大批大小: {self._config.max_batch_size}")
        logger.info(f"最大并发: {self._config.max_concurrent}")
        logger.info(f"CPU 卸载: {'启用' if self._config.enable_offload else '禁用'}")
        logger.info("=" * 60)
    
    @property
    def gpu_info(self) -> Optional[GPUInfo]:
        """获取 GPU 信息"""
        return self._gpu_info
    
    @property
    def config(self) -> AdapterConfig:
        """获取适配配置"""
        return self._config
    
    def get_config(self) -> AdapterConfig:
        """获取适配配置（方法形式）"""
        return self._config
    
    def get_device(self) -> torch.device:
        """
        获取 PyTorch 设备
        
        返回：
            torch.device: 计算设备
        """
        if self._gpu_info and self._gpu_info.is_available:
            return torch.device(f"cuda:{self._gpu_info.device_id}")
        return torch.device("cpu")
    
    def get_torch_dtype(self) -> torch.dtype:
        """
        获取 PyTorch 数据类型
        
        返回：
            torch.dtype: 数据类型
        """
        precision_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.INT8: torch.float16,  # INT8 运行时使用 FP16
            PrecisionMode.INT4: torch.float16,  # INT4 运行时使用 FP16
            PrecisionMode.INT4_OFFLOAD: torch.float16,
        }
        return precision_map.get(self._config.precision_mode, torch.float32)
    
    def optimize_memory(self) -> None:
        """优化显存"""
        if self._memory_manager:
            self._memory_manager.optimize_memory()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取显存统计"""
        if self._memory_manager:
            return self._memory_manager.get_stats()
        return {}
    
    def acquire_slot(self, blocking: bool = True, timeout: float = None) -> bool:
        """获取执行槽位"""
        if self._concurrency_controller:
            return self._concurrency_controller.acquire(blocking, timeout)
        return True
    
    def release_slot(self) -> None:
        """释放执行槽位"""
        if self._concurrency_controller:
            self._concurrency_controller.release()
    
    def get_concurrency_stats(self) -> Dict[str, Any]:
        """获取并发统计"""
        if self._concurrency_controller:
            return self._concurrency_controller.get_stats()
        return {}


# ============================================================================
# 便捷函数
# ============================================================================

def get_adapter(force_precision: Optional[str] = None) -> GPUAdapter:
    """
    获取 GPU 适配器实例
    
    参数：
        force_precision: 强制精度模式
        
    返回：
        GPUAdapter: 适配器实例
    """
    return GPUAdapter(force_precision)


def detect_gpu() -> Optional[GPUInfo]:
    """
    检测主 GPU
    
    返回：
        Optional[GPUInfo]: GPU 信息
    """
    detector = GPUDetector()
    return detector.detect_primary_gpu()


def select_precision(vram_gb: float = None) -> PrecisionMode:
    """
    选择精度模式
    
    参数：
        vram_gb: 可用显存（GB），为空时自动检测
        
    返回：
        PrecisionMode: 精度模式
    """
    if vram_gb is None:
        gpu_info = detect_gpu()
        vram_gb = gpu_info.available_memory if gpu_info else 0
    
    selector = PrecisionSelector()
    return selector.select_precision(vram_gb)


def print_gpu_info() -> None:
    """打印 GPU 信息"""
    detector = GPUDetector()
    gpus = detector.detect_all_gpus()
    
    if not gpus:
        print("未检测到可用 GPU")
        return
    
    print("\n" + "=" * 60)
    print("GPU 信息")
    print("=" * 60)
    
    for gpu in gpus:
        print(f"\n[GPU {gpu.device_id}] {gpu.name}")
        print(f"  供应商: {gpu.vendor.value}")
        print(f"  总显存: {gpu.total_memory:.1f} GB")
        print(f"  可用显存: {gpu.free_memory:.1f} GB")
        print(f"  计算能力: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        if gpu.driver_version:
            print(f"  驱动版本: {gpu.driver_version}")
        if gpu.cuda_version:
            print(f"  CUDA 版本: {gpu.cuda_version}")
    
    print("\n" + "=" * 60)
    
    # 打印推荐配置
    selector = PrecisionSelector()
    print("\n推荐配置:")
    print(selector.get_recommendation(gpus[0] if gpus else None))


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    # 枚举类
    'GPUVendor',
    'PrecisionMode',
    
    # 数据类
    'GPUInfo',
    'AdapterConfig',
    
    # 核心类
    'GPUDetector',
    'PrecisionSelector',
    'MemoryManager',
    'ConcurrencyController',
    'GPUAdapter',
    
    # 便捷函数
    'get_adapter',
    'detect_gpu',
    'select_precision',
    'print_gpu_info',
]


# ============================================================================
# 主函数（测试用）
# ============================================================================

if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 打印 GPU 信息
    print_gpu_info()
    
    # 测试适配器
    print("\n测试 GPU 适配器:")
    adapter = get_adapter()
    print(f"设备: {adapter.get_device()}")
    print(f"数据类型: {adapter.get_torch_dtype()}")
    print(f"配置: {adapter.config.to_dict()}")
