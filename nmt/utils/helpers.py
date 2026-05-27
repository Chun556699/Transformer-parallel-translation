"""
工具函数模块

功能说明：
    提供通用的工具函数，包括日志设置、随机种子、设备检测等。

依赖：
    - torch
    - logging
"""

import os
import sys
import random
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
import torch


def setup_logger(
    name: str = "nmt",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志记录器
    
    参数：
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    返回：
        logging.Logger: 配置好的日志器
    """
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可复现性
    
    参数：
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 运算的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    """
    获取可用的计算设备
    
    参数：
        force_cpu: 是否强制使用 CPU
    
    返回：
        torch.device: 计算设备
    """
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        # 获取 GPU 信息
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"使用 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
    else:
        print("CUDA 不可用，使用 CPU")
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    参数：
        seconds: 秒数
    
    返回：
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def get_project_root() -> Path:
    """
    获取项目根目录
    
    返回：
        Path: 项目根目录路径
    """
    # 从当前文件向上查找包含 requirements.txt 的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "requirements.txt").exists():
            return parent
    # 默认返回当前工作目录
    return Path.cwd()


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在则创建
    
    参数：
        path: 目录路径
    
    返回：
        Path: 目录路径对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> dict:
    """
    统计模型参数数量
    
    参数：
        model: PyTorch 模型
    
    返回：
        dict: 参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "total_mb": total_params * 4 / (1024**2),  # 假设 FP32
    }


def get_gpu_memory_info() -> dict:
    """
    获取 GPU 显存信息
    
    返回：
        dict: 显存信息（已用、可用、总量，单位 GB）
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - reserved
    
    return {
        "available": True,
        "total_gb": total,
        "reserved_gb": reserved,
        "allocated_gb": allocated,
        "free_gb": free,
    }
