#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本

功能说明：
    -训中英双向翻译模型
    -支持课程学习策略
    -支持混合精度训练 (BF16)
    - 自动保存检查点
    - 生成训练可视化图表

使用方法：
    python scripts/train.py [--direction DIR] [--epochs N] [--batch-size SIZE]

参数说明：
    --direction      :训方向 (zh2en, en2zh, both，默认: both)
    --epochs         :训轮数 (默认: 10)
    --batch-size     :批 (默认: 32)
    --learning-rate  :学率 (默认: 5e-5)
    --resume         : 从检查点恢复训练
    --data-path      : 数据集路径
    --output-dir     :模型输出目录

依赖：
    - Python 3.10+
    - PyTorch 2.0+ with CUDA
    -已完成数据准备

作者：NMT翻系统
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到 Python路
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.utils import setup_logger

# ============================================================================
#配置
# ============================================================================

DEFAULT_DATA_PATH = "outputs/data/dataset"
DEFAULT_OUTPUT_DIR = "outputs/models"

logger = setup_logger(__name__, level=logging.INFO)

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT模型训练流程")
    parser.add_argument("--direction", choices=["zh2en", "en2zh", "both"], 
                       default="both", help="训练方向")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--learning-rate", type=str, default="5e-5", help="学习率")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                       help="数据集路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="模型输出目录")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("NMT模型训练流程")
    
    #检查环境
    print_step("检查运行环境...")
    has_gpu = check_gpu()
    
    # 检查数据
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"数据集不存在: {data_path}")
        print("请先运行 python scripts/data_prep.py准备数据")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #模型配置
    models = {
        "zh2en": {
            "name": "models/Helsinki-NLP-opus-mt-zh-en",
            "output": output_dir / "zh2en"
        },
        "en2zh": {
            "name": "models/Helsinki-NLP-opus-mt-en-zh", 
            "output": output_dir / "en2zh"
        }
    }
    
    #执行训练
    results = {}
    
    if args.direction in ["both", "zh2en"]:
        config = models["zh2en"]
        results["zh2en"] = train_model(
            "zh2en", config["name"], data_path, config["output"]
        )
    
    if args.direction in ["both", "en2zh"]:
        config = models["en2zh"]
        results["en2zh"] = train_model(
            "en2zh", config["name"], data_path, config["output"]
        )
    
    #完成
    total_elapsed = time.time() - start_time
    print_header("训练流程完成")
    
    print("\n训练结果：")
    for direction, success in results.items():
        status = "成功" if success else "失败"
        color = "\033[32m" if success else "\033[31m"
        reset = "\033[0m"
        print(f"  - {direction}模型: {color}{status}{reset}")
    
    print("\n模型保存位置：")
    if args.direction in ["both", "zh2en"]:
        print(f"  - zh2en: {models['zh2en']['output']}")
    if args.direction in ["both", "en2zh"]:
        print(f"  - en2zh: {models['en2zh']['output']}")
    
    print(f"\n总耗时: {format_time(total_elapsed)}")
    print("\n下一步:运行 python scripts/compress.py进模型压缩")

# ============================================================================
#辅助函数
# ============================================================================

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_step(message):
    """打印步骤信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_gpu():
    """检查 GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_step(f"检测到 GPU: {gpu_name}")
            return True
        else:
            print("警告: 未检测到 GPU，将使用 CPU训（速度较慢）")
            return False
    except ImportError:
        print("警告: 无法检测 GPU 状态")
        return False

def train_model(direction: str, model_name: str, data_path: Path, 
                output_path: Path) -> bool:
    """训练单个模型"""
    print_header(f"训练 {direction} 模型")
    print_step(f"基础模型: {model_name}")
    print_step(f"数据集: {data_path}")
    print_step(f"输出目录: {output_path}")
    
    # 查找数据文件
    train_file = data_path / "train.jsonl"
    val_file = data_path / "val.jsonl"
    
    train_args = [
        "--direction", direction,
        "--model", model_name,
        "--train-data", str(train_file),
        "--output", str(output_path),
        "--config", "configs/training_config.yaml",
    ]
    
    if val_file.exists():
        train_args.extend(["--eval-data", str(val_file)])
    
    try:
        run_module("nmt.training.trainer", train_args)
        print_step(f"{direction} 模型训练完成")
        return True
    except Exception as e:
        logger.error(f"{direction}模型训练失败: {e}")
        return False

def run_module(module_name: str, args: list) -> None:
    """运行 Python 模块"""
    import subprocess
    
    cmd = [sys.executable, "-m", module_name] + args
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"模块 {module_name} 执行失败")

def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    main()