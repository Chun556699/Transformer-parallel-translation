#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型量化脚本

功能说明：
    - FP16半精度量化（GPU加速，推荐）
    - INT8动态量化（CPU部署）
    - 支持单独量化zh2en或en2zh模型
    - 显示压缩前后对比

使用方法：
    python scripts/quantize.py                    # FP16量化所有best模型（默认）
    python scripts/quantize.py --dtype fp16       # FP16半精度量化（GPU可用）
    python scripts/quantize.py --dtype int8       # INT8动态量化（仅CPU）
    python scripts/quantize.py --direction zh2en  # 只量化zh2en

量化效果：
    - FP16: 模型体积减少约50%，GPU可用，精度损失极小
    - INT8: 模型体积减少约75%，仅CPU，精度损失极小

作者：NMT翻译系统
"""

import os
import sys
import argparse
import logging
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import MarianMTModel, MarianTokenizer

# ============================================================================
# 配置
# ============================================================================

# 默认模型路径配置
DEFAULT_MODEL_BASE = "outputs/models"
DEFAULT_OUTPUT_BASE = "outputs/quantized"

logger = logging.getLogger(__name__)


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(message: str):
    """打印步骤信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_model_size(path: Path) -> float:
    """获取模型目录大小(MB)"""
    if not path.exists():
        return 0.0
    
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)  # 转换为 MB


def get_model_param_size(model: torch.nn.Module) -> float:
    """计算模型参数大小(MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 ** 2)


def quantize_model_fp16(
    model_path: Path,
    output_path: Path
) -> Dict:
    """
    FP16半精度量化（GPU可用）
    
    参数：
        model_path: 输入模型路径
        output_path: 输出模型路径
        
    返回：
        Dict: 量化统计信息
    """
    print_step(f"加载模型: {model_path}")
    
    # 加载原始模型
    tokenizer = MarianTokenizer.from_pretrained(str(model_path))
    model = MarianMTModel.from_pretrained(str(model_path))
    model.eval()
    
    # 计算原始大小
    original_size = get_model_param_size(model)
    print_step(f"原始模型参数大小: {original_size:.2f} MB")
    
    # 执行FP16量化
    print_step("执行 FP16 半精度量化...")
    
    start_time = time.time()
    
    # 将模型转换为半精度
    model = model.half()
    
    quantize_time = time.time() - start_time
    
    # 计算量化后大小
    quantized_size = get_model_param_size(model)
    
    # 计算压缩比
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    stats = {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "dtype": "fp16",
        "quantize_time_sec": quantize_time,
        "gpu_compatible": True
    }
    
    print_step(f"量化后模型参数大小: {quantized_size:.2f} MB")
    print_step(f"压缩比: {compression_ratio:.2f}x")
    
    # 保存量化模型
    print_step(f"保存量化模型到: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型（FP16格式）
    model.save_pretrained(str(output_path))
    
    # 保存分词器
    tokenizer.save_pretrained(str(output_path))
    
    # 保存量化统计
    with open(output_path / "quantization_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 计算实际磁盘大小
    actual_size = get_model_size(output_path)
    stats["actual_disk_size_mb"] = actual_size
    
    return stats


def quantize_model_int8(
    model_path: Path,
    output_path: Path
) -> Dict:
    """
    INT8动态量化（仅CPU）
    
    参数：
        model_path: 输入模型路径
        output_path: 输出模型路径
        
    返回：
        Dict: 量化统计信息
    """
    print_step(f"加载模型: {model_path}")
    
    # 加载原始模型
    tokenizer = MarianTokenizer.from_pretrained(str(model_path))
    model = MarianMTModel.from_pretrained(str(model_path))
    model.eval()
    
    # 计算原始大小
    original_size = get_model_param_size(model)
    print_step(f"原始模型参数大小: {original_size:.2f} MB")
    
    # 执行动态量化
    print_step("执行 INT8 动态量化...")
    
    start_time = time.time()
    
    # 使用 PyTorch 动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 量化 Linear 层
        dtype=torch.qint8
    )
    
    quantize_time = time.time() - start_time
    
    # 计算量化后大小
    quantized_size = get_model_param_size(quantized_model)
    
    # 统计量化层数
    quantized_layers = 0
    for name, module in quantized_model.named_modules():
        if hasattr(module, '_packed_params'):
            quantized_layers += 1
    
    # 计算压缩比
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    stats = {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": compression_ratio,
        "quantized_layers": quantized_layers,
        "dtype": "int8",
        "quantize_time_sec": quantize_time,
        "gpu_compatible": False
    }
    
    print_step(f"量化后模型参数大小: {quantized_size:.2f} MB")
    print_step(f"压缩比: {compression_ratio:.2f}x")
    print_step(f"量化层数: {quantized_layers}")
    
    # 保存量化模型
    print_step(f"保存量化模型到: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重
    torch.save(quantized_model.state_dict(), output_path / "pytorch_model.bin")
    
    # 复制配置文件和词表文件
    for file_name in ["config.json", "tokenizer_config.json", "source.spm", 
                      "target.spm", "vocab.json", "special_tokens_map.json",
                      "generation_config.json", "sentencepiece.bpe.model"]:
        src_file = model_path / file_name
        if src_file.exists():
            shutil.copy2(src_file, output_path / file_name)
    
    # 保存分词器
    tokenizer.save_pretrained(str(output_path))
    
    # 保存量化统计
    with open(output_path / "quantization_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 计算实际磁盘大小
    actual_size = get_model_size(output_path)
    stats["actual_disk_size_mb"] = actual_size
    
    return stats


def quantize_model(
    model_path: Path,
    output_path: Path,
    dtype: str = "fp16"
) -> Dict:
    """
    对模型进行量化
    
    参数：
        model_path: 输入模型路径
        output_path: 输出模型路径
        dtype: 量化类型 (fp16, int8)
        
    返回：
        Dict: 量化统计信息
    """
    if dtype == "fp16":
        return quantize_model_fp16(model_path, output_path)
    else:
        return quantize_model_int8(model_path, output_path)


def find_best_model(base_path: Path, direction: str) -> Optional[Path]:
    """查找最佳模型路径"""
    candidates = [
        base_path / direction / "best",
        base_path / direction / "final",
        base_path / direction,
    ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            return candidate
    
    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NMT模型量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/quantize.py                         # FP16量化所有best模型（默认）
    python scripts/quantize.py --dtype fp16            # FP16半精度量化（GPU可用）
    python scripts/quantize.py --dtype int8            # INT8动态量化（仅CPU）
    python scripts/quantize.py --direction zh2en       # 只量化zh2en模型
    python scripts/quantize.py --direction en2zh       # 只量化en2zh模型
        """
    )
    
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="输入模型路径 (默认自动查找 outputs/models/*/best)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认 outputs/quantized/*)"
    )
    parser.add_argument(
        "--direction", choices=["both", "zh2en", "en2zh"], default="both",
        help="量化方向 (默认: both)"
    )
    parser.add_argument(
        "--dtype", choices=["fp16", "int8"], default="fp16",
        help="量化精度: fp16=GPU可用, int8=仅CPU (默认: fp16)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    start_time = time.time()
    
    print_header("NMT模型量化工具")
    print_step(f"量化类型: {args.dtype.upper()}")
    if args.dtype == "fp16":
        print_step("FP16模式: GPU加速可用")
    else:
        print_step("INT8模式: 仅CPU推理")
    
    results = {}
    
    # 如果指定了具体模型路径
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"模型路径不存在: {model_path}")
            sys.exit(1)
        
        output_path = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_BASE) / model_path.name
        
        print_step(f"量化模型: {model_path}")
        stats = quantize_model(model_path, output_path, args.dtype)
        results[model_path.name] = stats
        
    else:
        # 自动查找并量化模型
        base_path = Path(DEFAULT_MODEL_BASE)
        output_base = Path(DEFAULT_OUTPUT_BASE)
        
        directions = ["zh2en", "en2zh"] if args.direction == "both" else [args.direction]
        
        for direction in directions:
            print_header(f"量化 {direction} 模型")
            
            # 查找模型
            model_path = find_best_model(base_path, direction)
            
            if model_path is None:
                print(f"警告: 未找到 {direction} 模型，跳过")
                print(f"  查找路径: {base_path / direction / 'best'}")
                print(f"  查找路径: {base_path / direction / 'final'}")
                continue
            
            # 确定输出路径
            if "best" in str(model_path):
                output_path = output_base / direction / "quantized"
            else:
                output_path = output_base / direction
            
            print_step(f"找到模型: {model_path}")
            
            # 记录原始磁盘大小
            original_disk_size = get_model_size(model_path)
            print_step(f"原始模型磁盘大小: {original_disk_size:.2f} MB")
            
            # 执行量化
            stats = quantize_model(model_path, output_path, args.dtype)
            stats["original_disk_size_mb"] = original_disk_size
            results[direction] = stats
    
    # 打印汇总
    print_header("量化完成")
    
    if not results:
        print("没有模型被量化")
        sys.exit(1)
    
    print("\n量化统计汇总:")
    print("-" * 70)
    
    total_original = 0
    total_quantized = 0
    
    for name, stats in results.items():
        print(f"\n【{name}】")
        print(f"  原始大小: {stats.get('original_disk_size_mb', stats['original_size_mb']):.2f} MB")
        print(f"  量化后大小: {stats.get('actual_disk_size_mb', stats['quantized_size_mb']):.2f} MB")
        print(f"  压缩比: {stats['compression_ratio']:.2f}x")
        if stats.get('quantized_layers'):
            print(f"  量化层数: {stats['quantized_layers']}")
        if stats.get('gpu_compatible'):
            print(f"  GPU兼容: 是")
        
        total_original += stats.get('original_disk_size_mb', stats['original_size_mb'])
        total_quantized += stats.get('actual_disk_size_mb', stats['quantized_size_mb'])
    
    print("\n" + "-" * 70)
    print(f"总计:")
    print(f"  原始大小: {total_original:.2f} MB")
    print(f"  量化后大小: {total_quantized:.2f} MB")
    print(f"  总压缩比: {total_original / total_quantized:.2f}x" if total_quantized > 0 else "")
    
    total_elapsed = time.time() - start_time
    print(f"\n总耗时: {format_time(total_elapsed)}")
    
    print(f"\n量化模型保存位置: {DEFAULT_OUTPUT_BASE}")
    print("\n下一步: 运行 python scripts/evaluate.py --model-path outputs/quantized 评估量化模型")


if __name__ == "__main__":
    main()
