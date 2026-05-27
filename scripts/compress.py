#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型压缩脚本

功能说明：
    - 结构化剪枝（FFN 70%，注意力头 50%）
    - 动态量化（INT8/INT4）

使用方法：
    python scripts/compress.py [--model-path PATH] [--output-dir DIR] [--quantize MODE]

参数说明：
    --model-path     : 训练后模型路径 (默认: outputs/models)
    --output-dir     : 模型输出路径 (默认: outputs/compressed)
    --quantize       : 量化精度 (int8, int4，默认: int8)
    --skip-pruning   : 跳过剪枝步骤
    --direction      : 压缩方向 (both, zh2en, en2zh)

目标：
    - 模型体积≈100MB/方向
    - BLEU 下降 <0.3

作者：NMT翻译系统
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

DEFAULT_MODEL_PATH = "outputs/models"
DEFAULT_OUTPUT_DIR = "outputs/compressed"

logger = setup_logger(__name__, level=logging.INFO)

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT模型压缩流程")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                       help="训练后模型路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="压缩模型输出路径")
    parser.add_argument("--quantize", choices=["int8", "int4"], default="int8",
                       help="量化精度")
    parser.add_argument("--skip-pruning", action="store_true", help="跳过剪枝步骤")
    parser.add_argument("--direction", choices=["both", "zh2en", "en2zh"], 
                       default="both", help="压缩方向")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("NMT模型压缩流程")
    
    #检查输入路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"模型路径不存在: {model_path}")
        print("请先运行 python scripts/train.py训练模型")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #执行压缩
    results = {}
    
    # zh2en模型
    if args.direction in ["both", "zh2en"]:
        # 检查用户指定的路径是否直接包含模型
        if (model_path / "config.json").exists():
            zh2en_path = model_path
        else:
            # 否则查找子目录
            zh2en_path = model_path / "zh2en" / "best"
            if not zh2en_path.exists():
                zh2en_path = model_path / "zh2en" / "final"
            if not zh2en_path.exists():
                zh2en_path = model_path / "zh2en"
        
        if zh2en_path.exists() and (zh2en_path / "config.json").exists():
            results["zh2en"] = compress_model(
                zh2en_path, output_dir / "zh2en", "zh2en",
                args.quantize, args.skip_pruning
            )
        else:
            print(f"警告: zh2en模型不存在 (查找路径: {zh2en_path})，跳过")
    
    # en2zh模型
    if args.direction in ["both", "en2zh"]:
        # 检查用户指定的路径是否直接包含模型
        if (model_path / "config.json").exists():
            en2zh_path = model_path
        else:
            # 否则查找子目录
            en2zh_path = model_path / "en2zh" / "best"
            if not en2zh_path.exists():
                en2zh_path = model_path / "en2zh" / "final"
            if not en2zh_path.exists():
                en2zh_path = model_path / "en2zh"
        
        if en2zh_path.exists() and (en2zh_path / "config.json").exists():
            results["en2zh"] = compress_model(
                en2zh_path, output_dir / "en2zh", "en2zh",
                args.quantize, args.skip_pruning
            )
        else:
            print(f"警告: en2zh模型不存在 (查找路径: {en2zh_path})，跳过")
    
    #完成
    total_elapsed = time.time() - start_time
    print_header("压缩流程完成")
    
    print("\n压缩结果：")
    for direction, success in results.items():
        status = "成功" if success else "失败"
        color = "\033[32m" if success else "\033[31m"
        reset = "\033[0m"
        print(f"  - {direction}模型: {color}{status}{reset}")
    
    print(f"\n压缩模型位置: {output_dir}")
    print(f"总耗时: {format_time(total_elapsed)}")
    print("\n下一步:运行 python scripts/evaluate.py评估模型质量")

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

def compress_model(input_path: Path, output_path: Path, direction: str,
                  quantize: str, skip_pruning: bool) -> bool:
    """压缩单个模型"""
    print_header(f"压缩 {direction}模型")
    
    original_size = get_model_size(input_path)
    print_step(f"原始模型大小: {original_size:.2f}MB")
    
    pruned_path = output_path / "pruned"
    quantized_path = output_path / "quantized"
    
    current_path = input_path
    
    #步骤 1：结构化剪枝
    if not skip_pruning:
        print_step("执行结构化剪枝...")
        prune_args = [
            "--model", str(input_path),
            "--output", str(pruned_path),
            "--ffn-ratio", "0.7",
            "--head-ratio", "0.5"
        ]
        
        try:
            run_module("nmt.compression.pruning", prune_args)
            pruned_size = get_model_size(pruned_path)
            compression_ratio = original_size / pruned_size if pruned_size > 0 else 0
            print_step(f"剪枝后大小: {pruned_size:.2f}MB (压缩比 {compression_ratio:.2f}x)")
            current_path = pruned_path
        except Exception as e:
            logger.error(f"剪枝失败: {e}")
            return False
    
    #步骤 2：动态量化
    print_step(f"执行 {quantize} 量化...")
    quantize_args = [
        "--model", str(current_path),
        "--output", str(quantized_path),
        "--dtype", quantize
    ]
    
    try:
        run_module("nmt.compression.quantization", quantize_args)
        quantized_size = get_model_size(quantized_path)
        print_step(f"量化后大小: {quantized_size:.2f}MB")
    except Exception as e:
        logger.error(f"量化失败: {e}")
        return False
    
    #输出统计
    final_size = get_model_size(quantized_path)
    if original_size > 0 and final_size > 0:
        compression_ratio = original_size / final_size
        print(f"\n压缩统计 ({direction}):")
        print(f"  -原始大小: {original_size:.2f}MB")
        print(f"  -最终大小: {final_size:.2f}MB")
        print(f"  -压比: {compression_ratio:.2f}x")
    
    return True

def get_model_size(path: Path) -> float:
    """获取模型目录大小(MB)"""
    if not path.exists():
        return 0.0
    
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)  #为 MB

def run_module(module_name: str, args: list) -> None:
    """运行 Python模块"""
    import subprocess
    
    cmd = [sys.executable, "-m", module_name] + args
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"模块 {module_name}执行失败")

def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
#入点
# ============================================================================

if __name__ == "__main__":
    main()