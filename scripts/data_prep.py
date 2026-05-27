#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本

功能说明：
    - 数据清洗（去重、标准化、过滤）
    -高质量数据筛选（LaBSE 语义相似度）
    - 数据分词处理
    -训练/验证/测试集划分
    - 生成数据可视化图表

使用方法：
    python scripts/data_prep.py [--data-path PATH] [--output-dir DIR] [--skip-filter]

参数说明：
    --data-path     :数据路径 (默认: mydata/translation2019zh)
    --output-dir    : 输出目录 (默认: outputs/data)
    --skip-filter   :跳过高质量筛选（加快处理速度）
    --max-samples   : 最大处理样本数 (默认: 0表示全量)
    --workers       :并行工作进程数 (默认: 4)

依赖：
    - Python 3.10+
    -已安装项目依赖 (pip install -r requirements.txt)

作者：NMT翻系统
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# 添加项目根目录到 Python路
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.utils import setup_logger

# ============================================================================
#配置
# ============================================================================

# 默认路径配置
DEFAULT_DATA_PATH = "mydata/translation2019zh"
DEFAULT_OUTPUT_DIR = "outputs/data"

# 日志配置
logger = setup_logger(__name__, level=logging.INFO)

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT 数据准备流程")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                       help="原始数据路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="输出目录")
    parser.add_argument("--skip-filter", action="store_true",
                       help="跳过高质量筛选")
    parser.add_argument("--high-quality-sampling", action="store_true",
                       help="启用高质量数据抽样（从520万中抽取30万）")
    parser.add_argument("--target-samples", type=int, default=300000,
                       help="目标高质量样本数 (默认: 300000)")
    parser.add_argument("--quality-threshold", type=float, default=0.75,
                       help="高质量阈值 (默认: 0.75)")
    parser.add_argument("--max-samples", type=int, default=0,
                       help="最大处理样本数 (0=全量)")
    parser.add_argument("--workers", type=int, default=4,
                       help="并行工作进程数")
    
    args = parser.parse_args()
    
    #记录开始时间
    start_time = time.time()
    
    print_header("NMT 数据准备流程")
    
    # 检查数据路径
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"数据路径不存在: {data_path}")
        sys.exit(1)
        
    # 如果是目录，自动查找数据文件
    if data_path.is_dir():
        # 优先查找 train 文件
        candidates = [
            data_path / "translation2019zh_train.json",
            data_path / "train.jsonl",
            data_path / "train.json",
        ]
        # 也查找目录下的任意 json/jsonl 文件
        for ext in ["*.json", "*.jsonl"]:
            for f in data_path.glob(ext):
                if f not in candidates:
                    candidates.append(f)
            
        found = False
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                print_step(f"自动选择数据文件: {data_path}")
                found = True
                break
            
        if not found:
            logger.error(f"目录中未找到数据文件: {data_path}")
            sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_step("检查运行环境...")
    check_environment()
    
    # ============================================================================
    # 步骤 1：高质量数据抽样（跳过数据清洗，数据源质量已很好）
    # ============================================================================
    
    if not args.skip_filter:
        if args.high_quality_sampling:
            print_header("步骤 1/3: 高质量数据抽样")
            
            sampled_dir = output_dir / "sampled"
            sampled_dir.mkdir(parents=True, exist_ok=True)
            sampled_file = sampled_dir / "sampled_data.jsonl"
            sampling_args = [
                "--input", str(data_path),  # 直接从原始数据抽样
                "--output", str(sampled_file),
                "--target-samples", str(args.target_samples),
                "--quality-threshold", str(args.quality_threshold),
                "--labse-model", "sentence-transformers/LaBSE"  # 使用LaBSE模型计算语义相似度
            ]
            
            print_step(f"从大规模数据中抽取 {args.target_samples:,} 高质量样本...")
            print_step(f"质量阈值: {args.quality_threshold} (高于标准0.7，确保Helsinki-NLP性能)")
            run_module("nmt.data.high_quality_sampler", sampling_args)
            processed_path = sampled_file
        else:
            # 标准高质量筛选模式
            print_header("步骤 1/3: 高质量数据筛选")
            
            filtered_dir = output_dir / "filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            filtered_file = filtered_dir / "filtered_data.jsonl"
            filter_args = [
                "--input", str(data_path),  # 直接从原始数据筛选
                "--output", str(filtered_file),
                "--min-similarity", "0.7",
                "--batch-size", "32"
            ]
            
            print_step("执行 LaBSE 语义筛选...")
            run_module("nmt.data.data_filter", filter_args)
            processed_path = filtered_file
    else:
        print_header("步骤 1/3: 跳过高质量处理")
        print_step("使用原始数据继续")
        processed_path = data_path
    
    # ============================================================================
    # 步骤 2：数据集构建
    # ============================================================================
    
    print_header("步骤 2/3: 数据集构建")
    
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_args = [
        "--input", str(processed_path),
        "--output", str(dataset_dir),
        "--split",
        "--train-ratio", "0.9",
        "--val-ratio", "0.05",
        "--test-ratio", "0.05"
    ]
    
    print_step("构建训练数据集...")
    run_module("nmt.data.dataset", dataset_args)
    
    # ============================================================================
    # 步骤 3：数据可视化
    # ============================================================================
    
    print_header("步骤 3/3: 生成可视化图表")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找数据集文件
    train_file = dataset_dir / "train.jsonl"
    if train_file.exists():
        viz_args = [
            "--data", str(train_file),
            "--output", str(viz_dir),
            "--font", "SourceHanSansSC-Regular-2.otf"
        ]
        
        print_step("生成数据分布图表...")
        try:
            run_module("nmt.visualization.data_viz", viz_args)
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
            print("警告:可视化生成失败，但不影响后续训练")
    else:
        print_step("跳过可视化: 训练数据文件不存在")
    
    # ============================================================================
    #完成
    # ============================================================================
    
    elapsed = time.time() - start_time
    print_header("数据准备完成")
    
    print("\n处理结果：")
    if not args.skip_filter:
        if args.high_quality_sampling:
            print(f"  - 抽样数据: {sampled_dir}")
        else:
            print(f"  - 筛选数据: {filtered_dir}")
    print(f"  - 训练数据: {dataset_dir}")
    print(f"  - 可视化图表: {viz_dir}")
    print(f"\n总耗时: {format_time(elapsed)}")
    print("\n下一步: 运行 python scripts/train.py 开始训练")

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

def check_environment():
    """检查运行环境"""
    try:
        import torch
        print_step(f"Python版本: {sys.version.split()[0]}")
        print_step(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print_step(f"CUDA可用: 是 ({torch.cuda.get_device_name(0)})")
        else:
            print_step("CUDA可用:否")
    except ImportError as e:
        logger.error(f"环境检查失败: {e}")
        sys.exit(1)

def run_module(module_name, args):
    """运行 Python模块"""
    import subprocess
    
    cmd = [sys.executable, "-m", module_name] + args
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"模块 {module_name}执行失败")

def format_time(seconds):
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