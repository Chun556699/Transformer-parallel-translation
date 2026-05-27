#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 (v2.0 - 学术级可视化增强版)

功能说明：
    - 多模型对比评估 (Original vs Trained vs Quantized)
    - 多元指标评测 (BLEU, chrF++, TER, COMET)
    - 自动生成学术级可视化图表 (Bar, Scatter, Radar)
    - 支持 WMT22/WMT19/JSON/CSV 多种数据源

使用方法：
    python scripts/evaluate.py [--model-path PATH] [--test-set PATH]

参数说明：
    --output-dir     : 评估结果输出路径 (默认: outputs/evaluation)
    --direction      : 评估方向 (both, zh2en, en2zh)
    --max-samples    : 最大样本数 (默认: 100, 调试用)
    --skip-comet     : 跳过 COMET 评估
    --only-quantized : 仅评估量化模型

作者：NMT翻译系统
"""

import os
import sys
import argparse
import logging
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 添加项目根目录到 Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.utils import setup_logger

# 尝试导入可视化模块
try:
    from scripts.visualization import (
        set_academic_style, 
        plot_bar_comparison, 
        plot_quality_latency_scatter, 
        plot_radar_chart,
        plot_model_efficiency_ranking,
        plot_size_comparison
    )
except ImportError:
    # 如果作为脚本直接运行，尝试相对导入
    try:
        sys.path.append(str(Path(__file__).parent))
        from visualization import (
            set_academic_style, 
            plot_bar_comparison, 
            plot_quality_latency_scatter, 
            plot_radar_chart,
            plot_model_efficiency_ranking,
            plot_size_comparison
        )

    except ImportError:
        print("警告: 未找到 visualization 模块，将跳过部分高级可视化功能。")
        set_academic_style = None
        plot_model_efficiency_ranking = None


# ============================================================================
# 配置
# ============================================================================

DEFAULT_OUTPUT_DIR = "outputs/evaluation"

# 评估数据源配置
EVAL_DATA_SOURCES = [
    {
        "name": "WMT19",
        "path": "wmt19",
        "type": "sacrebleu_wmt",
        "description": "WMT19标准测试集"
    },
    {
        "name": "WMT22",
        "path": "wmt22",
        "type": "sacrebleu_wmt",
        "description": "WMT22标准测试集"
    },
]



# 模型路径模板
MODEL_SEARCH_PATHS = {
    "zh2en": [
        ("Original", "models/Helsinki-NLP-opus-mt-zh-en"),
        ("Trained-Best", "outputs/models/zh2en/best"),
        ("Quantized", "outputs/quantized/zh2en/quantized"),
    ],
    "en2zh": [
        ("Original", "models/Helsinki-NLP-opus-mt-en-zh"),
        ("Trained-Best", "outputs/models/en2zh/best"),
        ("Quantized", "outputs/quantized/en2zh/quantized"),
    ]
}



logger = setup_logger(__name__, level=logging.INFO)

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT模型评估流程 (Academic Edition)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="评估结果输出路径")
    parser.add_argument("--direction", choices=["both", "zh2en", "en2zh"], default="both", help="评估方向")
    parser.add_argument("--max-samples", type=int, default=300, help="最大样本数 (默认: 300)")
    parser.add_argument("--skip-comet", action="store_true", help="跳过 COMET 评估")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam Search 束宽")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批次大小")

    parser.add_argument("--only-quantized", action="store_true", help="仅评估量化模型")

    args = parser.parse_args()
    
    start_time = time.time()
    print_header("NMT模型评估流程 (学术版)")
    
    # 1. 发现可用模型
    available_models = discover_models(args.direction, args.only_quantized)
    if not available_models:
        logger.error("未找到任何可用的模型！请检查 models/ 或 outputs/ 目录。")
        sys.exit(1)
        
    # 2. 发现可用数据源
    available_sources = discover_data_sources()
    if not available_sources:
        logger.error("未找到任何可用的评估数据源！")
        sys.exit(1)

    # 3. 执行评估循环
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = [] # 存储所有评估结果的列表

    for source in available_sources:
        print_header(f"正在评估数据源: {source['name']}")
        
        for direction in ["zh2en", "en2zh"]:
            if args.direction != "both" and args.direction != direction:
                continue
            
            # 检查是否有该方向的模型
            models_in_dir = available_models.get(direction, [])
            if not models_in_dir:
                continue
                
            # 加载数据
            print_step(f"加载 {direction} 测试数据...")
            source_texts, target_texts = load_test_data(source, direction, args.max_samples)
            
            if not source_texts or not target_texts:
                print_step(f"警告: 无法加载 {source['name']} 的 {direction} 数据，跳过。")
                continue
                
            print_step(f"数据加载完成: {len(source_texts)} 条样本")
            
            # 评估该方向的所有模型
            for model_name, model_path in models_in_dir:
                print_step(f">>> 正在评估模型: {model_name} ({direction})")
                
                # 为每个模型创建独立的输出子目录
                model_output_dir = output_dir / source['name'] / direction / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 执行评估
                scores = evaluate_model_internal(
                    model_path, 
                    model_output_dir,
                    direction,
                    source_texts, 
                    target_texts,
                    args.skip_comet,
                    args.num_beams,
                    args.batch_size
                )
                
                if scores:
                    # 记录结果
                    record = {
                        "Dataset": source['name'],
                        "Direction": direction,
                        "Model": model_name,
                        "BLEU": scores.get("bleu", 0),
                        "chrF": scores.get("chrf", 0),
                        "TER": scores.get("ter", 0),
                        "COMET": scores.get("comet", 0) if scores.get("comet") else 0,
                        "Latency": scores.get("latency_ms", 0),
                        "Size": scores.get("model_size_mb", 0),
                        "Path": str(model_path)
                    }
                    all_results.append(record)

    # 4. 生成报告与可视化
    print_header("生成评估报告与可视化")
    
    if not all_results:
        logger.error("没有任何评估结果生成。")
        sys.exit(1)
        
    df = pd.DataFrame(all_results)
    
    # 保存原始数据 CSV
    csv_path = output_dir / "evaluation_summary.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print_step(f"评估汇总数据已保存至: {csv_path}")
    
    # 打印终端表格
    print("\n评估结果汇总:")
    try:
        print(df[['Dataset', 'Direction', 'Model', 'BLEU', 'chrF', 'Latency', 'Size']].to_string(index=False))
    except:
        print(df)
    
    # 调用可视化模块
    if set_academic_style:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        font_path = project_root / "SourceHanSansSC-Regular-2.otf"
        font_prop = set_academic_style(font_path)
        
        # 1. 柱状对比图 (BLEU)
        plot_bar_comparison(
            df, "BLEU", "各模型 BLEU 分数对比", 
            viz_dir / "bleu_comparison.png", font_prop
        )
        
        # 2. 柱状对比图 (COMET)
        if df['COMET'].sum() > 0:
            plot_bar_comparison(
                df, "COMET", "各模型 COMET 分数对比", 
                viz_dir / "comet_comparison.png", font_prop
            )
            
        # 3. 质量-延迟 散点图
        plot_quality_latency_scatter(
            df, viz_dir / "quality_latency_tradeoff.png", font_prop
        )
        
        # 4. 综合效率排行图
        plot_model_efficiency_ranking(
            df, viz_dir / "model_efficiency_ranking.png", font_prop
        )
        
        # 5. 模型体积对比图
        plot_size_comparison(
            df, viz_dir / "model_size_comparison.png", font_prop
        )
        
        # 6. 雷达图



        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            
            # 为每个方向生成雷达图
            for direction in ["zh2en", "en2zh"]:
                target_df = dataset_df[dataset_df['Direction'] == direction]
                
                if target_df.empty:
                    continue
                    
                radar_data = {}
                
                # 计算各指标的最大值（用于归一化）
                # 注意：延迟和体积是越小越好，所以我们取倒数作为"性能"指标
                # 性能指标 = 1 / Value
                
                # 预计算性能指标
                perfs = []
                for _, row in target_df.iterrows():
                    p = {
                        "BLEU": row['BLEU'],
                        "chrF": row['chrF'],
                        "COMET": row['COMET'] * 100 if row['COMET'] else 0,
                        "Speed": (1000 / row['Latency']) if row['Latency'] > 0 else 0, # sentences/sec
                        "Compact": (1000 / row['Size']) if row['Size'] > 0 else 0      # 1/MB * 1000
                    }
                    perfs.append((row['Model'], p))
                
                # 找最大值
                if not perfs:
                    continue
                    
                max_vals = {
                    "BLEU": max(p[1]["BLEU"] for p in perfs) or 1,
                    "chrF": max(p[1]["chrF"] for p in perfs) or 1,
                    "COMET": max(p[1]["COMET"] for p in perfs) or 1,
                    "Speed": max(p[1]["Speed"] for p in perfs) or 1,
                    "Compact": max(p[1]["Compact"] for p in perfs) or 1
                }


                for model_name, p in perfs:
                    # 归一化到 0-100
                    score_dict = {
                        "BLEU": p["BLEU"] / max_vals["BLEU"] * 100,
                        "chrF": p["chrF"] / max_vals["chrF"] * 100,
                        "Speed": p["Speed"] / max_vals["Speed"] * 100,
                        "Compact": p["Compact"] / max_vals["Compact"] * 100,
                    }
                    if max_vals["COMET"] > 0:
                        score_dict["COMET"] = p["COMET"] / max_vals["COMET"] * 100
                    
                    radar_data[model_name] = score_dict
                
                if radar_data:
                    plot_radar_chart(
                        radar_data, 
                        viz_dir / f"radar_chart_{dataset}_{direction}.png", 
                        font_prop
                    )

    total_elapsed = time.time() - start_time
    print(f"\n总耗时: {format_time(total_elapsed)}")


# ============================================================================
# 辅助函数
# ============================================================================

def discover_models(target_direction: str, only_quantized: bool) -> Dict[str, List[Tuple[str, Path]]]:
    """搜索并返回所有存在的模型路径"""
    found = {}
    
    for direction, candidates in MODEL_SEARCH_PATHS.items():
        if target_direction != "both" and target_direction != direction:
            continue
            
        found[direction] = []
        for name, path_str in candidates:
            # 过滤逻辑
            if only_quantized and "Quantized" not in name:
                continue
                
            path = Path(path_str)
            if path.exists() and (path / "config.json").exists():
                found[direction].append((name, path))
                print_step(f"发现模型: {name} -> {path}")
            elif path.parent.exists() and name == "Original": 
                 # 原始模型可能未下载，给个提示
                 print_step(f"提示: 未找到原始模型 {path}，将跳过对比")
            
    return found

def discover_data_sources() -> List[Dict]:
    """发现可用的数据源"""
    valid_sources = []
    for source in EVAL_DATA_SOURCES:
        if source["type"] == "sacrebleu_wmt":
            # 简化处理：总是启用在线数据源，具体在加载时检查
            valid_sources.append(source)
            # print_step(f"启用数据源: {source['name']}")
    return valid_sources

def load_test_data(source: Dict, direction: str, max_samples: int) -> Tuple[List[str], List[str]]:
    """根据配置加载测试数据"""
    if source["type"] == "sacrebleu_wmt":
        return load_sacrebleu_data(source["path"], direction, max_samples)
    return [], []

def load_sacrebleu_data(dataset_name: str, direction: str, max_samples: int) -> Tuple[List[str], List[str]]:
    """使用 sacrebleu 加载数据"""
    import sacrebleu
    
    lang_pair = "zh-en" if direction == "zh2en" else "en-zh"
    try:
        # 获取源文件 (会自动下载)
        source_file = sacrebleu.get_source_file(dataset_name, lang_pair)
        # 获取参考文件 (取第一个)
        ref_files = sacrebleu.get_reference_files(dataset_name, lang_pair)
        if not ref_files:
             logger.warning(f"未找到参考文件: {dataset_name} {lang_pair}")
             return [], []
        ref_file = ref_files[0]
        
        with open(source_file, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f if line.strip()]
        with open(ref_file, 'r', encoding='utf-8') as f:
            ref_lines = [line.strip() for line in f if line.strip()]
        
        # 确保行数一致
        min_len = min(len(src_lines), len(ref_lines))
        src_lines = src_lines[:min_len]
        ref_lines = ref_lines[:min_len]
            
        # 截取
        if max_samples and max_samples > 0:
            src_lines = src_lines[:max_samples]
            ref_lines = ref_lines[:max_samples]
            
        return src_lines, ref_lines
    except Exception as e:
        logger.warning(f"无法加载 sacrebleu 数据 {dataset_name} ({lang_pair}): {e}")
        return [], []


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_step(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def get_model_size(model_dir: Path) -> float:
    total_size = 0
    for f in model_dir.glob('**/*'):
        if f.is_file():
            total_size += f.stat().st_size
    return total_size / (1024 * 1024)

def evaluate_model_internal(
    model_dir: Path, 
    output_path: Path, 
    direction: str,
    source_texts: List[str],
    target_texts: List[str],
    skip_comet: bool = False,
    num_beams: int = 4,
    batch_size: int = 16
) -> Dict[str, float]:

    # 获取模型体积
    model_size = get_model_size(model_dir)
    
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        from tqdm import tqdm
        
        tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
        
        # 智能加载逻辑
        quant_stats_file = model_dir / "quantization_stats.json"
        is_quantized = quant_stats_file.exists()
        
        model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if is_quantized:
             try:
                 # 尝试直接加载 (如果是 FP16 或 Transformers 支持的量化)
                 model = MarianMTModel.from_pretrained(str(model_dir))
             except:
                 # 失败则尝试作为动态量化模型加载 (INT8 CPU only)
                 print_step("尝试加载动态量化模型 (INT8)...")
                 config_path = model_dir / "config.json"
                 if config_path.exists():
                     # 加载基础架构
                     base_model_name = f"Helsinki-NLP/opus-mt-{direction[:2]}-{direction[3:]}"
                     base_model = MarianMTModel.from_pretrained(base_model_name)
                     # 应用量化
                     model = torch.quantization.quantize_dynamic(
                        base_model, {torch.nn.Linear}, dtype=torch.qint8
                     )
                     # 加载权重
                     state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
                     model.load_state_dict(state_dict)
                     device = torch.device("cpu") # 强制 CPU
                 else:
                     logger.error("无法加载量化模型: 缺少配置")
                     return {}
        else:
            model = MarianMTModel.from_pretrained(str(model_dir))
            
        if model is None:
            return {}
            
        model = model.to(device)
        model.eval()
        
        # 翻译循环
        translations = []
        total_time = 0
        
        # 预热
        if len(source_texts) > 0:
            dummy = tokenizer(source_texts[:1], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                model.generate(**dummy, max_length=10)
        
        for i in tqdm(range(0, len(source_texts), batch_size), desc="翻译中"):
            batch = source_texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=512, 
                    num_beams=num_beams,
                    early_stopping=True
                )
            total_time += (time.time() - t0)
            
            translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        avg_latency = (total_time / len(source_texts)) * 1000
        
        # 计算指标
        import sacrebleu
        tokenize = 'zh' if direction == 'en2zh' else '13a'
        bleu = sacrebleu.corpus_bleu(translations, [target_texts], tokenize=tokenize)
        chrf = sacrebleu.corpus_chrf(translations, [target_texts], word_order=2)
        ter = sacrebleu.corpus_ter(translations, [target_texts])
        
        scores = {
            "bleu": bleu.score,
            "chrf": chrf.score,
            "ter": ter.score,
            "model_size_mb": model_size,
            "latency_ms": avg_latency
        }
        
        # COMET
        if not skip_comet:
            try:
                from comet import download_model, load_from_checkpoint
                model_name = "Unbabel/wmt22-comet-da"
                # 尝试离线加载或下载
                try:
                    model_path = download_model(model_name)
                    comet_model = load_from_checkpoint(model_path)
                except Exception as e:
                    print_step(f"COMET下载失败，尝试本地缓存或跳过: {e}")
                    raise e
                    
                comet_model.eval()
                if torch.cuda.is_available():
                    comet_model.cuda()
                    
                # 针对 4060 8G 优化的 COMET 配置
                print_step("计算 COMET 指标 (神经网络评估)...")
                # 显存 8G 建议 batch_size 为 16，避免 OOM
                out = comet_model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0, progress_bar=True)

                scores["comet"] = out.system_score

            except Exception as e:
                # logger.warning(f"COMET 计算失败: {e}")
                pass
        
        # 保存结果到文件
        with open(output_path / "scores.json", 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2)
        with open(output_path / "hypothesis.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(translations))
        
        return scores
        
    except Exception as e:
        logger.error(f"评估失败 ({model_dir}): {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    main()