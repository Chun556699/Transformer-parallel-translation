#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级可视化图表生成模块

功能说明：
    - 提供符合顶级会议 (ACL/EMNLP) 标准的图表样式
    - 支持多模型、多指标对比
    - 自动生成：
        1. 柱状对比图 (Bar Chart) - BLEU/COMET
        2. 性能-效率权衡图 (Scatter Plot) - Quality vs Latency
        3. 多维雷达图 (Radar Chart) - 综合评估
        4. 模型压缩率图 (Size Comparison)

作者：NMT翻译系统
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# ============================================================================
# 样式配置
# ============================================================================

def set_academic_style(font_path: Path = None):
    """设置学术风格"""
    # 使用 seaborn 的 paper 风格
    sns.set_theme(style="ticks", context="talk", font_scale=1.0)
    
    # 字体配置
    plt.rcParams['font.family'] = ['sans-serif']
    # 增加更多备用中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans', 'sans-serif']
    
    # 如果有指定的自定义字体文件
    font_prop = None
    if font_path and font_path.exists():
        try:
            font_prop = fm.FontProperties(fname=str(font_path))
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
            print(f"  成功加载字体: {font_prop.get_name()}")
        except Exception as e:
            print(f"  警告: 无法加载指定字体文件: {e}")
            
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置全局背景色
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    return font_prop


# ============================================================================
# 图表生成函数
# ============================================================================

def plot_bar_comparison(
    data: pd.DataFrame, 
    metric: str, 
    title: str, 
    output_path: Path, 
    font_prop=None
):
    """
    生成柱状对比图
    data: DataFrame, columns=['Model', 'Direction', metric, 'Dataset']
    """
    plt.figure(figsize=(12, 7))
    
    # 优化调色板，使用更有质感的颜色
    premium_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]
    
    # 创建分组柱状图
    ax = sns.barplot(
        data=data, 
        x="Direction", 
        y=metric, 
        hue="Model", 
        palette=premium_colors,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.85
    )
    
    # 设置标题和标签
    if font_prop:
        plt.title(title, fontproperties=font_prop, fontsize=18, pad=25, fontweight='bold')
        plt.xlabel("翻译方向 (Translation Direction)", fontproperties=font_prop, fontsize=13, labelpad=10)
        plt.ylabel(f"{metric} 分数", fontproperties=font_prop, fontsize=13, labelpad=10)
        plt.legend(title="模型 (Models)", prop=font_prop, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.title(title, fontsize=18, pad=25, fontweight='bold')
        plt.xlabel("Direction", fontsize=13, labelpad=10)
        plt.ylabel(f"{metric} Score", fontsize=13, labelpad=10)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
    # 添加数值标签 (居中显示在柱子上方)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=5, fontsize=10, fontweight='bold')
        
    # 调整Y轴范围，留出头部空间
    y_max = data[metric].max()
    plt.ylim(0, y_max * 1.25)
    
    # 美化网格
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 去除多余边框
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已生成柱状图: {output_path}")


def plot_quality_latency_scatter(
    data: pd.DataFrame, 
    output_path: Path, 
    font_prop=None
):
    """
    生成 质量-延迟 权衡散点图 (Pareto Frontier)
    data: DataFrame, columns=['Model', 'Direction', 'BLEU', 'Latency', 'Size']
    """
    plt.figure(figsize=(13, 9))
    
    # 设置背景
    sns.set_style("whitegrid")
    
    # 绘制散点
    scatter = sns.scatterplot(
        data=data,
        x="Latency",
        y="BLEU",
        hue="Model",
        style="Direction",
        size="Size",
        sizes=(150, 800),
        alpha=0.7,
        palette="husl",
        edgecolor="black",
        linewidth=1
    )
    
    # 绘制帕累托前沿 (Pareto Frontier) - 简化示意
    # 对每个方向找出最优解连线
    for direction in data['Direction'].unique():
        sub = data[data['Direction'] == direction].sort_values('Latency')
        pareto_x = []
        pareto_y = []
        current_max_bleu = -1
        for _, row in sub.iterrows():
            if row['BLEU'] > current_max_bleu:
                pareto_x.append(row['Latency'])
                pareto_y.append(row['BLEU'])
                current_max_bleu = row['BLEU']
        
        if len(pareto_x) > 1:
            plt.plot(pareto_x, pareto_y, linestyle='--', alpha=0.4, color='grey')

    # 添加文本标签
    for i in range(data.shape[0]):
        row = data.iloc[i]
        plt.text(
            row.Latency + 0.5, 
            row.BLEU + 0.3, 
            f"{row.Model}\n{row.Size:.1f}MB", 
            fontsize=9,
            fontweight='medium',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )
        
    # 设置标题和标签
    if font_prop:
        plt.title("翻译质量 vs 推理延迟 权衡对比图", fontproperties=font_prop, fontsize=18, pad=25)
        plt.xlabel("平均单句延迟 (ms/sentence) [越左越好]", fontproperties=font_prop, fontsize=13)
        plt.ylabel("BLEU 分数 (越高越好)", fontproperties=font_prop, fontsize=13)
        plt.legend(prop=font_prop, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        plt.title("Translation Quality vs Inference Latency Trade-off", fontsize=18, pad=25)
        plt.xlabel("Latency (ms) [Lower is Better]", fontsize=13)
        plt.ylabel("BLEU Score [Higher is Better]", fontsize=13)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成质量-效率对比图: {output_path}")



def plot_radar_chart(
    results: Dict[str, Dict[str, float]], 
    output_path: Path, 
    font_prop=None
):
    """
    生成雷达图 (综合能力雷达图)
    """
    if not results:
        return

    # 准备数据
    categories = list(next(iter(results.values())).keys())
    # 汉化标签
    cat_mapping = {
        "BLEU": "翻译质量 (BLEU)",
        "chrF": "字符精度 (chrF)",
        "COMET": "语义评分 (COMET)",
        "Speed": "推理速度 (Speed)",
        "Compact": "模型轻量化 (Compact)"
    }
    display_categories = [cat_mapping.get(c, c) for c in categories]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(11, 11), facecolor='white')
    ax = plt.subplot(111, polar=True)
    
    # 画背景网格
    plt.xticks(angles[:-1], display_categories, size=12)
    if font_prop:
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
    
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="#666666", size=10)
    plt.ylim(0, 110)
    
    # 绘制
    premium_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    
    for i, (model_name, scores) in enumerate(results.items()):
        values = [scores.get(cat, 0) for cat in categories]
        values += values[:1]
        
        color = premium_colors[i % len(premium_colors)]
        ax.plot(angles, values, linewidth=3, linestyle='solid', label=model_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)
        
    # 装饰
    if font_prop:
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop=font_prop, fontsize=11)
        plt.title("模型综合性能雷达图 (Normalized Metrics)", fontproperties=font_prop, size=22, y=1.1, fontweight='bold')
    else:
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title("Model Integrated Performance Radar", size=22, y=1.1, fontweight='bold')
        
    # 网格线优化
    ax.grid(color='#EEEEEE', linewidth=1, linestyle='-')
    ax.set_facecolor('#FAFAFA')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成雷达图: {output_path}")


def plot_model_efficiency_ranking(
    data: pd.DataFrame,
    output_path: Path,
    font_prop=None
):
    """
    生成模型效率排行图 (综合对比)
    """
    plt.figure(figsize=(12, 8))
    
    # 计算综合得分 (简单加权: 60% BLEU + 20% Speed + 20% Compact)
    # 先归一化
    df = data.copy()
    for col in ['BLEU', 'Latency', 'Size']:
        if col not in df.columns:
            continue
        if col == 'BLEU':
            max_val = df[col].max()
            df['n_BLEU'] = df[col] / max_val if max_val > 0 else 0
        else:
            min_val = df[col].min()
            # 越小越好，用 min/value 归一化到 0-1
            df[f'n_{col}'] = df[col].apply(lambda x: min_val / x if x > 0 else 0)
            
    # 确保所有必要列都存在
    for n_col in ['n_BLEU', 'n_Latency', 'n_Size']:
        if n_col not in df.columns:
            df[n_col] = 0
            
    df['Efficiency_Score'] = (df['n_BLEU'] * 0.6 + df['n_Latency'] * 0.2 + df['n_Size'] * 0.2) * 100

    df = df.sort_values('Efficiency_Score', ascending=False)
    
    # 绘图
    ax = sns.barplot(
        data=df,
        y="Model",
        x="Efficiency_Score",
        hue="Direction",
        palette="flare",
        edgecolor="white"
    )
    
    # 设置标签
    if font_prop:
        plt.title("模型综合效率排行 (综合考量翻译质量与系统开销)", fontproperties=font_prop, fontsize=18, pad=25)
        plt.xlabel("综合效率得分 (0-100)", fontproperties=font_prop, fontsize=13)
        plt.ylabel("评估模型", fontproperties=font_prop, fontsize=13)
    else:
        plt.title("Model Efficiency Ranking (Quality vs Cost)", fontsize=18, pad=25)
        plt.xlabel("Integrated Score (0-100)", fontsize=13)
        plt.ylabel("Models", fontsize=13)
        
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=5, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成排行图: {output_path}")


def plot_size_comparison(
    data: pd.DataFrame,
    output_path: Path,
    font_prop=None
):
    """
    生成模型体积大小对比图 (Size Comparison)
    """
    plt.figure(figsize=(12, 7))
    
    # 过滤出每个模型的一条记录（体积与方向无关）
    df_size = data.drop_duplicates(subset=['Model']).copy()
    # 按体积从大到小排序
    df_size = df_size.sort_values('Size', ascending=False)
    
    # 针对 4060 等设备，关注显存/磁盘占用
    colors = sns.color_palette("viridis", len(df_size))
    
    ax = sns.barplot(
        data=df_size,
        x="Size",
        y="Model",
        palette=colors,
        edgecolor="black",
        linewidth=1,
        alpha=0.8
    )
    
    # 添加压缩率标注 (以第一个模型即 Original 为基准)
    base_size = df_size.iloc[0]['Size']
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ratio = (width / base_size) * 100
        ax.text(
            width + 10, 
            p.get_y() + p.get_height()/2, 
            f'{width:.1f} MB ({ratio:.1f}%)', 
            va='center', 
            fontsize=11, 
            fontweight='bold'
        )
    
    # 设置标题和标签
    if font_prop:
        plt.title("模型存储与显存占用对比 (Model Footprint)", fontproperties=font_prop, fontsize=18, pad=25)
        plt.xlabel("占用空间 (MB) [越小越好]", fontproperties=font_prop, fontsize=13)
        plt.ylabel("评估模型", fontproperties=font_prop, fontsize=13)
    else:
        plt.title("Model Storage & Memory Footprint Comparison", fontsize=18, pad=25)
        plt.xlabel("Size (MB) [Lower is Better]", fontsize=13)
        plt.ylabel("Models", fontsize=13)
    
    # 增加 X 轴范围，留出文字空间
    plt.xlim(0, base_size * 1.3)
    
    # 增加参考线
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已生成体积对比图: {output_path}")



