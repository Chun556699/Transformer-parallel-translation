#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练后/压缩后可视化模块

功能说明：
    - 模型结构对比图
    - 权重分布对比图
    - 推理时延对比图
    - 模型体积对比图
    - 评测指标雷达图
    - 多领域评测对比图
    - 显著性检验结果图
    - 四模型综合对比图

输出图表：
    - post_model_comparison.png: 模型结构对比
    - post_weight_distribution.png: 权重分布对比
    - post_latency_comparison.png: 推理时延对比
    - post_size_comparison.png: 模型体积对比
    - post_metrics_radar.png: 评测指标雷达图
    - post_domain_comparison.png: 多领域对比
    - post_significance_test.png: 显著性检验
    - post_four_model_comparison.png: 四模型综合对比

依赖：
    - matplotlib
    - seaborn
    - numpy

作者：NMT 翻译系统
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置
# ============================================================================

# 默认中文字体路径
DEFAULT_FONT_PATH = "e:\\Graduation-Project\\SourceHanSansSC-Regular-2.otf"

# 颜色方案
COLORS = {
    'baseline': '#3498db',    # 蓝色 - 基线模型
    'finetuned': '#2ecc71',   # 绿色 - 微调模型
    'pruned': '#f39c12',      # 橙色 - 剪枝模型
    'quantized': '#e74c3c',   # 红色 - 量化模型
    'cpu': '#9b59b6',         # 紫色 - CPU
    'gpu': '#1abc9c',         # 青色 - GPU
}

# 模型名称映射
MODEL_NAMES = {
    'baseline': '基线模型',
    'finetuned': '微调模型',
    'pruned': '剪枝模型',
    'quantized': '量化模型',
}


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ModelComparisonData:
    """模型对比数据"""
    
    # 模型体积 (MB)
    sizes: Dict[str, float] = None
    
    # 推理延迟 (ms)
    latencies: Dict[str, Dict[str, float]] = None  # {model: {cpu: x, gpu: y}}
    
    # 评测指标
    metrics: Dict[str, Dict[str, float]] = None  # {model: {bleu: x, comet: y, ...}}
    
    # 领域评测
    domain_scores: Dict[str, Dict[str, float]] = None  # {model: {news: x, patent: y, ...}}
    
    # 权重统计
    weight_stats: Dict[str, Dict[str, Any]] = None  # {model: {mean: x, std: y, ...}}
    
    def __post_init__(self):
        """初始化默认值"""
        if self.sizes is None:
            self.sizes = {}
        if self.latencies is None:
            self.latencies = {}
        if self.metrics is None:
            self.metrics = {}
        if self.domain_scores is None:
            self.domain_scores = {}
        if self.weight_stats is None:
            self.weight_stats = {}


# ============================================================================
# 可视化类
# ============================================================================

class PostTrainingVisualizer:
    """
    训练后/压缩后可视化器
    
    功能：
        - 生成模型对比图表
        - 评测结果可视化
        - 性能基准图表
    """
    
    def __init__(self, 
                 output_dir: str = "outputs/visualizations/post_training",
                 font_path: str = DEFAULT_FONT_PATH):
        """
        初始化可视化器
        
        参数：
            output_dir: 输出目录
            font_path: 中文字体路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置字体
        self._setup_font(font_path)
        
        # 配置样式
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 150,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
        })
        sns.set_style("whitegrid")
    
    def _setup_font(self, font_path: str) -> None:
        """配置中文字体"""
        import matplotlib
        
        path = Path(font_path)
        # 尝试多个可能的路径
        possible_paths = [
            path if path.is_absolute() else None,
            Path(__file__).parent.parent.parent / font_path,  # 项目根目录
            Path.cwd() / font_path,  # 当前工作目录
        ]
        possible_paths = [p for p in possible_paths if p is not None]
        
        font_loaded = False
        for p in possible_paths:
            if p.exists():
                try:
                    # 清除字体缓存
                    cache_dir = matplotlib.get_cachedir()
                    if Path(cache_dir).exists():
                        for f in Path(cache_dir).glob('fontlist*'):
                            try:
                                f.unlink()
                            except:
                                pass
                    
                    # 注册字体
                    fm.fontManager.addfont(str(p))
                    self.font_prop = fm.FontProperties(fname=str(p))
                    font_name = self.font_prop.get_name()
                    
                    # 强制设置全局字体（正确方式）
                    plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    font_loaded = True
                    break
                except Exception as e:
                    continue
        
        if not font_loaded:
            self.font_prop = None
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
    
    def _get_font_kwargs(self) -> Dict:
        """获取字体参数"""
        if self.font_prop:
            return {'fontproperties': self.font_prop}
        return {}
    
    def _apply_font_to_figure(self, fig) -> None:
        """将字体应用到图形的所有文本元素"""
        if self.font_prop is None:
            return
        
        for text in fig.findobj(lambda x: hasattr(x, 'set_fontproperties')):
            try:
                text.set_fontproperties(self.font_prop)
            except:
                pass
    
    def plot_model_size_comparison(self,
                                   sizes: Dict[str, float],
                                   title: str = "模型体积对比",
                                   save_name: str = "post_size_comparison.png") -> str:
        """
        绘制模型体积对比图
        
        参数：
            sizes: 各模型体积 {模型名: 体积MB}
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(sizes.keys())
        values = list(sizes.values())
        colors = [COLORS.get(m, '#666666') for m in models]
        labels = [MODEL_NAMES.get(m, m) for m in models]
        
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{val:.1f}MB', ha='center', fontsize=11, fontweight='bold',
                   **self._get_font_kwargs())
        
        # 添加压缩比标注
        if 'baseline' in sizes and len(sizes) > 1:
            baseline_size = sizes['baseline']
            for i, (model, size) in enumerate(sizes.items()):
                if model != 'baseline':
                    ratio = baseline_size / size
                    ax.annotate(f'{ratio:.1f}x 压缩', 
                               xy=(i, size/2),
                               ha='center', fontsize=10, color='white',
                               fontweight='bold', **self._get_font_kwargs())
        
        ax.set_ylabel('模型体积 (MB)', **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        ax.set_ylim(0, max(values) * 1.2)
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"模型体积对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_latency_comparison(self,
                                latencies: Dict[str, Dict[str, float]],
                                title: str = "推理时延对比",
                                save_name: str = "post_latency_comparison.png") -> str:
        """
        绘制推理时延对比图
        
        参数：
            latencies: {模型名: {cpu: 延迟ms, gpu: 延迟ms}}
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(latencies.keys())
        labels = [MODEL_NAMES.get(m, m) for m in models]
        x = np.arange(len(models))
        width = 0.35
        
        cpu_values = [latencies[m].get('cpu', 0) for m in models]
        gpu_values = [latencies[m].get('gpu', 0) for m in models]
        
        bars1 = ax.bar(x - width/2, cpu_values, width, label='CPU', 
                      color=COLORS['cpu'], edgecolor='white')
        bars2 = ax.bar(x + width/2, gpu_values, width, label='GPU', 
                      color=COLORS['gpu'], edgecolor='white')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.1f}', ha='center', fontsize=9,
                       **self._get_font_kwargs())
        
        ax.set_ylabel('延迟 (ms/100 tokens)', **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, **self._get_font_kwargs())
        ax.legend(prop=self.font_prop if self.font_prop else None)
        
        # 添加目标线
        ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='CPU 目标 (60ms)')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='GPU 目标 (20ms)')
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"推理时延对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_metrics_radar(self,
                           metrics: Dict[str, Dict[str, float]],
                           title: str = "评测指标雷达图",
                           save_name: str = "post_metrics_radar.png") -> str:
        """
        绘制评测指标雷达图
        
        参数：
            metrics: {模型名: {指标名: 分数}}
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        if not metrics:
            logger.warning("无评测指标数据，跳过绘图")
            return ""
        
        # 获取所有指标名
        all_metrics = set()
        for model_metrics in metrics.values():
            all_metrics.update(model_metrics.keys())
        metric_names = sorted(list(all_metrics))
        
        # 指标名称映射
        metric_labels = {
            'bleu': 'BLEU',
            'sacrebleu': 'sacreBLEU',
            'comet': 'COMET',
            'bertscore': 'BERTScore',
            'chrf': 'chrF++',
            'ter': 'TER (↓)',
        }
        
        labels = [metric_labels.get(m, m) for m in metric_names]
        num_vars = len(metric_names)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for model, model_metrics in metrics.items():
            values = [model_metrics.get(m, 0) for m in metric_names]
            # TER 需要反转（越低越好）
            if 'ter' in metric_names:
                ter_idx = metric_names.index('ter')
                values[ter_idx] = 100 - values[ter_idx]  # 反转
            
            values += values[:1]  # 闭合
            
            color = COLORS.get(model, '#666666')
            label = MODEL_NAMES.get(model, model)
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=label, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, y=1.08, **self._get_font_kwargs())
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                 prop=self.font_prop if self.font_prop else None)
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"评测指标雷达图已保存: {save_path}")
        return str(save_path)
    
    def plot_domain_comparison(self,
                               domain_scores: Dict[str, Dict[str, float]],
                               title: str = "多领域评测对比",
                               save_name: str = "post_domain_comparison.png") -> str:
        """
        绘制多领域评测对比图
        
        参数：
            domain_scores: {模型名: {领域名: BLEU分数}}
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        if not domain_scores:
            logger.warning("无领域评测数据，跳过绘图")
            return ""
        
        # 领域名称映射
        domain_labels = {
            'news': '新闻',
            'patent': '专利',
            'subtitle': '字幕',
            'spoken': '口语',
            'medical': '医疗',
        }
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        models = list(domain_scores.keys())
        domains = set()
        for scores in domain_scores.values():
            domains.update(scores.keys())
        domains = sorted(list(domains))
        
        x = np.arange(len(domains))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            scores = [domain_scores[model].get(d, 0) for d in domains]
            color = COLORS.get(model, '#666666')
            label = MODEL_NAMES.get(model, model)
            
            bars = ax.bar(x + i * width - 0.4 + width/2, scores, width,
                         label=label, color=color, edgecolor='white')
            
            # 添加数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.1f}', ha='center', fontsize=8,
                       **self._get_font_kwargs())
        
        domain_names = [domain_labels.get(d, d) for d in domains]
        ax.set_xticks(x)
        ax.set_xticklabels(domain_names, **self._get_font_kwargs())
        ax.set_ylabel('BLEU 分数', **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        ax.legend(prop=self.font_prop if self.font_prop else None)
        ax.set_ylim(0, max([max(s.values()) for s in domain_scores.values()]) * 1.15)
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"多领域对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_weight_distribution(self,
                                 weight_data: Dict[str, np.ndarray],
                                 title: str = "权重分布对比",
                                 save_name: str = "post_weight_distribution.png") -> str:
        """
        绘制权重分布对比图
        
        参数：
            weight_data: {模型名: 权重数组}
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        if not weight_data:
            logger.warning("无权重数据，跳过绘图")
            return ""
        
        fig, axes = plt.subplots(1, len(weight_data), figsize=(5*len(weight_data), 5))
        if len(weight_data) == 1:
            axes = [axes]
        
        for ax, (model, weights) in zip(axes, weight_data.items()):
            color = COLORS.get(model, '#666666')
            label = MODEL_NAMES.get(model, model)
            
            weights_flat = weights.flatten()
            ax.hist(weights_flat, bins=100, color=color, alpha=0.7, 
                   edgecolor='white', density=True)
            
            # 统计信息
            mean = np.mean(weights_flat)
            std = np.std(weights_flat)
            ax.axvline(x=mean, color='red', linestyle='--', 
                      label=f'均值: {mean:.4f}')
            ax.axvline(x=mean+std, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(x=mean-std, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_title(label, fontsize=12, **self._get_font_kwargs())
            ax.set_xlabel('权重值', **self._get_font_kwargs())
            ax.set_ylabel('密度', **self._get_font_kwargs())
            ax.legend(fontsize=8, prop=self.font_prop if self.font_prop else None)
        
        fig.suptitle(title, fontsize=16, **self._get_font_kwargs())
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"权重分布对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_four_model_comparison(self,
                                   data: ModelComparisonData,
                                   title: str = "四模型综合对比",
                                   save_name: str = "post_four_model_comparison.png") -> str:
        """
        绘制四模型（基线/微调/剪枝/量化）综合对比图
        
        参数：
            data: 模型对比数据
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 2x2 布局
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 模型体积
        ax1 = fig.add_subplot(gs[0, 0])
        if data.sizes:
            models = list(data.sizes.keys())
            values = list(data.sizes.values())
            colors = [COLORS.get(m, '#666') for m in models]
            labels = [MODEL_NAMES.get(m, m) for m in models]
            
            bars = ax1.bar(labels, values, color=colors)
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
                        f'{val:.0f}MB', ha='center', fontsize=10)
            ax1.set_ylabel('体积 (MB)', **self._get_font_kwargs())
            ax1.set_title('模型体积', fontsize=14, **self._get_font_kwargs())
        
        # 2. 推理延迟
        ax2 = fig.add_subplot(gs[0, 1])
        if data.latencies:
            models = list(data.latencies.keys())
            labels = [MODEL_NAMES.get(m, m) for m in models]
            x = np.arange(len(models))
            width = 0.35
            
            cpu = [data.latencies[m].get('cpu', 0) for m in models]
            gpu = [data.latencies[m].get('gpu', 0) for m in models]
            
            ax2.bar(x - width/2, cpu, width, label='CPU', color=COLORS['cpu'])
            ax2.bar(x + width/2, gpu, width, label='GPU', color=COLORS['gpu'])
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, **self._get_font_kwargs())
            ax2.set_ylabel('延迟 (ms)', **self._get_font_kwargs())
            ax2.set_title('推理延迟', fontsize=14, **self._get_font_kwargs())
            ax2.legend(prop=self.font_prop if self.font_prop else None)
        
        # 3. BLEU 分数
        ax3 = fig.add_subplot(gs[1, 0])
        if data.metrics:
            models = list(data.metrics.keys())
            bleu_scores = [data.metrics[m].get('bleu', data.metrics[m].get('sacrebleu', 0)) 
                          for m in models]
            colors = [COLORS.get(m, '#666') for m in models]
            labels = [MODEL_NAMES.get(m, m) for m in models]
            
            bars = ax3.bar(labels, bleu_scores, color=colors)
            for bar, val in zip(bars, bleu_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                        f'{val:.1f}', ha='center', fontsize=10)
            ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='目标 (30.0)')
            ax3.set_ylabel('BLEU', **self._get_font_kwargs())
            ax3.set_title('翻译质量 (BLEU)', fontsize=14, **self._get_font_kwargs())
            ax3.legend(prop=self.font_prop if self.font_prop else None)
        
        # 4. 综合评分雷达图
        ax4 = fig.add_subplot(gs[1, 1], polar=True)
        if data.metrics:
            categories = ['BLEU', 'COMET', 'BERTScore', '压缩比', '速度']
            num_vars = len(categories)
            angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            for model in data.metrics:
                m = data.metrics[model]
                # 归一化各指标到 0-100
                values = [
                    m.get('bleu', m.get('sacrebleu', 0)) / 40 * 100,
                    m.get('comet', 0) * 100,
                    m.get('bertscore', 0) * 100,
                    min(data.sizes.get('baseline', 300) / data.sizes.get(model, 300), 3) / 3 * 100 if data.sizes else 50,
                    100 - min(data.latencies.get(model, {}).get('gpu', 20) / 60 * 100, 100) if data.latencies else 50,
                ]
                values += values[:1]
                
                color = COLORS.get(model, '#666')
                label = MODEL_NAMES.get(model, model)
                ax4.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
                ax4.fill(angles, values, alpha=0.15, color=color)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories, **self._get_font_kwargs())
            ax4.set_title('综合评分', fontsize=14, y=1.1, **self._get_font_kwargs())
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                      prop=self.font_prop if self.font_prop else None)
        
        fig.suptitle(title, fontsize=18, **self._get_font_kwargs())
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"四模型综合对比图已保存: {save_path}")
        return str(save_path)
    
    def generate_all_post_plots(self, 
                                 data: ModelComparisonData) -> List[str]:
        """
        生成所有训练后图表
        
        参数：
            data: 模型对比数据
            
        返回：
            List[str]: 生成的文件路径列表
        """
        saved_files = []
        
        if data.sizes:
            path = self.plot_model_size_comparison(data.sizes)
            if path:
                saved_files.append(path)
        
        if data.latencies:
            path = self.plot_latency_comparison(data.latencies)
            if path:
                saved_files.append(path)
        
        if data.metrics:
            path = self.plot_metrics_radar(data.metrics)
            if path:
                saved_files.append(path)
        
        if data.domain_scores:
            path = self.plot_domain_comparison(data.domain_scores)
            if path:
                saved_files.append(path)
        
        # 综合对比图
        path = self.plot_four_model_comparison(data)
        if path:
            saved_files.append(path)
        
        logger.info(f"共生成 {len(saved_files)} 个图表")
        return saved_files


# ============================================================================
# 便捷函数
# ============================================================================

def create_post_visualizer(output_dir: str = None, 
                           font_path: str = None) -> PostTrainingVisualizer:
    """创建训练后可视化器"""
    kwargs = {}
    if output_dir:
        kwargs['output_dir'] = output_dir
    if font_path:
        kwargs['font_path'] = font_path
    return PostTrainingVisualizer(**kwargs)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'ModelComparisonData',
    'PostTrainingVisualizer',
    'create_post_visualizer',
]


# ============================================================================
# 主函数（演示）
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练后可视化工具')
    parser.add_argument('--output-dir', type=str,
                       default='e:\\Graduation-Project\\outputs\\visualizations\\post_training',
                       help='输出目录')
    parser.add_argument('--demo', action='store_true', help='生成演示图表')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.demo:
        print("生成演示图表...")
        
        # 演示数据
        data = ModelComparisonData(
            sizes={
                'baseline': 298.5,
                'finetuned': 298.5,
                'pruned': 120.3,
                'quantized': 95.7,
            },
            latencies={
                'baseline': {'cpu': 85.2, 'gpu': 28.5},
                'finetuned': {'cpu': 85.0, 'gpu': 28.3},
                'pruned': {'cpu': 52.1, 'gpu': 18.6},
                'quantized': {'cpu': 45.3, 'gpu': 15.2},
            },
            metrics={
                'baseline': {'bleu': 28.5, 'comet': 0.78, 'bertscore': 0.82, 'chrf': 52.3, 'ter': 48.5},
                'finetuned': {'bleu': 31.2, 'comet': 0.82, 'bertscore': 0.85, 'chrf': 55.8, 'ter': 45.2},
                'pruned': {'bleu': 30.8, 'comet': 0.81, 'bertscore': 0.84, 'chrf': 55.1, 'ter': 46.0},
                'quantized': {'bleu': 30.5, 'comet': 0.80, 'bertscore': 0.83, 'chrf': 54.5, 'ter': 46.5},
            },
            domain_scores={
                'baseline': {'news': 28.5, 'patent': 25.3, 'subtitle': 30.2, 'spoken': 27.8, 'medical': 24.1},
                'finetuned': {'news': 31.2, 'patent': 28.5, 'subtitle': 33.5, 'spoken': 30.8, 'medical': 27.2},
                'pruned': {'news': 30.8, 'patent': 28.0, 'subtitle': 33.0, 'spoken': 30.2, 'medical': 26.8},
                'quantized': {'news': 30.5, 'patent': 27.5, 'subtitle': 32.5, 'spoken': 29.8, 'medical': 26.3},
            }
        )
        
        visualizer = PostTrainingVisualizer(output_dir=args.output_dir)
        files = visualizer.generate_all_post_plots(data)
        
        print(f"已生成 {len(files)} 个图表:")
        for f in files:
            print(f"  - {f}")
