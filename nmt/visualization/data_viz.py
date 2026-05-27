"""
数据可视化模块

功能说明：
    提供训练前数据相关的可视化功能，包括：
    - 原始数据分布图（长度、词频）
    - 清洗前后对比图
    - 数据筛选效果图
    - BPE 分词效果展示
    - 难度分布统计图

字体配置：
    使用 SourceHanSansSC-Regular-2.otf 字体以正确显示中文

输出目录：
    outputs/visualizations/pre_training/

依赖：
    - matplotlib: 绑图
    - seaborn: 统计可视化
    - numpy: 数值计算
    - pandas: 数据处理

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from dataclasses import dataclass

import numpy as np

# 可视化库（可选导入）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib 未安装，可视化功能将被禁用")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ====================================
# 常量定义
# ====================================

# 默认字体路径
DEFAULT_FONT_PATH = "SourceHanSansSC-Regular-2.otf"

# 默认输出目录
DEFAULT_OUTPUT_DIR = "outputs/visualizations"

# 图表颜色配置
COLORS = {
    "primary": "#2E86AB",      # 主色调
    "secondary": "#A23B72",    # 次色调
    "accent": "#F18F01",       # 强调色
    "success": "#C73E1D",      # 成功色
    "background": "#F5F5F5",   # 背景色
    "easy": "#4CAF50",         # 简单样本
    "medium": "#FF9800",       # 中等样本
    "hard": "#F44336",         # 困难样本
}

# 图表样式配置
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)


@dataclass
class VisualizationConfig:
    """
    可视化配置数据类
    
    属性：
        font_path: 字体文件路径
        output_dir: 输出目录
        dpi: 图像分辨率
        figure_size: 图像尺寸
        style: seaborn 样式
    """
    font_path: str = DEFAULT_FONT_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    dpi: int = FIGURE_DPI
    figure_size: Tuple[int, int] = FIGURE_SIZE
    style: str = "whitegrid"


class DataVisualizer:
    """
    数据可视化器
    
    功能说明：
        生成训练前阶段的各种数据可视化图表：
        - 数据分布分析
        - 清洗效果对比
        - 筛选结果展示
        - 难度分级统计
    
    参数：
        config: 可视化配置
        logger: 日志记录器
        
    示例：
        >>> visualizer = DataVisualizer()
        >>> visualizer.plot_length_distribution(data, "length_dist.png")
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化可视化器
        
        参数：
            config: 可视化配置
            logger: 日志记录器
        """
        self.config = config or VisualizationConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 检查依赖
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib 不可用，可视化功能受限")
            return
        
        # 设置字体
        self._setup_font()
        
        # 设置样式
        self._setup_style()
        
        # 确保输出目录存在
        self._ensure_output_dirs()
        
        self.logger.info("数据可视化器初始化完成")
    
    def _setup_font(self) -> None:
        """
        设置中文字体
        
        使用 SourceHanSansSC 字体确保中文正确显示。
        """
        import matplotlib
        
        font_path = Path(self.config.font_path)
        
        # 尝试多个可能的路径（绝对路径优先）
        possible_paths = [
            font_path if font_path.is_absolute() else None,
            Path(__file__).parent.parent.parent / font_path,  # 项目根目录
            Path.cwd() / font_path,  # 当前工作目录
        ]
        possible_paths = [p for p in possible_paths if p is not None]
        
        for path in possible_paths:
            if path.exists():
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
                    fm.fontManager.addfont(str(path))
                    font_prop = fm.FontProperties(fname=str(path))
                    font_name = font_prop.get_name()
                    
                    # 强制设置全局字体（正确方式）
                    plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    self.logger.info(f"字体加载成功: {path}, 字体名: {font_name}")
                    self.font_prop = font_prop
                    self.font_name = font_name
                    return
                except Exception as e:
                    self.logger.warning(f"字体加载失败: {e}")
                    continue
        
        # 使用系统默认中文字体
        self.logger.warning(f"字体文件未找到: {font_path}，使用系统默认")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        self.font_prop = None
        self.font_name = None
    
    def _setup_style(self) -> None:
        """设置图表样式"""
        if SEABORN_AVAILABLE:
            sns.set_style(self.config.style)
            sns.set_palette([COLORS["primary"], COLORS["secondary"], COLORS["accent"]])
        
        # matplotlib 基本配置
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        plt.rcParams['figure.figsize'] = self.config.figure_size
    
    def _apply_font_to_figure(self, fig) -> None:
        """
        将字体应用到图形的所有文本元素
        
        参数：
            fig: matplotlib Figure对象
        """
        if self.font_prop is None:
            return
        
        # 遍历所有文本元素
        for text in fig.findobj(lambda x: hasattr(x, 'set_fontproperties')):
            try:
                text.set_fontproperties(self.font_prop)
            except:
                pass
    
    def _ensure_output_dirs(self) -> None:
        """确保输出目录存在"""
        output_dir = Path(self.config.output_dir)
        
        # 创建子目录
        subdirs = ["pre_training", "during_training", "post_training"]
        for subdir in subdirs:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _get_output_path(self, filename: str, stage: str = "pre_training") -> Path:
        """
        获取输出文件路径
        
        参数：
            filename: 文件名
            stage: 阶段（pre_training, during_training, post_training）
            
        返回：
            Path: 完整路径
        """
        return Path(self.config.output_dir) / stage / filename
    
    def plot_length_distribution(
        self,
        samples: List[Dict[str, str]],
        output_filename: str = "pre_data_distribution.png",
        zh_key: str = "chinese",
        en_key: str = "english",
        title: str = "数据长度分布"
    ) -> Optional[str]:
        """
        绑制句子长度分布图
        
        参数：
            samples: 样本列表
            output_filename: 输出文件名
            zh_key: 中文字段名
            en_key: 英文字段名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib 不可用")
            return None
        
        # 统计长度
        zh_lengths = [len(s.get(zh_key, "")) for s in samples]
        en_lengths = [len(s.get(en_key, "")) for s in samples]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 中文长度分布（直方图）
        ax1 = axes[0, 0]
        ax1.hist(zh_lengths, bins=50, color=COLORS["primary"], alpha=0.7, edgecolor='white')
        ax1.set_xlabel("中文字符数")
        ax1.set_ylabel("样本数量")
        ax1.set_title("中文长度分布")
        ax1.axvline(np.mean(zh_lengths), color=COLORS["accent"], linestyle='--', 
                    label=f'均值: {np.mean(zh_lengths):.1f}')
        ax1.legend()
        
        # 英文长度分布（直方图）
        ax2 = axes[0, 1]
        ax2.hist(en_lengths, bins=50, color=COLORS["secondary"], alpha=0.7, edgecolor='white')
        ax2.set_xlabel("英文字符数")
        ax2.set_ylabel("样本数量")
        ax2.set_title("英文长度分布")
        ax2.axvline(np.mean(en_lengths), color=COLORS["accent"], linestyle='--',
                    label=f'均值: {np.mean(en_lengths):.1f}')
        ax2.legend()
        
        # 中英长度对比（箱线图）
        ax3 = axes[1, 0]
        box_data = [zh_lengths, en_lengths]
        bp = ax3.boxplot(box_data, tick_labels=['中文', '英文'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS["primary"])
        bp['boxes'][1].set_facecolor(COLORS["secondary"])
        ax3.set_ylabel("字符数")
        ax3.set_title("中英文长度对比")
        
        # 长度比例分布
        ax4 = axes[1, 1]
        ratios = [zh / en if en > 0 else 0 for zh, en in zip(zh_lengths, en_lengths)]
        ax4.hist(ratios, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor='white')
        ax4.set_xlabel("中英长度比例")
        ax4.set_ylabel("样本数量")
        ax4.set_title("长度比例分布")
        ax4.axvline(np.mean(ratios), color=COLORS["primary"], linestyle='--',
                    label=f'均值: {np.mean(ratios):.2f}')
        ax4.legend()
        
        # 添加总标题
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 应用字体到所有文本元素
        self._apply_font_to_figure(fig)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"长度分布图已保存: {output_path}")
        return str(output_path)
    
    def plot_cleaning_comparison(
        self,
        before_count: int,
        after_count: int,
        cleaning_stats: Dict[str, int],
        output_filename: str = "pre_cleaning_comparison.png",
        title: str = "数据清洗对比"
    ) -> Optional[str]:
        """
        绑制清洗前后对比图
        
        参数：
            before_count: 清洗前样本数
            after_count: 清洗后样本数
            cleaning_stats: 各类过滤的统计
            output_filename: 输出文件名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 清洗前后数量对比
        ax1 = axes[0]
        bars = ax1.bar(['清洗前', '清洗后'], [before_count, after_count],
                       color=[COLORS["secondary"], COLORS["primary"]])
        ax1.set_ylabel("样本数量")
        ax1.set_title("样本数量变化")
        
        # 添加数值标签
        for bar, count in zip(bars, [before_count, after_count]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + before_count*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=12)
        
        # 保留率
        retention = after_count / before_count * 100 if before_count > 0 else 0
        ax1.text(0.5, 0.9, f'保留率: {retention:.1f}%', transform=ax1.transAxes,
                fontsize=14, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 过滤原因分布（饼图）
        ax2 = axes[1]
        filter_labels = list(cleaning_stats.keys())
        filter_values = list(cleaning_stats.values())
        
        # 过滤掉零值
        non_zero = [(l, v) for l, v in zip(filter_labels, filter_values) if v > 0]
        if non_zero:
            labels, values = zip(*non_zero)
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax2.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90
            )
            ax2.set_title("过滤原因分布")
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        self._apply_font_to_figure(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"清洗对比图已保存: {output_path}")
        return str(output_path)
    
    def plot_filtering_effect(
        self,
        similarity_scores: List[float],
        threshold: float = 0.7,
        output_filename: str = "pre_filtering_effect.png",
        title: str = "数据筛选效果"
    ) -> Optional[str]:
        """
        绑制数据筛选效果图
        
        参数：
            similarity_scores: 相似度分数列表
            threshold: 筛选阈值
            output_filename: 输出文件名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 相似度分布
        ax1 = axes[0]
        ax1.hist(similarity_scores, bins=50, color=COLORS["primary"], 
                 alpha=0.7, edgecolor='white')
        ax1.axvline(threshold, color=COLORS["accent"], linestyle='--', linewidth=2,
                   label=f'阈值: {threshold}')
        ax1.axvline(np.mean(similarity_scores), color=COLORS["secondary"], 
                   linestyle=':', linewidth=2,
                   label=f'均值: {np.mean(similarity_scores):.3f}')
        ax1.set_xlabel("语义相似度")
        ax1.set_ylabel("样本数量")
        ax1.set_title("相似度分布")
        ax1.legend()
        
        # 筛选前后对比
        ax2 = axes[1]
        above_threshold = sum(1 for s in similarity_scores if s >= threshold)
        below_threshold = len(similarity_scores) - above_threshold
        
        bars = ax2.bar(['保留', '过滤'], [above_threshold, below_threshold],
                      color=[COLORS["success"], COLORS["hard"]])
        ax2.set_ylabel("样本数量")
        ax2.set_title("筛选结果")
        
        for bar, count in zip(bars, [above_threshold, below_threshold]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=12)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        self._apply_font_to_figure(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"筛选效果图已保存: {output_path}")
        return str(output_path)
    
    def plot_difficulty_distribution(
        self,
        difficulty_counts: Dict[str, int],
        output_filename: str = "pre_difficulty_distribution.png",
        title: str = "难度分布统计"
    ) -> Optional[str]:
        """
        绑制难度分布图
        
        参数：
            difficulty_counts: 各难度级别的样本数量
            output_filename: 输出文件名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 柱状图
        ax1 = axes[0]
        difficulties = ['easy', 'medium', 'hard']
        counts = [difficulty_counts.get(d, 0) for d in difficulties]
        colors = [COLORS["easy"], COLORS["medium"], COLORS["hard"]]
        
        bars = ax1.bar(['简单', '中等', '困难'], counts, color=colors)
        ax1.set_ylabel("样本数量")
        ax1.set_title("各难度级别样本数")
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=11)
        
        # 饼图
        ax2 = axes[1]
        if sum(counts) > 0:
            wedges, texts, autotexts = ax2.pie(
                counts, labels=['简单', '中等', '困难'], autopct='%1.1f%%',
                colors=colors, startangle=90, explode=(0.02, 0.02, 0.02)
            )
            ax2.set_title("难度比例")
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        self._apply_font_to_figure(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"难度分布图已保存: {output_path}")
        return str(output_path)
    
    def plot_bpe_tokenization(
        self,
        text_samples: List[Tuple[str, List[str]]],
        output_filename: str = "pre_bpe_tokenization.png",
        title: str = "BPE 分词效果展示"
    ) -> Optional[str]:
        """
        绘制 BPE 分词效果图
        
        参数：
            text_samples: [(原文, [子词列表]), ...] 列表
            output_filename: 输出文件名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        num_samples = min(len(text_samples), 5)
        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))
        
        if num_samples == 1:
            axes = [axes]
        
        for idx, (ax, (text, tokens)) in enumerate(zip(axes, text_samples[:num_samples])):
            # 显示原文
            ax.text(0.02, 0.7, f"原文: {text[:60]}{'...' if len(text) > 60 else ''}",
                   transform=ax.transAxes, fontsize=11, verticalalignment='top')
            
            # 显示分词结果
            token_str = ' | '.join(tokens[:15])
            if len(tokens) > 15:
                token_str += f' ... (+{len(tokens)-15})'
            ax.text(0.02, 0.35, f"子词: {token_str}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   color=COLORS["primary"])
            
            # 显示统计信息
            ax.text(0.02, 0.1, f"子词数: {len(tokens)}  |  压缩率: {len(text)/len(tokens):.2f}",
                   transform=ax.transAxes, fontsize=10, color=COLORS["secondary"])
            
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        self._apply_font_to_figure(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"BPE 分词图已保存: {output_path}")
        return str(output_path)
    
    def plot_word_frequency(
        self,
        word_counts: Dict[str, int],
        top_n: int = 30,
        output_filename: str = "pre_word_frequency.png",
        title: str = "高频词统计"
    ) -> Optional[str]:
        """
        绘制词频统计图
        
        参数：
            word_counts: 词频统计字典
            top_n: 显示前 N 个高频词
            output_filename: 输出文件名
            title: 图表标题
            
        返回：
            str: 输出文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # 获取 top N
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, counts = zip(*sorted_words) if sorted_words else ([], [])
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 水平柱状图
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, counts, color=COLORS["primary"], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("出现次数")
        ax.set_title(title)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=9)
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        
        output_path = self._get_output_path(output_filename)
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"词频统计图已保存: {output_path}")
        return str(output_path)
    
    def generate_data_report(
        self,
        samples: List[Dict[str, str]],
        cleaning_stats: Optional[Dict[str, int]] = None,
        similarity_scores: Optional[List[float]] = None,
        difficulty_counts: Optional[Dict[str, int]] = None,
        bpe_samples: Optional[List[Tuple[str, List[str]]]] = None,
        zh_key: str = "chinese",
        en_key: str = "english"
    ) -> Dict[str, str]:
        """
        生成完整的数据报告（所有训练前可视化）
        
        参数：
            samples: 样本列表
            cleaning_stats: 清洗统计
            similarity_scores: 相似度分数
            difficulty_counts: 难度分布
            bpe_samples: BPE 分词样本
            zh_key: 中文字段名
            en_key: 英文字段名
            
        返回：
            Dict[str, str]: 生成的图表文件路径
        """
        generated_files = {}
        
        # 1. 长度分布图
        path = self.plot_length_distribution(samples, zh_key=zh_key, en_key=en_key)
        if path:
            generated_files["length_distribution"] = path
        
        # 2. 清洗对比图
        if cleaning_stats:
            total = cleaning_stats.get("原始样本数", len(samples))
            final = cleaning_stats.get("最终保留", len(samples))
            path = self.plot_cleaning_comparison(total, final, cleaning_stats)
            if path:
                generated_files["cleaning_comparison"] = path
        
        # 3. 筛选效果图
        if similarity_scores:
            path = self.plot_filtering_effect(similarity_scores)
            if path:
                generated_files["filtering_effect"] = path
        
        # 4. 难度分布图
        if difficulty_counts:
            path = self.plot_difficulty_distribution(difficulty_counts)
            if path:
                generated_files["difficulty_distribution"] = path
        
        # 5. BPE 分词图
        if bpe_samples:
            path = self.plot_bpe_tokenization(bpe_samples)
            if path:
                generated_files["bpe_tokenization"] = path
        
        self.logger.info(f"数据报告生成完成，共 {len(generated_files)} 个图表")
        return generated_files


# ====================================
# 便捷函数
# ====================================

def create_data_visualizations(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    font_path: Optional[str] = None,
    zh_key: str = "chinese",
    en_key: str = "english"
) -> Dict[str, str]:
    """
    一键生成数据可视化
    
    参数：
        data_path: 数据文件路径（JSONL）
        output_dir: 输出目录
        font_path: 字体文件路径
        zh_key: 中文字段名
        en_key: 英文字段名
        
    返回：
        Dict[str, str]: 生成的图表路径
    """
    # 加载数据
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # 创建可视化器
    config = VisualizationConfig(output_dir=str(output_dir))
    if font_path:
        config.font_path = font_path
    
    visualizer = DataVisualizer(config)
    
    # 统计难度分布
    difficulty_counts = Counter()
    similarity_scores = []
    
    for sample in samples:
        difficulty = sample.get("difficulty", "unknown")
        difficulty_counts[difficulty] += 1
        
        if "similarity_score" in sample:
            similarity_scores.append(sample["similarity_score"])
    
    # 生成报告
    return visualizer.generate_data_report(
        samples=samples,
        similarity_scores=similarity_scores if similarity_scores else None,
        difficulty_counts=dict(difficulty_counts) if difficulty_counts else None,
        zh_key=zh_key,
        en_key=en_key
    )


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="数据可视化工具"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="数据文件路径（JSONL 格式）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）"
    )
    parser.add_argument(
        "--font",
        type=str,
        default=DEFAULT_FONT_PATH,
        help=f"字体文件路径（默认: {DEFAULT_FONT_PATH}）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 生成可视化
    files = create_data_visualizations(
        data_path=args.data,
        output_dir=args.output,
        font_path=args.font
    )
    
    print("\n生成的可视化文件:")
    for name, path in files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
