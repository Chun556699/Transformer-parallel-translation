#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练可视化模块

功能说明：
    - 训练过程实时监控图表
    - 损失曲线可视化
    - 学习率调度可视化
    - 梯度分布直方图
    - 注意力热力图
    - 训练进度追踪

输出图表：
    - train_loss_curve.png: 训练/验证损失曲线
    - train_lr_schedule.png: 学习率变化曲线
    - train_gradient_distribution.png: 梯度分布直方图
    - train_attention_heatmap.png: 注意力热力图
    - train_progress.png: 训练进度图

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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================================
# 日志配置
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置
# ============================================================================

# 默认中文字体路径
DEFAULT_FONT_PATH = "e:\\Graduation-Project\\SourceHanSansSC-Regular-2.otf"

# 图表样式配置
PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

# 颜色方案
COLORS = {
    'train': '#3498db',      # 蓝色
    'valid': '#e74c3c',      # 红色
    'lr': '#2ecc71',         # 绿色
    'gradient': '#9b59b6',   # 紫色
    'attention': 'YlOrRd',   # 热力图配色
}


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TrainingMetrics:
    """
    训练指标数据类
    
    属性：
        steps: 步数列表
        train_loss: 训练损失
        valid_loss: 验证损失
        learning_rate: 学习率
        gradient_norms: 梯度范数
        epoch_times: 每个 epoch 耗时
    """
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    valid_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def add_step(self, step: int, train_loss: float, 
                 lr: float, grad_norm: float = None):
        """添加一步训练数据"""
        self.steps.append(step)
        self.train_loss.append(train_loss)
        self.learning_rate.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
    
    def add_valid_loss(self, loss: float):
        """添加验证损失"""
        self.valid_loss.append(loss)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'steps': self.steps,
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'learning_rate': self.learning_rate,
            'gradient_norms': self.gradient_norms,
            'epoch_times': self.epoch_times,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """从字典创建"""
        return cls(
            steps=data.get('steps', []),
            train_loss=data.get('train_loss', []),
            valid_loss=data.get('valid_loss', []),
            learning_rate=data.get('learning_rate', []),
            gradient_norms=data.get('gradient_norms', []),
            epoch_times=data.get('epoch_times', []),
        )
    
    @classmethod
    def from_json(cls, path: str) -> 'TrainingMetrics':
        """从 JSON 文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, path: str) -> None:
        """保存为 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# 可视化类
# ============================================================================

class TrainingVisualizer:
    """
    训练可视化器
    
    功能：
        - 生成训练过程的各类图表
        - 支持中文字体
        - 自动保存高分辨率图片
    
    示例：
        >>> visualizer = TrainingVisualizer(output_dir='outputs/viz')
        >>> visualizer.plot_loss_curve(metrics)
        >>> visualizer.plot_lr_schedule(metrics)
    """
    
    def __init__(self, 
                 output_dir: str = "outputs/visualizations/during_training",
                 font_path: str = DEFAULT_FONT_PATH):
        """
        初始化可视化器
        
        参数：
            output_dir: 图表输出目录
            font_path: 中文字体路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置字体
        self._setup_font(font_path)
        
        # 应用样式
        plt.rcParams.update(PLOT_STYLE)
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
                    
                    logger.info(f"已加载字体: {p}, 字体名: {font_name}")
                    font_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"字体加载失败: {e}")
                    continue
        
        if not font_loaded:
            self.font_prop = None
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            logger.warning(f"字体文件不存在，使用系统字体")
    
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
    
    def plot_loss_curve(self, 
                        metrics: TrainingMetrics,
                        title: str = "训练损失曲线",
                        save_name: str = "train_loss_curve.png") -> str:
        """
        绘制训练/验证损失曲线
        
        参数：
            metrics: 训练指标数据
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制训练损失
        if metrics.train_loss:
            ax.plot(metrics.steps, metrics.train_loss, 
                   color=COLORS['train'], label='训练损失', alpha=0.8)
            
            # 添加平滑曲线
            if len(metrics.train_loss) > 50:
                smoothed = self._smooth_curve(metrics.train_loss)
                ax.plot(metrics.steps, smoothed, 
                       color=COLORS['train'], linestyle='--', 
                       label='训练损失 (平滑)', linewidth=2.5)
        
        # 绘制验证损失
        if metrics.valid_loss:
            # 验证损失通常是按 epoch 记录的
            valid_steps = np.linspace(0, metrics.steps[-1] if metrics.steps else 0, 
                                     len(metrics.valid_loss))
            ax.plot(valid_steps, metrics.valid_loss, 
                   color=COLORS['valid'], label='验证损失', 
                   marker='o', markersize=6, linewidth=2)
        
        # 设置标签
        ax.set_xlabel('训练步数', **self._get_font_kwargs())
        ax.set_ylabel('损失值', **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        ax.legend(prop=self.font_prop if self.font_prop else None)
        
        # 添加最佳损失标记
        if metrics.valid_loss:
            best_idx = np.argmin(metrics.valid_loss)
            best_loss = metrics.valid_loss[best_idx]
            ax.axhline(y=best_loss, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'最佳验证损失: {best_loss:.4f}',
                       xy=(valid_steps[best_idx], best_loss),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, **self._get_font_kwargs())
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"损失曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_lr_schedule(self,
                         metrics: TrainingMetrics,
                         title: str = "学习率调度曲线",
                         save_name: str = "train_lr_schedule.png") -> str:
        """
        绘制学习率变化曲线
        
        参数：
            metrics: 训练指标数据
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        if not metrics.learning_rate:
            logger.warning("无学习率数据，跳过绘图")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(metrics.steps, metrics.learning_rate, 
               color=COLORS['lr'], linewidth=2)
        
        # 标记关键点
        warmup_end = len(metrics.learning_rate) // 10  # 假设 warmup 占 10%
        if warmup_end > 0 and warmup_end < len(metrics.steps):
            ax.axvline(x=metrics.steps[warmup_end], color='gray', 
                      linestyle='--', alpha=0.5, label='Warmup 结束')
        
        # 设置标签
        ax.set_xlabel('训练步数', **self._get_font_kwargs())
        ax.set_ylabel('学习率', **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        ax.legend(prop=self.font_prop if self.font_prop else None)
        
        # 使用科学计数法显示 y 轴
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
        
        # 添加学习率区间标注
        max_lr = max(metrics.learning_rate)
        min_lr = min(metrics.learning_rate)
        ax.annotate(f'最大: {max_lr:.2e}', xy=(0.02, 0.95), 
                   xycoords='axes fraction', fontsize=10,
                   **self._get_font_kwargs())
        ax.annotate(f'最小: {min_lr:.2e}', xy=(0.02, 0.88), 
                   xycoords='axes fraction', fontsize=10,
                   **self._get_font_kwargs())
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"学习率曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_gradient_distribution(self,
                                   gradient_data: List[np.ndarray],
                                   layer_names: List[str] = None,
                                   title: str = "梯度分布直方图",
                                   save_name: str = "train_gradient_distribution.png") -> str:
        """
        绘制梯度分布直方图
        
        参数：
            gradient_data: 各层梯度数据
            layer_names: 层名称列表
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        if not gradient_data:
            logger.warning("无梯度数据，跳过绘图")
            return ""
        
        n_layers = min(len(gradient_data), 6)  # 最多显示 6 层
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_layers):
            ax = axes[i]
            data = gradient_data[i].flatten()
            
            # 绘制直方图
            ax.hist(data, bins=50, color=COLORS['gradient'], 
                   alpha=0.7, edgecolor='white')
            
            # 添加统计信息
            mean = np.mean(data)
            std = np.std(data)
            ax.axvline(x=mean, color='red', linestyle='--', 
                      label=f'均值: {mean:.2e}')
            ax.axvline(x=mean + std, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(x=mean - std, color='orange', linestyle=':', alpha=0.7)
            
            # 设置标签
            layer_name = layer_names[i] if layer_names else f'Layer {i+1}'
            ax.set_title(layer_name, fontsize=12, **self._get_font_kwargs())
            ax.set_xlabel('梯度值', fontsize=10, **self._get_font_kwargs())
            ax.set_ylabel('频数', fontsize=10, **self._get_font_kwargs())
            ax.legend(fontsize=8, prop=self.font_prop if self.font_prop else None)
        
        # 隐藏多余的子图
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, **self._get_font_kwargs())
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"梯度分布图已保存: {save_path}")
        return str(save_path)
    
    def plot_attention_heatmap(self,
                               attention_weights: np.ndarray,
                               source_tokens: List[str],
                               target_tokens: List[str],
                               title: str = "注意力权重热力图",
                               save_name: str = "train_attention_heatmap.png") -> str:
        """
        绘制注意力热力图
        
        参数：
            attention_weights: 注意力权重矩阵 (target_len, source_len)
            source_tokens: 源序列 token
            target_tokens: 目标序列 token
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图
        sns.heatmap(attention_weights, 
                   xticklabels=source_tokens,
                   yticklabels=target_tokens,
                   cmap=COLORS['attention'],
                   annot=False,
                   fmt='.2f',
                   ax=ax,
                   cbar_kws={'label': '注意力权重'})
        
        # 设置标签
        ax.set_xlabel('源序列', fontsize=12, **self._get_font_kwargs())
        ax.set_ylabel('目标序列', fontsize=12, **self._get_font_kwargs())
        ax.set_title(title, fontsize=16, **self._get_font_kwargs())
        
        # 旋转 x 轴标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"注意力热力图已保存: {save_path}")
        return str(save_path)
    
    def plot_training_progress(self,
                               metrics: TrainingMetrics,
                               total_steps: int,
                               total_epochs: int,
                               current_epoch: int,
                               title: str = "训练进度",
                               save_name: str = "train_progress.png") -> str:
        """
        绘制训练进度图
        
        参数：
            metrics: 训练指标数据
            total_steps: 总步数
            total_epochs: 总 epoch 数
            current_epoch: 当前 epoch
            title: 图表标题
            save_name: 保存文件名
            
        返回：
            str: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        current_step = metrics.steps[-1] if metrics.steps else 0
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        
        # 1. 进度条
        ax1 = axes[0, 0]
        ax1.barh(['进度'], [progress_pct], color=COLORS['train'], height=0.5)
        ax1.barh(['进度'], [100 - progress_pct], left=[progress_pct], 
                color='lightgray', height=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('完成百分比 (%)', **self._get_font_kwargs())
        ax1.set_title(f'训练进度: {progress_pct:.1f}%', fontsize=12, 
                     **self._get_font_kwargs())
        ax1.axvline(x=progress_pct, color='red', linestyle='--', alpha=0.5)
        
        # 2. Epoch 进度
        ax2 = axes[0, 1]
        epoch_progress = [(i, 'completed' if i < current_epoch else 
                          'current' if i == current_epoch else 'pending') 
                         for i in range(total_epochs)]
        colors = {'completed': COLORS['train'], 
                 'current': COLORS['valid'], 
                 'pending': 'lightgray'}
        bars = ax2.bar([str(i+1) for i, _ in epoch_progress],
                      [1] * len(epoch_progress),
                      color=[colors[status] for _, status in epoch_progress])
        ax2.set_xlabel('Epoch', **self._get_font_kwargs())
        ax2.set_ylabel('状态', **self._get_font_kwargs())
        ax2.set_title(f'Epoch 进度: {current_epoch}/{total_epochs}', 
                     fontsize=12, **self._get_font_kwargs())
        ax2.set_ylim(0, 1.2)
        ax2.set_yticks([])
        
        # 3. 损失趋势
        ax3 = axes[1, 0]
        if metrics.train_loss:
            recent_loss = metrics.train_loss[-min(100, len(metrics.train_loss)):]
            ax3.plot(recent_loss, color=COLORS['train'], linewidth=2)
            ax3.set_xlabel('最近步数', **self._get_font_kwargs())
            ax3.set_ylabel('损失值', **self._get_font_kwargs())
            ax3.set_title('近期损失趋势', fontsize=12, **self._get_font_kwargs())
        
        # 4. 时间估算
        ax4 = axes[1, 1]
        if metrics.epoch_times:
            avg_epoch_time = np.mean(metrics.epoch_times)
            remaining_epochs = total_epochs - current_epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            time_data = {
                '已用时间': sum(metrics.epoch_times),
                '预计剩余': estimated_remaining,
            }
            
            bars = ax4.bar(list(time_data.keys()), list(time_data.values()),
                          color=[COLORS['train'], 'lightgray'])
            ax4.set_ylabel('时间 (分钟)', **self._get_font_kwargs())
            ax4.set_title('时间统计', fontsize=12, **self._get_font_kwargs())
            
            # 添加数值标签
            for bar, val in zip(bars, time_data.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}分钟', ha='center', fontsize=10,
                        **self._get_font_kwargs())
        else:
            ax4.text(0.5, 0.5, '暂无时间数据', ha='center', va='center',
                    fontsize=12, **self._get_font_kwargs(),
                    transform=ax4.transAxes)
            ax4.set_axis_off()
        
        fig.suptitle(title, fontsize=16, **self._get_font_kwargs())
        self._apply_font_to_figure(fig)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"训练进度图已保存: {save_path}")
        return str(save_path)
    
    def _smooth_curve(self, values: List[float], weight: float = 0.9) -> List[float]:
        """
        指数移动平均平滑曲线
        
        参数：
            values: 原始数据
            weight: 平滑权重
            
        返回：
            List[float]: 平滑后的数据
        """
        smoothed = []
        last = values[0]
        for val in values:
            smoothed_val = last * weight + (1 - weight) * val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def generate_all_plots(self, 
                           metrics: TrainingMetrics,
                           gradient_data: List[np.ndarray] = None,
                           attention_data: Tuple[np.ndarray, List[str], List[str]] = None,
                           total_steps: int = 0,
                           total_epochs: int = 0,
                           current_epoch: int = 0) -> List[str]:
        """
        生成所有训练图表
        
        参数：
            metrics: 训练指标
            gradient_data: 梯度数据
            attention_data: 注意力数据 (weights, source_tokens, target_tokens)
            total_steps: 总步数
            total_epochs: 总 epoch
            current_epoch: 当前 epoch
            
        返回：
            List[str]: 生成的文件路径列表
        """
        saved_files = []
        
        # 损失曲线
        path = self.plot_loss_curve(metrics)
        if path:
            saved_files.append(path)
        
        # 学习率曲线
        path = self.plot_lr_schedule(metrics)
        if path:
            saved_files.append(path)
        
        # 梯度分布
        if gradient_data:
            path = self.plot_gradient_distribution(gradient_data)
            if path:
                saved_files.append(path)
        
        # 注意力热力图
        if attention_data:
            weights, src_tokens, tgt_tokens = attention_data
            path = self.plot_attention_heatmap(weights, src_tokens, tgt_tokens)
            if path:
                saved_files.append(path)
        
        # 训练进度
        if total_steps > 0:
            path = self.plot_training_progress(
                metrics, total_steps, total_epochs, current_epoch
            )
            if path:
                saved_files.append(path)
        
        logger.info(f"共生成 {len(saved_files)} 个图表")
        return saved_files


# ============================================================================
# 便捷函数
# ============================================================================

def create_visualizer(output_dir: str = None, 
                      font_path: str = None) -> TrainingVisualizer:
    """
    创建训练可视化器
    
    参数：
        output_dir: 输出目录
        font_path: 字体路径
        
    返回：
        TrainingVisualizer: 可视化器实例
    """
    kwargs = {}
    if output_dir:
        kwargs['output_dir'] = output_dir
    if font_path:
        kwargs['font_path'] = font_path
    return TrainingVisualizer(**kwargs)


def plot_training_metrics(metrics_file: str, 
                          output_dir: str = None) -> List[str]:
    """
    从文件加载指标并生成图表
    
    参数：
        metrics_file: 指标 JSON 文件路径
        output_dir: 输出目录
        
    返回：
        List[str]: 生成的文件路径列表
    """
    metrics = TrainingMetrics.from_json(metrics_file)
    visualizer = create_visualizer(output_dir)
    return visualizer.generate_all_plots(metrics)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'TrainingMetrics',
    'TrainingVisualizer',
    'create_visualizer',
    'plot_training_metrics',
]


# ============================================================================
# 主函数（测试用）
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练可视化工具')
    parser.add_argument('--metrics-file', type=str, help='指标文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='e:\\Graduation-Project\\outputs\\visualizations\\during_training',
                       help='输出目录')
    parser.add_argument('--demo', action='store_true', help='生成演示图表')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.demo:
        # 生成演示数据
        print("生成演示图表...")
        
        metrics = TrainingMetrics()
        for i in range(1000):
            loss = 2.0 * np.exp(-i / 300) + 0.1 + np.random.normal(0, 0.05)
            lr = 5e-5 * min(i / 100, 1) * (1 - i / 1000)
            metrics.add_step(i, loss, lr, np.random.uniform(0.1, 1.0))
        
        for epoch in range(10):
            metrics.add_valid_loss(1.8 * np.exp(-epoch / 3) + 0.15)
            metrics.epoch_times.append(np.random.uniform(5, 10))
        
        visualizer = TrainingVisualizer(output_dir=args.output_dir)
        files = visualizer.generate_all_plots(
            metrics,
            total_steps=1000,
            total_epochs=10,
            current_epoch=5
        )
        
        print(f"已生成 {len(files)} 个图表:")
        for f in files:
            print(f"  - {f}")
    
    elif args.metrics_file:
        files = plot_training_metrics(args.metrics_file, args.output_dir)
        print(f"已生成 {len(files)} 个图表")
