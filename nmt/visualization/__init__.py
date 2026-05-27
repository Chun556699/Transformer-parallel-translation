"""
可视化模块

功能说明：
    提供全程可视化功能（训练前/中/后对比），包括：
    - 数据可视化（data_viz.py）
    - 训练过程可视化（training_viz.py）
    - 评估结果可视化（evaluation_viz.py）

模块组成：
    - data_viz: 数据阶段可视化
    - training_viz: 训练阶段可视化
    - evaluation_viz: 评估阶段可视化
    - plots: 综合可视化管理器

作者：NMT Project
版本：1.0.0
"""

from .data_viz import (
    DataVisualizer,
    VisualizationConfig,
    create_data_visualizations,
    COLORS,
)

from .training_viz import (
    TrainingMetrics,
    TrainingVisualizer,
    create_visualizer as create_training_visualizer,
    plot_training_metrics,
)

from .post_training_viz import (
    ModelComparisonData,
    PostTrainingVisualizer,
    create_post_visualizer,
)

# 从 data_viz 导入 plots（如果存在）
try:
    from .data_viz import plots
except ImportError:
    plots = None

__all__ = [
    # 数据可视化
    "DataVisualizer",
    "VisualizationConfig",
    "create_data_visualizations",
    "COLORS",
    
    # 训练可视化
    "TrainingMetrics",
    "TrainingVisualizer",
    "create_training_visualizer",
    "plot_training_metrics",
    
    # 训练后可视化
    "ModelComparisonData",
    "PostTrainingVisualizer",
    "create_post_visualizer",
    
    # 综合管理器
    "VisualizationManager",
    "plots",
]


class VisualizationManager:
    """
    可视化综合管理器
    
    功能说明：
        统一管理所有可视化功能，支持：
        - 训练前数据可视化
        - 训练中实时监控
        - 训练后评估对比
    
    参数：
        output_dir: 输出目录
        font_path: 字体文件路径
        
    示例：
        >>> manager = VisualizationManager(output_dir="outputs/visualizations")
        >>> manager.plot_data_distribution(samples)
        >>> manager.plot_training_progress(logs)
        >>> manager.plot_evaluation_results(metrics)
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/visualizations",
        font_path: str = "SourceHanSansSC-Regular-2.otf"
    ):
        """
        初始化可视化管理器
        """
        self.output_dir = output_dir
        self.font_path = font_path
        
        # 初始化数据可视化器
        config = VisualizationConfig(
            output_dir=output_dir,
            font_path=font_path
        )
        self.data_visualizer = DataVisualizer(config)
    
    def plot_data_distribution(self, samples, **kwargs):
        """绑制数据分布"""
        return self.data_visualizer.plot_length_distribution(samples, **kwargs)
    
    def plot_cleaning_comparison(self, before, after, stats, **kwargs):
        """绑制清洗对比"""
        return self.data_visualizer.plot_cleaning_comparison(before, after, stats, **kwargs)
    
    def plot_filtering_effect(self, scores, **kwargs):
        """绑制筛选效果"""
        return self.data_visualizer.plot_filtering_effect(scores, **kwargs)
    
    def plot_difficulty_distribution(self, counts, **kwargs):
        """绑制难度分布"""
        return self.data_visualizer.plot_difficulty_distribution(counts, **kwargs)
    
    def generate_data_report(self, samples, **kwargs):
        """生成数据报告"""
        return self.data_visualizer.generate_data_report(samples, **kwargs)
