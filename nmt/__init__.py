"""
NMT - 工业级神经网络翻译系统

功能模块：
    - data: 数据处理（清洗、筛选、分词、课程学习）
    - model: 模型定义与配置
    - training: 训练器与训练脚本
    - compression: 模型压缩（剪枝、量化）
    - inference: 推理引擎（ONNX Runtime、Prompt Cache）
    - evaluation: 质量评估（多元指标）
    - visualization: 可视化图表生成
    - utils: 工具函数

作者：毕业设计项目
版本：1.0.0
"""

# 必须在导入其他模块前设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

__version__ = "1.0.0"
__author__ = "Graduation Project"

# 导入子模块
from . import data
from . import model
from . import training
from . import compression
from . import inference
from . import evaluation
from . import visualization
from . import utils
