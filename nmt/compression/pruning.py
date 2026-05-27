"""
结构化剪枝模块

功能说明：
    实现 MarianMT 模型的结构化剪枝，包括：
    - FFN 中间层剪枝（目标：70%，2048→614）
    - 注意力头剪枝（目标：50%，8→4 heads）
    - L1 范数重要性评估
    - 迭代剪枝 + 微调恢复

压缩目标：
    - 原始模型：~300MB
    - 剪枝后：~100MB（配合量化）

依赖：
    - torch: PyTorch 核心
    - transformers: MarianMT 模型

作者：NMT Project
版本：1.0.0
"""

import os
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from transformers import MarianMTModel, MarianConfig, MarianTokenizer


# ====================================
# 常量定义
# ====================================

# 默认剪枝比例
DEFAULT_FFN_PRUNE_RATIO = 0.7    # 70% FFN 剪枝
DEFAULT_HEAD_PRUNE_RATIO = 0.5   # 50% 注意力头剪枝

# MarianMT 默认配置
DEFAULT_FFN_DIM = 2048
DEFAULT_NUM_HEADS = 8
DEFAULT_D_MODEL = 512


@dataclass
class PruningConfig:
    """
    剪枝配置数据类
    
    属性：
        ffn_prune_ratio: FFN 剪枝比例
        head_prune_ratio: 注意力头剪枝比例
        prune_encoder: 是否剪枝编码器
        prune_decoder: 是否剪枝解码器
        importance_metric: 重要性评估方法
        iterative_steps: 迭代剪枝步数
    """
    ffn_prune_ratio: float = DEFAULT_FFN_PRUNE_RATIO
    head_prune_ratio: float = DEFAULT_HEAD_PRUNE_RATIO
    prune_encoder: bool = True
    prune_decoder: bool = True
    importance_metric: str = "l1"  # l1, l2, magnitude
    iterative_steps: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ffn_prune_ratio": self.ffn_prune_ratio,
            "head_prune_ratio": self.head_prune_ratio,
            "prune_encoder": self.prune_encoder,
            "prune_decoder": self.prune_decoder,
            "importance_metric": self.importance_metric,
            "iterative_steps": self.iterative_steps,
        }


@dataclass
class PruningStats:
    """
    剪枝统计信息
    
    属性：
        original_params: 原始参数数量
        pruned_params: 剪枝后参数数量
        original_size_mb: 原始模型大小
        pruned_size_mb: 剪枝后模型大小
        compression_ratio: 压缩比例
        pruned_heads: 剪枝的注意力头数
        pruned_ffn_neurons: 剪枝的 FFN 神经元数
    """
    original_params: int = 0
    pruned_params: int = 0
    original_size_mb: float = 0.0
    pruned_size_mb: float = 0.0
    compression_ratio: float = 0.0
    pruned_heads: Dict[str, int] = field(default_factory=dict)
    pruned_ffn_neurons: Dict[str, int] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"剪枝统计:\n"
            f"  原始参数: {self.original_params:,}\n"
            f"  剪枝后参数: {self.pruned_params:,}\n"
            f"  原始大小: {self.original_size_mb:.2f} MB\n"
            f"  剪枝后大小: {self.pruned_size_mb:.2f} MB\n"
            f"  压缩比例: {self.compression_ratio:.2%}\n"
            f"  剪枝注意力头: {sum(self.pruned_heads.values())}\n"
            f"  剪枝 FFN 神经元: {sum(self.pruned_ffn_neurons.values())}"
        )


class StructuredPruner:
    """
    结构化剪枝器
    
    功能说明：
        对 MarianMT 模型进行结构化剪枝：
        - 基于 L1 范数评估神经元重要性
        - 剪枝 FFN 中间层神经元
        - 剪枝注意力头
        - 支持迭代剪枝
    
    参数：
        model: MarianMT 模型
        config: 剪枝配置
        logger: 日志记录器
        
    示例：
        >>> pruner = StructuredPruner(model, PruningConfig())
        >>> pruned_model = pruner.prune()
        >>> print(pruner.stats)
    """
    
    def __init__(
        self,
        model: MarianMTModel,
        config: Optional[PruningConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化剪枝器
        
        参数：
            model: 待剪枝的模型
            config: 剪枝配置
            logger: 日志记录器
        """
        self.model = model
        self.config = config or PruningConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 统计信息
        self.stats = PruningStats()
        
        # 记录原始参数
        self.stats.original_params = sum(p.numel() for p in model.parameters())
        self.stats.original_size_mb = self.stats.original_params * 4 / (1024 ** 2)
        
        self.logger.info("剪枝器初始化完成")
        self.logger.info(f"  原始参数: {self.stats.original_params:,}")
        self.logger.info(f"  FFN 剪枝比例: {self.config.ffn_prune_ratio:.0%}")
        self.logger.info(f"  注意力头剪枝比例: {self.config.head_prune_ratio:.0%}")
    
    def compute_importance_scores(
        self,
        weight: torch.Tensor,
        metric: str = "l1"
    ) -> torch.Tensor:
        """
        计算权重的重要性分数
        
        参数：
            weight: 权重张量
            metric: 评估方法（l1, l2, magnitude）
            
        返回：
            torch.Tensor: 重要性分数
        """
        if metric == "l1":
            # L1 范数：沿输出维度求和
            scores = weight.abs().sum(dim=0)
        elif metric == "l2":
            # L2 范数：沿输出维度求平方和再开根
            scores = (weight ** 2).sum(dim=0).sqrt()
        elif metric == "magnitude":
            # 幅度：取最大绝对值
            scores = weight.abs().max(dim=0)[0]
        else:
            raise ValueError(f"不支持的评估方法: {metric}")
        
        return scores
    
    def get_pruning_mask(
        self,
        scores: torch.Tensor,
        prune_ratio: float
    ) -> torch.Tensor:
        """
        根据重要性分数生成剪枝掩码
        
        参数：
            scores: 重要性分数
            prune_ratio: 剪枝比例
            
        返回：
            torch.Tensor: 剪枝掩码（1 保留，0 剪枝）
        """
        # 计算保留数量
        num_elements = scores.numel()
        num_keep = int(num_elements * (1 - prune_ratio))
        num_keep = max(num_keep, 1)  # 至少保留一个
        
        # 获取 top-k 索引
        _, indices = torch.topk(scores, num_keep)
        
        # 创建掩码
        mask = torch.zeros_like(scores)
        mask[indices] = 1
        
        return mask.bool()
    
    def prune_ffn_layer(
        self,
        ffn_layer: nn.Module,
        layer_name: str
    ) -> int:
        """
        剪枝单个 FFN 层
        
        FFN 结构：Linear(d_model, ffn_dim) -> ReLU -> Linear(ffn_dim, d_model)
        剪枝目标：减少 ffn_dim
        
        参数：
            ffn_layer: FFN 层
            layer_name: 层名称
            
        返回：
            int: 剪枝的神经元数量
        """
        # 获取 fc1 和 fc2 权重
        # MarianMT 中 FFN 结构为 fc1 -> activation -> fc2
        fc1_weight = ffn_layer.fc1.weight.data  # (ffn_dim, d_model)
        fc2_weight = ffn_layer.fc2.weight.data  # (d_model, ffn_dim)
        
        # 计算 fc1 输出神经元的重要性
        importance = self.compute_importance_scores(
            fc1_weight,
            self.config.importance_metric
        )
        
        # 生成剪枝掩码
        mask = self.get_pruning_mask(importance, self.config.ffn_prune_ratio)
        keep_indices = torch.where(mask)[0]
        num_pruned = (~mask).sum().item()
        
        # 剪枝 fc1：保留重要的输出神经元
        new_fc1_weight = fc1_weight[keep_indices]
        new_fc1_bias = ffn_layer.fc1.bias.data[keep_indices] if ffn_layer.fc1.bias is not None else None
        
        # 剪枝 fc2：保留对应的输入神经元
        new_fc2_weight = fc2_weight[:, keep_indices]
        
        # 更新层
        new_ffn_dim = len(keep_indices)
        ffn_layer.fc1 = nn.Linear(
            fc1_weight.shape[1],
            new_ffn_dim,
            bias=ffn_layer.fc1.bias is not None
        )
        ffn_layer.fc1.weight.data = new_fc1_weight
        if new_fc1_bias is not None:
            ffn_layer.fc1.bias.data = new_fc1_bias
        
        ffn_layer.fc2 = nn.Linear(
            new_ffn_dim,
            fc2_weight.shape[0],
            bias=ffn_layer.fc2.bias is not None
        )
        ffn_layer.fc2.weight.data = new_fc2_weight
        if ffn_layer.fc2.bias is not None:
            ffn_layer.fc2.bias.data = ffn_layer.fc2.bias.data.clone()
        
        # 记录统计
        self.stats.pruned_ffn_neurons[layer_name] = num_pruned
        
        return num_pruned
    
    def compute_head_importance(
        self,
        attention_layer: nn.Module
    ) -> torch.Tensor:
        """
        计算注意力头的重要性
        
        参数：
            attention_layer: 注意力层
            
        返回：
            torch.Tensor: 各头的重要性分数
        """
        # 获取输出投影权重
        out_proj_weight = attention_layer.out_proj.weight.data  # (d_model, d_model)
        
        # 获取头数和头维度
        num_heads = attention_layer.num_heads
        head_dim = attention_layer.head_dim
        
        # 重塑权重为 (d_model, num_heads, head_dim)
        reshaped_weight = out_proj_weight.view(
            out_proj_weight.shape[0], num_heads, head_dim
        )
        
        # 计算每个头的 L2 范数
        head_importance = reshaped_weight.norm(dim=(0, 2))  # (num_heads,)
        
        return head_importance
    
    def prune_attention_heads(
        self,
        attention_layer: nn.Module,
        layer_name: str
    ) -> int:
        """
        剪枝注意力头
        
        参数：
            attention_layer: 注意力层
            layer_name: 层名称
            
        返回：
            int: 剪枝的头数
        """
        # 计算头重要性
        head_importance = self.compute_head_importance(attention_layer)
        
        # 生成剪枝掩码
        mask = self.get_pruning_mask(head_importance, self.config.head_prune_ratio)
        keep_indices = torch.where(mask)[0]
        num_pruned = (~mask).sum().item()
        
        # 记录要保留的头
        heads_to_prune = torch.where(~mask)[0].tolist()
        
        # 标记需要剪枝的头
        attention_layer.pruned_heads = set(heads_to_prune)
        attention_layer.num_heads = len(keep_indices)
        
        # 更新投影权重
        head_dim = attention_layer.head_dim
        d_model = attention_layer.embed_dim
        
        # 保留的头索引范围
        keep_ranges = []
        for idx in keep_indices:
            start = idx * head_dim
            end = (idx + 1) * head_dim
            keep_ranges.append((start.item(), end.item()))
        
        # 创建索引列表
        keep_indices_flat = []
        for start, end in keep_ranges:
            keep_indices_flat.extend(range(start, end))
        
        # 剪枝 Q, K, V 投影
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            proj = getattr(attention_layer, proj_name)
            new_weight = proj.weight.data[keep_indices_flat]
            new_bias = proj.bias.data[keep_indices_flat] if proj.bias is not None else None
            
            new_proj = nn.Linear(d_model, len(keep_indices_flat), bias=proj.bias is not None)
            new_proj.weight.data = new_weight
            if new_bias is not None:
                new_proj.bias.data = new_bias
            
            setattr(attention_layer, proj_name, new_proj)
        
        # 剪枝输出投影
        out_proj = attention_layer.out_proj
        new_out_weight = out_proj.weight.data[:, keep_indices_flat]
        new_out_proj = nn.Linear(len(keep_indices_flat), d_model, bias=out_proj.bias is not None)
        new_out_proj.weight.data = new_out_weight
        if out_proj.bias is not None:
            new_out_proj.bias.data = out_proj.bias.data.clone()
        attention_layer.out_proj = new_out_proj
        
        # 记录统计
        self.stats.pruned_heads[layer_name] = num_pruned
        
        return num_pruned
    
    def prune_model(self) -> MarianMTModel:
        """
        执行完整的模型剪枝
        
        返回：
            MarianMTModel: 剪枝后的模型
        """
        self.logger.info("开始模型剪枝...")
        
        model = self.model
        
        # 剪枝编码器
        if self.config.prune_encoder:
            self.logger.info("剪枝编码器...")
            for i, layer in enumerate(model.model.encoder.layers):
                layer_name = f"encoder_layer_{i}"
                
                # 剪枝 FFN
                if hasattr(layer, 'fc1') and hasattr(layer, 'fc2'):
                    self.prune_ffn_layer(layer, f"{layer_name}_ffn")
                
                # 剪枝注意力头（暂时跳过，MarianMT 结构复杂）
                # self.prune_attention_heads(layer.self_attn, f"{layer_name}_attn")
        
        # 剪枝解码器
        if self.config.prune_decoder:
            self.logger.info("剪枝解码器...")
            for i, layer in enumerate(model.model.decoder.layers):
                layer_name = f"decoder_layer_{i}"
                
                # 剪枝 FFN
                if hasattr(layer, 'fc1') and hasattr(layer, 'fc2'):
                    self.prune_ffn_layer(layer, f"{layer_name}_ffn")
        
        # 更新统计
        self.stats.pruned_params = sum(p.numel() for p in model.parameters())
        self.stats.pruned_size_mb = self.stats.pruned_params * 4 / (1024 ** 2)
        self.stats.compression_ratio = 1 - (self.stats.pruned_params / self.stats.original_params)
        
        self.logger.info(str(self.stats))
        
        return model
    
    def prune(self) -> MarianMTModel:
        """
        执行剪枝的主入口
        
        返回：
            MarianMTModel: 剪枝后的模型
        """
        return self.prune_model()


def prune_model(
    model: MarianMTModel,
    ffn_prune_ratio: float = DEFAULT_FFN_PRUNE_RATIO,
    head_prune_ratio: float = DEFAULT_HEAD_PRUNE_RATIO,
    **kwargs
) -> Tuple[MarianMTModel, PruningStats]:
    """
    便捷函数：对模型进行剪枝
    
    参数：
        model: 待剪枝的模型
        ffn_prune_ratio: FFN 剪枝比例
        head_prune_ratio: 注意力头剪枝比例
        
    返回：
        Tuple[MarianMTModel, PruningStats]: 剪枝后的模型和统计信息
    """
    config = PruningConfig(
        ffn_prune_ratio=ffn_prune_ratio,
        head_prune_ratio=head_prune_ratio,
        **kwargs
    )
    
    pruner = StructuredPruner(model, config)
    pruned_model = pruner.prune()
    
    return pruned_model, pruner.stats


def iterative_prune_and_finetune(
    model: MarianMTModel,
    train_dataloader,
    eval_dataloader,
    target_ratio: float = 0.7,
    num_iterations: int = 3,
    finetune_epochs: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[MarianMTModel, PruningStats]:
    """
    迭代剪枝 + 微调
    
    渐进式剪枝可以获得更好的精度保持。
    
    参数：
        model: 待剪枝的模型
        train_dataloader: 训练数据加载器
        eval_dataloader: 验证数据加载器
        target_ratio: 目标剪枝比例
        num_iterations: 迭代次数
        finetune_epochs: 每次微调的 epoch 数
        logger: 日志记录器
        
    返回：
        Tuple[MarianMTModel, PruningStats]: 剪枝后的模型和统计信息
    """
    logger = logger or logging.getLogger(__name__)
    
    # 计算每次迭代的剪枝比例
    per_iteration_ratio = 1 - (1 - target_ratio) ** (1 / num_iterations)
    
    current_model = model
    total_stats = None
    
    for iteration in range(num_iterations):
        logger.info(f"\n迭代 {iteration + 1}/{num_iterations}")
        logger.info(f"当前剪枝比例: {per_iteration_ratio:.2%}")
        
        # 剪枝
        pruner = StructuredPruner(
            current_model,
            PruningConfig(
                ffn_prune_ratio=per_iteration_ratio,
                head_prune_ratio=per_iteration_ratio * 0.5,  # 头剪枝更保守
            )
        )
        current_model = pruner.prune()
        
        # TODO: 微调恢复（需要配合训练器）
        # finetune(current_model, train_dataloader, eval_dataloader, finetune_epochs)
        
        if total_stats is None:
            total_stats = pruner.stats
        else:
            total_stats = pruner.stats
    
    return current_model, total_stats


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="模型结构化剪枝工具"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="输入模型路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出模型路径"
    )
    parser.add_argument(
        "--ffn-ratio",
        type=float,
        default=DEFAULT_FFN_PRUNE_RATIO,
        help=f"FFN 剪枝比例（默认: {DEFAULT_FFN_PRUNE_RATIO}）"
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=DEFAULT_HEAD_PRUNE_RATIO,
        help=f"注意力头剪枝比例（默认: {DEFAULT_HEAD_PRUNE_RATIO}）"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 加载模型和分词器
    model = MarianMTModel.from_pretrained(args.model)
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    
    # 剪枝
    pruned_model, stats = prune_model(
        model,
        ffn_prune_ratio=args.ffn_ratio,
        head_prune_ratio=args.head_ratio
    )
    
    # 更新模型配置中的 FFN 维度
    # 计算新的 FFN 维度（从第一层获取）
    if hasattr(pruned_model.model.encoder, 'layers') and len(pruned_model.model.encoder.layers) > 0:
        first_layer = pruned_model.model.encoder.layers[0]
        if hasattr(first_layer, 'fc1'):
            new_ffn_dim = first_layer.fc1.out_features
            pruned_model.config.encoder_ffn_dim = new_ffn_dim
            pruned_model.config.decoder_ffn_dim = new_ffn_dim
            print(f"更新配置: FFN 维度 -> {new_ffn_dim}")
    
    # 保存模型和分词器
    pruned_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n剪枝模型已保存至: {args.output}")
    print(stats)


if __name__ == "__main__":
    main()
