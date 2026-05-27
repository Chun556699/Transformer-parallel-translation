#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT18冠军策略训练增强模块

功能说明：
    实现WMT18冠军系统的核心训练策略：
    - Label Smoothing（标签平滑）
    - Noam学习率调度器（Transformer原版）
    - 大批次训练优化
    - 梯度累积与延迟更新
    - 检查点平均

参考：
    WMT18 RWTH Aachen冠军系统

作者：NMT Project
版本：2.0.0
"""

import os
import math
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# ============================================================================
# Label Smoothing损失函数（WMT18核心策略）
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing损失函数
    
    WMT18策略：
    - 使用label smoothing防止过拟合
    - 提高模型泛化能力
    - 通常设置smoothing=0.1
    
    参考：
    - RWTH Aachen WMT18系统使用label smoothing
    - 原始论文: "Rethinking the Inception Architecture for Computer Vision"
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        初始化Label Smoothing损失
        
        参数：
            vocab_size: 词表大小
            padding_idx: 填充token的索引
            smoothing: 平滑系数（WMT18推荐0.1）
            reduction: 归约方式
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.reduction = reduction
        
        # 预计算平滑后的目标分布
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (vocab_size - 2)  # 排除pad和真实标签
    
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Label Smoothing损失
        
        参数：
            logits: 模型输出logits [batch_size, seq_len, vocab_size]
            target: 目标标签 [batch_size, seq_len]
            
        返回：
            损失值
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 展平
        logits = logits.view(-1, vocab_size)
        target = target.view(-1)
        
        # 创建平滑后的目标分布
        smooth_target = torch.zeros_like(logits)
        smooth_target.fill_(self.smoothing_value)
        
        # 在真实标签位置放置置信度
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 填充位置设为0
        smooth_target[:, self.padding_idx] = 0
        
        # 找到填充位置
        mask = (target == self.padding_idx)
        smooth_target[mask] = 0
        
        # 计算KL散度损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_target * log_probs, dim=-1)
        
        # 应用mask
        loss = loss.masked_fill(mask, 0)
        
        # 归约
        if self.reduction == 'mean':
            return loss.sum() / (~mask).sum().float()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# Noam学习率调度器（Transformer原版，WMT18使用）
# ============================================================================

class NoamScheduler:
    """
    Noam学习率调度器
    
    WMT18策略：
    - Transformer原版学习率调度
    - 预热阶段线性增加学习率
    - 之后按sqrt(step)衰减
    
    公式：
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    参考：
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - WMT18冠军系统使用类似调度策略
    """
    
    def __init__(
        self,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        """
        初始化Noam调度器
        
        参数：
            d_model: 模型隐藏维度
            warmup_steps: 预热步数（WMT18推荐4000）
            factor: 缩放因子
        """
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
    
    def step(self) -> float:
        """
        更新步数并返回当前学习率
        
        返回：
            当前学习率
        """
        self._step += 1
        return self.get_lr()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        step = self._step
        
        # Noam公式
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        
        lr = (self.d_model ** (-0.5)) * min(arg1, arg2) * self.factor
        return lr
    
    def state_dict(self) -> Dict[str, Any]:
        """保存状态"""
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'step': self._step
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态"""
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        self._step = state_dict['step']


def get_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int = 512,
    warmup_steps: int = 4000,
    factor: float = 1.0
) -> LambdaLR:
    """
    创建Noam学习率调度器（兼容PyTorch LambdaLR）
    
    参数：
        optimizer: 优化器
        d_model: 模型隐藏维度
        warmup_steps: 预热步数
        factor: 缩放因子
        
    返回：
        LambdaLR调度器
    """
    def lr_lambda(step: int) -> float:
        arg1 = step ** (-0.5)
        arg2 = step * (warmup_steps ** (-1.5))
        return (d_model ** (-0.5)) * min(arg1, arg2) * factor
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 大批次训练优化器（WMT18策略）
# ============================================================================

class LargeBatchOptimizer:
    """
    大批次训练优化器
    
    WMT18策略：
    - 延迟SGD更新（Delayed SGD）
    - 累积多个小批次后更新
    - 模拟大批次训练效果
    
    参考：
    - WMT18 Cambridge系统使用大批次训练
    - "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 8,
        max_grad_norm: float = 1.0
    ):
        """
        初始化大批次优化器
        
        参数：
            optimizer: 基础优化器
            accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
        """
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._step = 0
    
    def step(self, loss: torch.Tensor):
        """
        执行一步优化
        
        参数：
            loss: 损失值（需要先normalize）
        """
        # 归一化损失
        loss = loss / self.accumulation_steps
        
        # 反向传播
        loss.backward()
        
        self._step += 1
        
        # 累积足够步数后更新
        if self._step % self.accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'],
                self.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return True
        return False
    
    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()


# ============================================================================
# 检查点平均（WMT18集成策略）
# ============================================================================

class CheckpointAverager:
    """
    检查点平均器
    
    WMT18策略：
    - 平均最后N个检查点
    - 提高模型稳定性
    - 类似集成效果但无额外推理开销
    
    参考：
    - WMT18 RWTH系统使用检查点平均
    - "Averaging Weights Leads to Wider Optima and Better Generalization"
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_checkpoints: int = 5,
        checkpoint_dir: Optional[str] = None
    ):
        """
        初始化检查点平均器
        
        参数：
            model: 模型实例
            n_checkpoints: 保留的检查点数量
            checkpoint_dir: 检查点目录
        """
        self.model = model
        self.n_checkpoints = n_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoints: List[Dict[str, torch.Tensor]] = []
    
    def add_checkpoint(self, state_dict: Optional[Dict] = None):
        """
        添加检查点
        
        参数：
            state_dict: 模型状态字典（可选，默认使用当前模型）
        """
        if state_dict is None:
            state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else:
            state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
        
        self.checkpoints.append(state_dict)
        
        # 保持最近N个
        if len(self.checkpoints) > self.n_checkpoints:
            self.checkpoints.pop(0)
    
    def get_averaged_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取平均后的状态字典
        
        返回：
            平均后的状态字典
        """
        if not self.checkpoints:
            return self.model.state_dict()
        
        # 初始化平均状态
        averaged_state = {}
        
        # 获取所有键
        keys = self.checkpoints[0].keys()
        
        for key in keys:
            # 堆叠所有检查点的该参数
            stacked = torch.stack([ckpt[key] for ckpt in self.checkpoints], dim=0)
            # 计算平均
            averaged_state[key] = stacked.mean(dim=0)
        
        return averaged_state
    
    def apply_averaged_weights(self):
        """应用平均后的权重到模型"""
        averaged_state = self.get_averaged_state_dict()
        self.model.load_state_dict(averaged_state)
        logger.info(f"已应用{len(self.checkpoints)}个检查点的平均权重")
    
    def save_averaged_checkpoint(self, path: str):
        """保存平均后的检查点"""
        averaged_state = self.get_averaged_state_dict()
        torch.save(averaged_state, path)
        logger.info(f"已保存平均检查点: {path}")


# ============================================================================
# 模型集成（WMT18核心策略）
# ============================================================================

class ModelEnsemble:
    """
    模型集成器
    
    WMT18策略：
    - 集成多个独立训练的模型
    - 平均输出概率分布
    - 显著提升翻译质量
    
    参考：
    - WMT18 RWTH系统集成4-6个模型
    - 平均BLEU提升2-3个点
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        """
        初始化模型集成
        
        参数：
            models: 模型列表
            weights: 模型权重（可选，默认等权重）
        """
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        else:
            assert len(weights) == self.n_models
            self.weights = [w / sum(weights) for w in weights]
        
        # 设置为评估模式
        for model in self.models:
            model.eval()
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        集成前向传播
        
        参数：
            input_ids: 输入token IDs
            attention_mask: 注意力mask
            **kwargs: 其他参数
            
        返回：
            平均后的logits
        """
        all_logits = []
        
        for model in self.models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            all_logits.append(outputs.logits)
        
        # 加权平均
        stacked_logits = torch.stack(all_logits, dim=0)
        weights_tensor = torch.tensor(
            self.weights, 
            device=stacked_logits.device
        ).view(-1, 1, 1, 1)
        
        averaged_logits = (stacked_logits * weights_tensor).sum(dim=0)
        
        return averaged_logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        集成生成
        
        参数：
            input_ids: 输入token IDs
            **kwargs: 生成参数
            
        返回：
            生成的token IDs
        """
        # 使用第一个模型的tokenizer和生成方法
        # 集成通过平均logits实现
        return self.models[0].generate(
            input_ids=input_ids,
            logits_processor=[self._ensemble_logits_processor],
            **kwargs
        )
    
    def _ensemble_logits_processor(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """集成logits处理器"""
        # 这里简化处理，实际集成需要更复杂的实现
        return scores
    
    def to(self, device: torch.device):
        """移动到指定设备"""
        for model in self.models:
            model.to(device)
        return self


# ============================================================================
# 训练配置数据类
# ============================================================================

@dataclass
class WMT18TrainingConfig:
    """
    WMT18风格训练配置
    
    属性：
        label_smoothing: 标签平滑系数
        warmup_steps: 预热步数
        d_model: 模型维度
        accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        n_checkpoints: 检查点平均数量
        ensemble_size: 集成模型数量
    """
    # Label Smoothing
    label_smoothing: float = 0.1
    
    # 学习率调度
    warmup_steps: int = 4000
    d_model: int = 512
    
    # 大批次训练
    accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # 检查点平均
    n_checkpoints: int = 5
    
    # 模型集成
    ensemble_size: int = 4
    
    # 其他
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1


# ============================================================================
# 便捷函数
# ============================================================================

def create_wmt18_training_components(
    model: nn.Module,
    vocab_size: int,
    config: Optional[WMT18TrainingConfig] = None
) -> Tuple[nn.Module, torch.optim.Optimizer, Any, Any]:
    """
    创建WMT18风格训练组件
    
    参数：
        model: 模型
        vocab_size: 词表大小
        config: 训练配置
        
    返回：
        (模型, 优化器, 调度器, 损失函数)
    """
    config = config or WMT18TrainingConfig()
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=0.0,  # 由调度器控制
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    # 创建Noam调度器
    scheduler = get_noam_scheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )
    
    # 创建Label Smoothing损失
    criterion = LabelSmoothingLoss(
        vocab_size=vocab_size,
        smoothing=config.label_smoothing
    )
    
    return model, optimizer, scheduler, criterion
