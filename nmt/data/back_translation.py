#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反向翻译数据增强模块

功能说明：
    实现WMT18冠军系统的反向翻译策略：
    - 使用目标语言单语数据
    - 反向翻译生成伪平行数据
    - 数据过滤与质量筛选
    - 混合真实与合成数据

参考：
    WMT18 RWTH Aachen和Cambridge系统都使用反向翻译

作者：NMT Project
版本：2.0.0
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class BackTranslationConfig:
    """
    反向翻译配置
    
    属性：
        synthetic_ratio: 合成数据比例（相对于真实数据）
        min_confidence: 最小置信度阈值
        filter_by_length: 是否按长度过滤
        max_length_ratio: 最大长度比
        temperature: 采样温度
        beam_size: 束搜索大小
    """
    synthetic_ratio: float = 0.5  # 合成数据占真实数据的比例
    min_confidence: float = 0.7  # 最小翻译置信度
    filter_by_length: bool = True
    max_length_ratio: float = 3.0
    temperature: float = 1.0
    beam_size: int = 5
    batch_size: int = 32
    max_samples: int = 1000000  # 最大合成样本数


@dataclass
class SyntheticSample:
    """合成样本"""
    source: str  # 反向翻译生成的源语言
    target: str  # 原始目标语言
    confidence: float  # 翻译置信度
    model_score: float  # 模型打分


# ============================================================================
# 反向翻译器
# ============================================================================

class BackTranslator:
    """
    反向翻译器
    
    WMT18策略：
    - 使用反向模型翻译单语数据
    - 生成伪平行语料
    - 过滤低质量翻译
    - 与真实数据混合训练
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        config: Optional[BackTranslationConfig] = None
    ):
        """
        初始化反向翻译器
        
        参数：
            model_path: 反向模型路径（如en->zh模型用于生成zh->en数据）
            device: 设备
            config: 配置
        """
        self.model_path = model_path
        self.device = device
        self.config = config or BackTranslationConfig()
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载反向模型"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            logger.info(f"加载反向翻译模型: {self.model_path}")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.model = MarianMTModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("反向翻译模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    @torch.no_grad()
    def translate_batch(
        self,
        texts: List[str],
        return_scores: bool = True
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        批量翻译
        
        参数：
            texts: 输入文本列表
            return_scores: 是否返回置信度分数
            
        返回：
            翻译结果（和分数）
        """
        # 编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成
        outputs = self.model.generate(
            **inputs,
            num_beams=self.config.beam_size,
            temperature=self.config.temperature,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=512
        )
        
        # 解码
        translations = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        if return_scores:
            # 计算置信度分数
            scores = self._compute_confidence(outputs)
            return translations, scores
        
        return translations
    
    def _compute_confidence(self, outputs) -> List[float]:
        """计算翻译置信度"""
        # 使用序列分数的指数作为置信度
        if hasattr(outputs, 'sequences_scores'):
            scores = outputs.sequences_scores.exp().tolist()
        else:
            scores = [1.0] * len(outputs.sequences)
        return scores
    
    def generate_synthetic_data(
        self,
        monolingual_texts: List[str],
        show_progress: bool = True
    ) -> List[SyntheticSample]:
        """
        生成合成数据
        
        参数：
            monolingual_texts: 单语文本列表（目标语言）
            show_progress: 是否显示进度条
            
        返回：
            合成样本列表
        """
        synthetic_samples = []
        batch_size = self.config.batch_size
        
        # 限制样本数
        n_samples = min(len(monolingual_texts), self.config.max_samples)
        monolingual_texts = monolingual_texts[:n_samples]
        
        # 批量处理
        iterator = range(0, len(monolingual_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="反向翻译生成")
        
        for i in iterator:
            batch = monolingual_texts[i:i + batch_size]
            
            try:
                translations, scores = self.translate_batch(batch)
                
                for target, source, score in zip(batch, translations, scores):
                    # 过滤低置信度
                    if score < self.config.min_confidence:
                        continue
                    
                    # 过滤长度异常
                    if self.config.filter_by_length:
                        len_ratio = len(source) / max(len(target), 1)
                        if len_ratio > self.config.max_length_ratio or len_ratio < 1 / self.config.max_length_ratio:
                            continue
                    
                    synthetic_samples.append(SyntheticSample(
                        source=source,
                        target=target,
                        confidence=score,
                        model_score=score
                    ))
            except Exception as e:
                logger.warning(f"批次处理失败: {e}")
                continue
        
        logger.info(f"生成合成样本: {len(synthetic_samples)}/{n_samples}")
        return synthetic_samples


# ============================================================================
# 数据混合器
# ============================================================================

class DataMixer:
    """
    真实与合成数据混合器
    
    WMT18策略：
    - 按比例混合真实和合成数据
    - 保持数据分布平衡
    - 支持动态采样
    """
    
    def __init__(
        self,
        real_data: List[Tuple[str, str]],
        synthetic_data: List[SyntheticSample],
        synthetic_ratio: float = 0.5
    ):
        """
        初始化数据混合器
        
        参数：
            real_data: 真实平行数据 [(源, 目标)]
            synthetic_data: 合成数据
            synthetic_ratio: 合成数据比例
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.synthetic_ratio = synthetic_ratio
        
        # 计算需要的合成数据量
        n_real = len(real_data)
        n_synthetic_needed = int(n_real * synthetic_ratio)
        
        # 采样合成数据
        if len(synthetic_data) > n_synthetic_needed:
            self.sampled_synthetic = random.sample(
                synthetic_data, n_synthetic_needed
            )
        else:
            self.sampled_synthetic = synthetic_data
    
    def get_mixed_data(self) -> List[Tuple[str, str, bool]]:
        """
        获取混合后的数据
        
        返回：
            [(源, 目标, 是否合成)]
        """
        mixed = []
        
        # 添加真实数据
        for src, tgt in self.real_data:
            mixed.append((src, tgt, False))
        
        # 添加合成数据
        for sample in self.sampled_synthetic:
            mixed.append((sample.source, sample.target, True))
        
        # 打乱
        random.shuffle(mixed)
        
        return mixed
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'real_samples': len(self.real_data),
            'synthetic_samples': len(self.sampled_synthetic),
            'total_samples': len(self.real_data) + len(self.sampled_synthetic),
            'synthetic_ratio': len(self.sampled_synthetic) / max(len(self.real_data), 1)
        }


# ============================================================================
# 迭代式反向翻译
# ============================================================================

class IterativeBackTranslation:
    """
    迭代式反向翻译
    
    WMT18高级策略：
    - 多轮反向翻译
    - 每轮使用更好的模型
    - 逐步提升数据质量
    """
    
    def __init__(
        self,
        forward_model_path: str,
        backward_model_path: str,
        device: str = "cuda",
        config: Optional[BackTranslationConfig] = None
    ):
        """
        初始化迭代式反向翻译
        
        参数：
            forward_model_path: 正向模型路径
            backward_model_path: 反向模型路径
            device: 设备
            config: 配置
        """
        self.forward_model_path = forward_model_path
        self.backward_model_path = backward_model_path
        self.device = device
        self.config = config or BackTranslationConfig()
        
        self.forward_translator = None
        self.backward_translator = None
    
    def _load_forward(self):
        """加载正向模型"""
        if self.forward_translator is None:
            self.forward_translator = BackTranslator(
                self.forward_model_path,
                self.device,
                self.config
            )
    
    def _load_backward(self):
        """加载反向模型"""
        if self.backward_translator is None:
            self.backward_translator = BackTranslator(
                self.backward_model_path,
                self.device,
                self.config
            )
    
    def iterative_augment(
        self,
        parallel_data: List[Tuple[str, str]],
        monolingual_source: List[str],
        monolingual_target: List[str],
        n_iterations: int = 2
    ) -> List[Tuple[str, str]]:
        """
        迭代式数据增强
        
        参数：
            parallel_data: 平行数据
            monolingual_source: 源语言单语数据
            monolingual_target: 目标语言单语数据
            n_iterations: 迭代次数
            
        返回：
            增强后的平行数据
        """
        augmented_data = list(parallel_data)
        
        for iteration in range(n_iterations):
            logger.info(f"迭代 {iteration + 1}/{n_iterations}")
            
            # 反向翻译目标语言单语数据
            self._load_backward()
            synthetic_backward = self.backward_translator.generate_synthetic_data(
                monolingual_target[:self.config.max_samples // n_iterations]
            )
            
            # 添加合成数据
            for sample in synthetic_backward:
                augmented_data.append((sample.source, sample.target))
            
            logger.info(f"迭代后数据量: {len(augmented_data)}")
        
        return augmented_data


# ============================================================================
# 便捷函数
# ============================================================================

def create_back_translator(
    model_path: str,
    device: str = "cuda",
    synthetic_ratio: float = 0.5
) -> BackTranslator:
    """
    创建反向翻译器
    
    参数：
        model_path: 模型路径
        device: 设备
        synthetic_ratio: 合成数据比例
        
    返回：
        反向翻译器实例
    """
    config = BackTranslationConfig(synthetic_ratio=synthetic_ratio)
    return BackTranslator(model_path, device, config)


def augment_with_back_translation(
    parallel_data: List[Tuple[str, str]],
    monolingual_data: List[str],
    back_translator: BackTranslator
) -> List[Tuple[str, str]]:
    """
    使用反向翻译增强数据
    
    参数：
        parallel_data: 平行数据
        monolingual_data: 单语数据
        back_translator: 反向翻译器
        
    返回：
        增强后的平行数据
    """
    # 生成合成数据
    synthetic = back_translator.generate_synthetic_data(monolingual_data)
    
    # 混合数据
    mixer = DataMixer(
        parallel_data,
        synthetic,
        back_translator.config.synthetic_ratio
    )
    
    mixed = mixer.get_mixed_data()
    
    # 返回 (源, 目标) 格式
    return [(src, tgt) for src, tgt, _ in mixed]
