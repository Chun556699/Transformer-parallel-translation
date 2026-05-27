#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合BPE预处理模块

功能说明：
    实现WMT18冠军系统的核心数据预处理策略：
    - 联合BPE训练（源语言+目标语言共享词表）
    - 智能分词与归一化
    - 子词正则化增强

参考：
    WMT18 RWTH Aachen冠军系统使用50k合并操作的联合BPE

作者：NMT Project
版本：2.0.0
"""

import os
import re
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class BPEConfig:
    """
    BPE配置
    
    属性：
        vocab_size: 词表大小（WMT18推荐50000）
        min_frequency: 最小词频
        special_tokens: 特殊标记
        lowercase: 是否小写
        split_digits: 是否拆分数字
        split_punctuation: 是否拆分标点
    """
    vocab_size: int = 50000  # WMT18冠军配置
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<s>", "</s>", "<unk>", "<mask>"
    ])
    lowercase: bool = False
    split_digits: bool = True
    split_punctuation: bool = True
    joint_bpe: bool = True  # 联合BPE（WMT18策略）


@dataclass
class BPEStats:
    """BPE统计信息"""
    vocab_size: int = 0
    merge_ops: int = 0
    source_tokens: int = 0
    target_tokens: int = 0
    source_unique: int = 0
    target_unique: int = 0
    oov_rate: float = 0.0


# ============================================================================
# 联合BPE处理器
# ============================================================================

class JointBPEProcessor:
    """
    联合BPE处理器
    
    WMT18冠军策略：
    - 联合训练源语言和目标语言的BPE
    - 共享词表减少参数量
    - 50k合并操作平衡性能和效率
    """
    
    def __init__(self, config: Optional[BPEConfig] = None):
        """
        初始化BPE处理器
        
        参数：
            config: BPE配置
        """
        self.config = config or BPEConfig()
        self.bpe = None
        self.vocab = {}
        self.stats = BPEStats()
        
        # 尝试导入sentencepiece或tokenizers
        self._init_tokenizer_backend()
    
    def _init_tokenizer_backend(self):
        """初始化分词器后端"""
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import (
                Whitespace, Digits, Punctuation, Metaspace
            )
            from tokenizers.normalizers import NFKC
            self.backend = "tokenizers"
            self._tokenizer_cls = Tokenizer
            self._bpe_cls = BPE
            self._trainer_cls = BpeTrainer
            logger.info("使用tokenizers后端")
        except ImportError:
            self.backend = "sentencepiece"
            logger.info("使用sentencepiece后端")
    
    def train_joint_bpe(
        self,
        source_texts: List[str],
        target_texts: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练联合BPE（WMT18核心策略）
        
        参数：
            source_texts: 源语言文本列表
            target_texts: 目标语言文本列表
            output_path: 输出路径
            
        返回：
            训练结果字典
        """
        logger.info(f"开始训练联合BPE，词表大小: {self.config.vocab_size}")
        
        if self.backend == "tokenizers":
            return self._train_with_tokenizers(source_texts, target_texts, output_path)
        else:
            return self._train_with_sentencepiece(source_texts, target_texts, output_path)
    
    def _train_with_tokenizers(
        self,
        source_texts: List[str],
        target_texts: List[str],
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """使用tokenizers库训练"""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import (
            Whitespace, Digits, Punctuation, Metaspace
        )
        from tokenizers.normalizers import NFKC, Lowercase, Sequence
        
        # 创建分词器
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # 设置归一化
        normalizers = [NFKC()]
        if self.config.lowercase:
            normalizers.append(Lowercase())
        tokenizer.normalizer = Sequence(normalizers)
        
        # 设置预分词
        tokenizer.pre_tokenizer = Whitespace()
        
        # 创建训练器
        trainer = BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True
        )
        
        # 联合训练（WMT18策略：混合源语言和目标语言）
        all_texts = source_texts + target_texts
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                         encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            # 训练
            tokenizer.train([temp_file], trainer)
            
            # 保存
            if output_path:
                tokenizer.save(output_path)
                logger.info(f"BPE模型已保存: {output_path}")
            
            # 更新统计
            self.stats.vocab_size = tokenizer.get_vocab_size()
            self.stats.merge_ops = self.config.vocab_size
            
            return {
                'vocab_size': self.stats.vocab_size,
                'merge_ops': self.stats.merge_ops,
                'output_path': output_path
            }
        finally:
            os.unlink(temp_file)
    
    def _train_with_sentencepiece(
        self,
        source_texts: List[str],
        target_texts: List[str],
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """使用sentencepiece训练"""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("请安装sentencepiece: pip install sentencepiece")
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                         encoding='utf-8') as f:
            for text in source_texts + target_texts:
                f.write(text + '\n')
            temp_file = f.name
        
        try:
            # 训练参数
            model_prefix = output_path.replace('.model', '') if output_path else 'joint_bpe'
            
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=self.config.vocab_size,
                character_coverage=0.9995,
                model_type='bpe',
                unk_id=3,
                bos_id=1,
                eos_id=2,
                pad_id=0,
                user_defined_symbols=self.config.special_tokens
            )
            
            # 加载模型
            self.bpe = spm.SentencePieceProcessor()
            self.bpe.load(f"{model_prefix}.model")
            
            # 更新统计
            self.stats.vocab_size = self.bpe.get_piece_size()
            self.stats.merge_ops = self.config.vocab_size
            
            return {
                'vocab_size': self.stats.vocab_size,
                'merge_ops': self.stats.merge_ops,
                'output_path': f"{model_prefix}.model"
            }
        finally:
            os.unlink(temp_file)
    
    def encode(
        self,
        text: str,
        lang: str = "source"
    ) -> List[int]:
        """
        编码文本为token ID序列
        
        参数：
            text: 输入文本
            lang: 语言类型（source/target）
            
        返回：
            token ID列表
        """
        if self.bpe is None:
            raise ValueError("BPE模型未训练或加载")
        
        return self.bpe.encode(text, out_type=int)
    
    def decode(
        self,
        ids: List[int],
        lang: str = "source"
    ) -> str:
        """
        解码token ID序列为文本
        
        参数：
            ids: token ID列表
            lang: 语言类型
            
        返回：
            解码后的文本
        """
        if self.bpe is None:
            raise ValueError("BPE模型未训练或加载")
        
        return self.bpe.decode(ids)
    
    def save(self, path: str):
        """保存BPE模型"""
        if self.bpe is not None:
            # sentencepiece模型在训练时已保存
            pass
    
    def load(self, path: str):
        """加载BPE模型"""
        try:
            import sentencepiece as spm
            self.bpe = spm.SentencePieceProcessor()
            self.bpe.load(path)
            self.stats.vocab_size = self.bpe.get_piece_size()
            logger.info(f"已加载BPE模型: {path}, 词表大小: {self.stats.vocab_size}")
        except ImportError:
            # 尝试加载tokenizers格式
            from tokenizers import Tokenizer
            self.bpe = Tokenizer.from_file(path)
            self.stats.vocab_size = self.bpe.get_vocab_size()
            logger.info(f"已加载BPE模型: {path}, 词表大小: {self.stats.vocab_size}")


# ============================================================================
# 数据预处理管道
# ============================================================================

class WMT18Preprocessor:
    """
    WMT18风格数据预处理器
    
    实现冠军系统的完整预处理流程：
    1. 文本归一化
    2. 分词
    3. 联合BPE编码
    4. 长度过滤
    """
    
    def __init__(
        self,
        bpe_processor: Optional[JointBPEProcessor] = None,
        max_length: int = 128,
        min_length: int = 3
    ):
        """
        初始化预处理器
        
        参数：
            bpe_processor: BPE处理器
            max_length: 最大序列长度
            min_length: 最小序列长度
        """
        self.bpe = bpe_processor
        self.max_length = max_length
        self.min_length = min_length
    
    def normalize_text(self, text: str, lang: str = "zh") -> str:
        """
        文本归一化（WMT18策略）
        
        参数：
            text: 输入文本
            lang: 语言代码
            
        返回：
            归一化后的文本
        """
        import unicodedata
        
        # Unicode归一化
        text = unicodedata.normalize('NFKC', text)
        
        # 移除控制字符
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
        
        # 标准化空白
        text = ' '.join(text.split())
        
        # 语言特定处理
        if lang == "zh":
            # 中文：移除多余空格
            text = re.sub(r'\s+', '', text)
        else:
            # 英文：标准化空格
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_pair(
        self,
        source: str,
        target: str,
        source_lang: str = "zh",
        target_lang: str = "en"
    ) -> Optional[Tuple[str, str, List[int], List[int]]]:
        """
        预处理句对
        
        参数：
            source: 源语言文本
            target: 目标语言文本
            source_lang: 源语言代码
            target_lang: 目标语言代码
            
        返回：
            (归一化源文, 归一化译文, 源文tokens, 译文tokens) 或 None
        """
        # 归一化
        source_norm = self.normalize_text(source, source_lang)
        target_norm = self.normalize_text(target, target_lang)
        
        # 长度检查
        if len(source_norm) < self.min_length or len(target_norm) < self.min_length:
            return None
        
        if len(source_norm) > self.max_length or len(target_norm) > self.max_length:
            return None
        
        # BPE编码
        if self.bpe is not None:
            source_ids = self.bpe.encode(source_norm, "source")
            target_ids = self.bpe.encode(target_norm, "target")
            
            # 再次检查长度
            if len(source_ids) > self.max_length or len(target_ids) > self.max_length:
                return None
            
            return (source_norm, target_norm, source_ids, target_ids)
        
        return (source_norm, target_norm, [], [])
    
    def preprocess_batch(
        self,
        sources: List[str],
        targets: List[str],
        source_lang: str = "zh",
        target_lang: str = "en"
    ) -> Tuple[List[str], List[str], List[List[int]], List[List[int]]]:
        """
        批量预处理
        
        参数：
            sources: 源语言文本列表
            targets: 目标语言文本列表
            source_lang: 源语言代码
            target_lang: 目标语言代码
            
        返回：
            (源文列表, 译文列表, 源文tokens列表, 译文tokens列表)
        """
        results = []
        
        for src, tgt in zip(sources, targets):
            result = self.preprocess_pair(src, tgt, source_lang, target_lang)
            if result:
                results.append(result)
        
        if not results:
            return [], [], [], []
        
        # 解包结果
        src_texts, tgt_texts, src_ids, tgt_ids = zip(*results)
        return list(src_texts), list(tgt_texts), list(src_ids), list(tgt_ids)


# ============================================================================
# 便捷函数
# ============================================================================

def train_joint_bpe(
    source_texts: List[str],
    target_texts: List[str],
    vocab_size: int = 50000,
    output_path: Optional[str] = None
) -> JointBPEProcessor:
    """
    训练联合BPE的便捷函数
    
    参数：
        source_texts: 源语言文本列表
        target_texts: 目标语言文本列表
        vocab_size: 词表大小
        output_path: 输出路径
        
    返回：
        训练好的BPE处理器
    """
    config = BPEConfig(vocab_size=vocab_size)
    processor = JointBPEProcessor(config)
    processor.train_joint_bpe(source_texts, target_texts, output_path)
    return processor


def create_wmt18_preprocessor(
    bpe_model_path: Optional[str] = None,
    max_length: int = 128
) -> WMT18Preprocessor:
    """
    创建WMT18风格预处理器
    
    参数：
        bpe_model_path: BPE模型路径
        max_length: 最大序列长度
        
    返回：
        预处理器实例
    """
    bpe = None
    if bpe_model_path:
        bpe = JointBPEProcessor()
        bpe.load(bpe_model_path)
    
    return WMT18Preprocessor(bpe, max_length=max_length)
