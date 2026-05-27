"""
分词器模块

功能说明：
    封装 Helsinki-NLP MarianMT 模型的分词器，提供：
    - 中英双向分词支持
    - BPE 子词切分
    - 动态长度截断与填充
    - 自定义词表扩展接口

依赖：
    - transformers: MarianTokenizer
    - sentencepiece: SentencePiece 分词

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import torch
from transformers import MarianTokenizer, PreTrainedTokenizer

# ====================================
# 常量定义
# ====================================

# 默认最大序列长度
DEFAULT_MAX_LENGTH = 512

# 默认模型路径
DEFAULT_ZH_EN_MODEL = "Helsinki-NLP/opus-mt-zh-en"
DEFAULT_EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"

# 特殊 token
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@dataclass
class TokenizerConfig:
    """
    分词器配置数据类
    
    属性：
        model_path: 模型路径
        max_length: 最大序列长度
        padding: 填充策略
        truncation: 是否截断
        return_tensors: 返回张量类型
    """
    model_path: str
    max_length: int = DEFAULT_MAX_LENGTH
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"


class TranslationTokenizer:
    """
    翻译模型分词器
    
    功能说明：
        封装 MarianTokenizer，支持中英双向翻译的分词需求：
        - 自动加载 Helsinki-NLP 预训练分词器
        - 支持批量分词与反分词
        - 支持动态填充和截断
        - 提供词表统计和特殊 token 管理
    
    参数：
        model_name_or_path: 模型名称或路径
        max_length: 最大序列长度
        padding: 填充策略 ('max_length', 'longest', False)
        truncation: 是否截断超长序列
        direction: 翻译方向 ('zh2en' 或 'en2zh')
        logger: 日志记录器
        
    示例：
        >>> tokenizer = TranslationTokenizer(
        ...     model_name_or_path="models/Helsinki-NLP-opus-mt-zh-en",
        ...     direction="zh2en"
        ... )
        >>> encoded = tokenizer.encode("你好世界")
        >>> print(tokenizer.decode(encoded["input_ids"][0]))
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        padding: Union[str, bool] = "max_length",
        truncation: bool = True,
        direction: str = "zh2en",
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化分词器
        
        参数：
            model_name_or_path: 模型名称或本地路径
            max_length: 最大序列长度
            padding: 填充策略
            truncation: 是否截断
            direction: 翻译方向
            logger: 日志记录器
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.direction = direction
        self.logger = logger or logging.getLogger(__name__)
        
        # 加载分词器
        self.tokenizer = self._load_tokenizer()
        
        # 记录词表信息
        self.logger.info(f"分词器加载完成: {model_name_or_path}")
        self.logger.info(f"  词表大小: {self.vocab_size}")
        self.logger.info(f"  最大长度: {max_length}")
        self.logger.info(f"  翻译方向: {direction}")
    
    def _load_tokenizer(self) -> MarianTokenizer:
        """
        加载 MarianTokenizer
        
        MarianTokenizer 使用 SentencePiece 进行 BPE 子词切分，
        支持多语言翻译任务。
        
        返回：
            MarianTokenizer: 加载的分词器
        """
        try:
            tokenizer = MarianTokenizer.from_pretrained(
                self.model_name_or_path,
                model_max_length=self.max_length
            )
            return tokenizer
        except Exception as e:
            self.logger.error(f"分词器加载失败: {e}")
            raise
    
    @property
    def vocab_size(self) -> int:
        """获取词表大小"""
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """获取 PAD token ID"""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """获取 EOS token ID"""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """获取 BOS token ID"""
        return self.tokenizer.bos_token_id
    
    @property
    def unk_token_id(self) -> int:
        """获取 UNK token ID"""
        return self.tokenizer.unk_token_id
    
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Optional[Union[str, bool]] = None,
        truncation: Optional[bool] = None,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        编码文本为 token IDs
        
        参数：
            text: 输入文本（字符串或列表）
            max_length: 最大长度（覆盖默认值）
            padding: 填充策略（覆盖默认值）
            truncation: 是否截断（覆盖默认值）
            return_tensors: 返回张量类型
            add_special_tokens: 是否添加特殊 token
            
        返回：
            Dict[str, torch.Tensor]: 包含 input_ids, attention_mask 的字典
        """
        # 使用默认值或覆盖值
        max_length = max_length or self.max_length
        padding = padding if padding is not None else self.padding
        truncation = truncation if truncation is not None else self.truncation
        
        # 调用分词器
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
        
        return encoded
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        解码 token IDs 为文本
        
        参数：
            token_ids: token ID 列表或张量
            skip_special_tokens: 是否跳过特殊 token
            clean_up_tokenization_spaces: 是否清理多余空格
            
        返回：
            str: 解码后的文本
        """
        # 转换张量为列表
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def batch_decode(
        self,
        token_ids_batch: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> List[str]:
        """
        批量解码 token IDs
        
        参数：
            token_ids_batch: token ID 批次
            skip_special_tokens: 是否跳过特殊 token
            clean_up_tokenization_spaces: 是否清理多余空格
            
        返回：
            List[str]: 解码后的文本列表
        """
        return self.tokenizer.batch_decode(
            token_ids_batch,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def tokenize(self, text: str) -> List[str]:
        """
        将文本切分为子词列表（不转换为 ID）
        
        参数：
            text: 输入文本
            
        返回：
            List[str]: 子词列表
        """
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        将子词列表转换为 ID 列表
        
        参数：
            tokens: 子词列表
            
        返回：
            List[int]: ID 列表
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        将 ID 列表转换为子词列表
        
        参数：
            ids: ID 列表
            
        返回：
            List[str]: 子词列表
        """
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def get_vocab(self) -> Dict[str, int]:
        """
        获取完整词表
        
        返回：
            Dict[str, int]: token 到 ID 的映射
        """
        return self.tokenizer.get_vocab()
    
    def save(self, save_directory: Union[str, Path]) -> None:
        """
        保存分词器到目录
        
        参数：
            save_directory: 保存目录
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        self.logger.info(f"分词器已保存至: {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> "TranslationTokenizer":
        """
        从预训练模型加载分词器
        
        参数：
            model_name_or_path: 模型名称或路径
            **kwargs: 其他参数
            
        返回：
            TranslationTokenizer: 分词器实例
        """
        return cls(model_name_or_path=model_name_or_path, **kwargs)
    
    def prepare_translation_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        准备翻译训练批次
        
        参数：
            src_texts: 源语言文本列表
            tgt_texts: 目标语言文本列表（训练时需要）
            max_length: 源语言最大长度
            max_target_length: 目标语言最大长度
            
        返回：
            Dict[str, torch.Tensor]: 包含 input_ids, attention_mask, labels 的字典
        """
        max_length = max_length or self.max_length
        max_target_length = max_target_length or self.max_length
        
        # 编码源语言
        model_inputs = self.encode(
            src_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        # 编码目标语言（如果提供）
        if tgt_texts is not None:
            # 使用 text_target 参数编码目标文本
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt_texts,
                    max_length=max_target_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            # 将 PAD token 替换为 -100（忽略损失计算）
            labels_ids = labels["input_ids"]
            labels_ids[labels_ids == self.pad_token_id] = -100
            model_inputs["labels"] = labels_ids
        
        return model_inputs
    
    def __len__(self) -> int:
        """返回词表大小"""
        return self.vocab_size
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"TranslationTokenizer("
            f"model='{self.model_name_or_path}', "
            f"vocab_size={self.vocab_size}, "
            f"max_length={self.max_length}, "
            f"direction='{self.direction}')"
        )


class BilingualTokenizer:
    """
    双语分词器
    
    功能说明：
        管理中英双向翻译的分词器对，提供统一接口：
        - zh2en: 中文→英文分词器
        - en2zh: 英文→中文分词器
    
    参数：
        zh_en_model: 中→英模型路径
        en_zh_model: 英→中模型路径
        max_length: 最大序列长度
        logger: 日志记录器
        
    示例：
        >>> tokenizer = BilingualTokenizer(
        ...     zh_en_model="models/Helsinki-NLP-opus-mt-zh-en",
        ...     en_zh_model="models/Helsinki-NLP-opus-mt-en-zh"
        ... )
        >>> zh2en_batch = tokenizer.prepare_zh2en_batch(["你好"], ["Hello"])
        >>> en2zh_batch = tokenizer.prepare_en2zh_batch(["Hello"], ["你好"])
    """
    
    def __init__(
        self,
        zh_en_model: str,
        en_zh_model: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化双语分词器
        
        参数：
            zh_en_model: 中→英模型路径
            en_zh_model: 英→中模型路径
            max_length: 最大序列长度
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.max_length = max_length
        
        # 加载中→英分词器
        self.logger.info("加载中→英分词器...")
        self.zh_en_tokenizer = TranslationTokenizer(
            model_name_or_path=zh_en_model,
            max_length=max_length,
            direction="zh2en",
            logger=self.logger
        )
        
        # 加载英→中分词器
        self.logger.info("加载英→中分词器...")
        self.en_zh_tokenizer = TranslationTokenizer(
            model_name_or_path=en_zh_model,
            max_length=max_length,
            direction="en2zh",
            logger=self.logger
        )
        
        self.logger.info("双语分词器初始化完成")
    
    def get_tokenizer(self, direction: str) -> TranslationTokenizer:
        """
        获取指定方向的分词器
        
        参数：
            direction: 翻译方向 ('zh2en' 或 'en2zh')
            
        返回：
            TranslationTokenizer: 对应的分词器
        """
        if direction == "zh2en":
            return self.zh_en_tokenizer
        elif direction == "en2zh":
            return self.en_zh_tokenizer
        else:
            raise ValueError(f"不支持的翻译方向: {direction}")
    
    def prepare_zh2en_batch(
        self,
        chinese_texts: List[str],
        english_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        准备中→英翻译批次
        
        参数：
            chinese_texts: 中文文本列表（源语言）
            english_texts: 英文文本列表（目标语言，训练时需要）
            
        返回：
            Dict[str, torch.Tensor]: 模型输入字典
        """
        return self.zh_en_tokenizer.prepare_translation_batch(
            src_texts=chinese_texts,
            tgt_texts=english_texts
        )
    
    def prepare_en2zh_batch(
        self,
        english_texts: List[str],
        chinese_texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        准备英→中翻译批次
        
        参数：
            english_texts: 英文文本列表（源语言）
            chinese_texts: 中文文本列表（目标语言，训练时需要）
            
        返回：
            Dict[str, torch.Tensor]: 模型输入字典
        """
        return self.en_zh_tokenizer.prepare_translation_batch(
            src_texts=english_texts,
            tgt_texts=chinese_texts
        )
    
    def encode_chinese(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        编码中文文本（用于中→英翻译）
        
        参数：
            texts: 中文文本
            
        返回：
            Dict[str, torch.Tensor]: 编码结果
        """
        return self.zh_en_tokenizer.encode(texts)
    
    def encode_english(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        编码英文文本（用于英→中翻译）
        
        参数：
            texts: 英文文本
            
        返回：
            Dict[str, torch.Tensor]: 编码结果
        """
        return self.en_zh_tokenizer.encode(texts)
    
    def decode_english(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码为英文文本
        
        参数：
            token_ids: token IDs
            skip_special_tokens: 是否跳过特殊 token
            
        返回：
            str: 英文文本
        """
        return self.zh_en_tokenizer.decode(token_ids, skip_special_tokens)
    
    def decode_chinese(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码为中文文本
        
        参数：
            token_ids: token IDs
            skip_special_tokens: 是否跳过特殊 token
            
        返回：
            str: 中文文本
        """
        return self.en_zh_tokenizer.decode(token_ids, skip_special_tokens)
    
    def get_vocab_info(self) -> Dict[str, Dict[str, int]]:
        """
        获取词表信息
        
        返回：
            Dict: 包含两个方向词表大小的字典
        """
        return {
            "zh2en": {
                "vocab_size": self.zh_en_tokenizer.vocab_size,
                "pad_token_id": self.zh_en_tokenizer.pad_token_id,
                "eos_token_id": self.zh_en_tokenizer.eos_token_id,
            },
            "en2zh": {
                "vocab_size": self.en_zh_tokenizer.vocab_size,
                "pad_token_id": self.en_zh_tokenizer.pad_token_id,
                "eos_token_id": self.en_zh_tokenizer.eos_token_id,
            }
        }
    
    def save(self, save_directory: Union[str, Path]) -> None:
        """
        保存双语分词器
        
        参数：
            save_directory: 保存目录
        """
        save_directory = Path(save_directory)
        
        # 保存中→英分词器
        zh_en_dir = save_directory / "zh2en"
        self.zh_en_tokenizer.save(zh_en_dir)
        
        # 保存英→中分词器
        en_zh_dir = save_directory / "en2zh"
        self.en_zh_tokenizer.save(en_zh_dir)
        
        self.logger.info(f"双语分词器已保存至: {save_directory}")


def analyze_tokenization(
    tokenizer: TranslationTokenizer,
    texts: List[str],
    show_samples: int = 5
) -> Dict[str, Any]:
    """
    分析分词效果
    
    参数：
        tokenizer: 分词器
        texts: 文本列表
        show_samples: 显示样本数量
        
    返回：
        Dict: 分词统计信息
    """
    total_tokens = 0
    total_chars = 0
    token_lengths = []
    
    samples = []
    
    for i, text in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)
        total_chars += len(text)
        token_lengths.append(len(tokens))
        
        # 收集样本
        if i < show_samples:
            samples.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "tokens": tokens[:20],
                "num_tokens": len(tokens)
            })
    
    return {
        "total_texts": len(texts),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "avg_tokens_per_text": total_tokens / len(texts) if texts else 0,
        "avg_chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "samples": samples
    }


# ====================================
# 命令行接口
# ====================================

def main():
    """
    命令行入口函数
    
    使用方式：
        python tokenizer.py --model models/Helsinki-NLP-opus-mt-zh-en --text "你好世界"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="翻译模型分词工具"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="要分词的文本"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="包含文本的文件路径"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"最大序列长度（默认: {DEFAULT_MAX_LENGTH}）"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="分析分词效果"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 创建分词器
    tokenizer = TranslationTokenizer(
        model_name_or_path=args.model,
        max_length=args.max_length
    )
    
    # 处理输入
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("请提供 --text 或 --file 参数")
        return
    
    # 分词或分析
    if args.analyze:
        stats = analyze_tokenization(tokenizer, texts)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        for text in texts:
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer.encode(text)
            print(f"\n原文: {text}")
            print(f"子词: {tokens}")
            print(f"IDs: {encoded['input_ids'].tolist()}")
            print(f"长度: {len(tokens)}")


if __name__ == "__main__":
    main()
