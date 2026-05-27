"""
模型配置模块

功能说明：
    提供 Helsinki-NLP MarianMT 模型的配置管理，包括：
    - 模型架构配置
    - 训练超参数配置
    - 混合精度配置（BF16）
    - 优化器配置

依赖：
    - transformers: MarianConfig
    - torch: 数据类型

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict

import torch
from transformers import (
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
)

# ====================================
# 常量定义
# ====================================

# 默认模型路径
DEFAULT_ZH_EN_MODEL = "Helsinki-NLP/opus-mt-zh-en"
DEFAULT_EN_ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"

# 默认训练配置
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5  # 微调推荐值，避免灾难性遗忘
DEFAULT_WARMUP_STEPS = 100
DEFAULT_NUM_EPOCHS = 5

# 混合精度类型
PRECISION_TYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class ModelConfig:
    """
    模型配置数据类
    
    属性：
        model_name_or_path: 模型名称或路径
        max_length: 最大序列长度
        vocab_size: 词表大小
        d_model: 模型隐藏层维度
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
        encoder_attention_heads: 编码器注意力头数
        decoder_attention_heads: 解码器注意力头数
        encoder_ffn_dim: 编码器 FFN 维度
        decoder_ffn_dim: 解码器 FFN 维度
        dropout: Dropout 比例
        activation_function: 激活函数
    """
    model_name_or_path: str = DEFAULT_ZH_EN_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    vocab_size: int = 65001
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    dropout: float = 0.1
    activation_function: str = "relu"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "ModelConfig":
        """
        从预训练模型加载配置
        
        参数：
            model_name_or_path: 模型名称或路径
            
        返回：
            ModelConfig: 模型配置
        """
        config = MarianConfig.from_pretrained(model_name_or_path)
        
        return cls(
            model_name_or_path=model_name_or_path,
            max_length=config.max_length,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            encoder_attention_heads=config.encoder_attention_heads,
            decoder_attention_heads=config.decoder_attention_heads,
            encoder_ffn_dim=config.encoder_ffn_dim,
            decoder_ffn_dim=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation_function=config.activation_function,
        )


@dataclass
class TrainingConfig:
    """
    训练配置数据类
    
    属性：
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        warmup_steps: 预热步数
        weight_decay: 权重衰减
        max_grad_norm: 梯度裁剪阈值
        save_steps: 保存间隔
        eval_steps: 评估间隔
        logging_steps: 日志间隔
        fp16: 是否使用 FP16
        bf16: 是否使用 BF16
        gradient_checkpointing: 是否使用梯度检查点
        seed: 随机种子
    """
    output_dir: str = "outputs/checkpoints"
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = 8
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    fp16: bool = False
    bf16: bool = True  # RTX 5090 推荐使用 BF16
    gradient_checkpointing: bool = True
    seed: int = 42
    
    # 课程学习配置
    curriculum_enabled: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        """计算等效批次大小"""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def precision(self) -> str:
        """获取精度类型"""
        if self.bf16:
            return "bf16"
        elif self.fp16:
            return "fp16"
        return "fp32"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class OptimizerConfig:
    """
    优化器配置数据类
    
    属性：
        optimizer_type: 优化器类型
        learning_rate: 学习率
        betas: Adam betas
        eps: Adam epsilon
        weight_decay: 权重衰减
        scheduler_type: 调度器类型
        warmup_steps: 预热步数
        num_training_steps: 总训练步数
    """
    optimizer_type: str = "adamw"
    learning_rate: float = DEFAULT_LEARNING_RATE
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"  # linear, cosine, constant
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    warmup_ratio: float = 0.1
    num_training_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "scheduler_type": self.scheduler_type,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "num_training_steps": self.num_training_steps,
        }


class NMTModelManager:
    """
    NMT 模型管理器
    
    功能说明：
        统一管理模型的加载、配置和保存：
        - 支持中英双向模型
        - 配置混合精度训练
        - 支持梯度检查点
    
    参数：
        zh_en_model: 中→英模型路径
        en_zh_model: 英→中模型路径
        device: 计算设备
        precision: 精度类型
        logger: 日志记录器
        
    示例：
        >>> manager = NMTModelManager(
        ...     zh_en_model="models/Helsinki-NLP-opus-mt-zh-en",
        ...     en_zh_model="models/Helsinki-NLP-opus-mt-en-zh"
        ... )
        >>> zh2en_model = manager.get_model("zh2en")
    """
    
    def __init__(
        self,
        zh_en_model: str = DEFAULT_ZH_EN_MODEL,
        en_zh_model: str = DEFAULT_EN_ZH_MODEL,
        device: Optional[str] = None,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化模型管理器
        
        参数：
            zh_en_model: 中→英模型路径
            en_zh_model: 英→中模型路径
            device: 计算设备
            precision: 精度类型
            gradient_checkpointing: 是否启用梯度检查点
            logger: 日志记录器
        """
        self.zh_en_model_path = zh_en_model
        self.en_zh_model_path = en_zh_model
        self.precision = precision
        self.gradient_checkpointing = gradient_checkpointing
        self.logger = logger or logging.getLogger(__name__)
        
        # 确定设备
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"使用 GPU: {gpu_name}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("使用 CPU")
        
        # 获取精度类型
        self.dtype = PRECISION_TYPES.get(precision, torch.float32)
        
        # 模型缓存
        self._models: Dict[str, MarianMTModel] = {}
        self._tokenizers: Dict[str, MarianTokenizer] = {}
        self._configs: Dict[str, ModelConfig] = {}
        
        self.logger.info(f"模型管理器初始化完成")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  精度: {precision}")
    
    def load_model(
        self,
        direction: str,
        for_training: bool = True
    ) -> MarianMTModel:
        """
        加载指定方向的模型
        
        参数：
            direction: 翻译方向 ('zh2en' 或 'en2zh')
            for_training: 是否用于训练
            
        返回：
            MarianMTModel: 加载的模型
        """
        if direction in self._models:
            return self._models[direction]
        
        # 确定模型路径
        if direction == "zh2en":
            model_path = self.zh_en_model_path
        elif direction == "en2zh":
            model_path = self.en_zh_model_path
        else:
            raise ValueError(f"不支持的翻译方向: {direction}")
        
        self.logger.info(f"加载模型: {model_path}")
        
        # 加载模型
        model = MarianMTModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype if not for_training else torch.float32
        )
        
        # 启用梯度检查点（节省显存）
        if for_training and self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            self.logger.info("已启用梯度检查点")
        
        # 移动到设备
        model = model.to(self.device)
        
        # 缓存模型
        self._models[direction] = model
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型加载完成:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        
        return model
    
    def load_tokenizer(self, direction: str) -> MarianTokenizer:
        """
        加载指定方向的分词器
        
        参数：
            direction: 翻译方向
            
        返回：
            MarianTokenizer: 分词器
        """
        if direction in self._tokenizers:
            return self._tokenizers[direction]
        
        if direction == "zh2en":
            model_path = self.zh_en_model_path
        elif direction == "en2zh":
            model_path = self.en_zh_model_path
        else:
            raise ValueError(f"不支持的翻译方向: {direction}")
        
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        self._tokenizers[direction] = tokenizer
        
        return tokenizer
    
    def get_model(self, direction: str) -> MarianMTModel:
        """获取模型（如未加载则自动加载）"""
        if direction not in self._models:
            self.load_model(direction)
        return self._models[direction]
    
    def get_tokenizer(self, direction: str) -> MarianTokenizer:
        """获取分词器（如未加载则自动加载）"""
        if direction not in self._tokenizers:
            self.load_tokenizer(direction)
        return self._tokenizers[direction]
    
    def get_config(self, direction: str) -> ModelConfig:
        """获取模型配置"""
        if direction not in self._configs:
            if direction == "zh2en":
                model_path = self.zh_en_model_path
            else:
                model_path = self.en_zh_model_path
            
            self._configs[direction] = ModelConfig.from_pretrained(model_path)
        
        return self._configs[direction]
    
    def save_model(
        self,
        direction: str,
        output_dir: Union[str, Path]
    ) -> None:
        """
        保存模型
        
        参数：
            direction: 翻译方向
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.get_model(direction)
        tokenizer = self.get_tokenizer(direction)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"模型已保存至: {output_dir}")
    
    def get_model_info(self, direction: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        参数：
            direction: 翻译方向
            
        返回：
            Dict: 模型信息
        """
        model = self.get_model(direction)
        config = self.get_config(direction)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            "direction": direction,
            "model_path": config.model_name_or_path,
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "encoder_layers": config.encoder_layers,
            "decoder_layers": config.decoder_layers,
            "attention_heads": config.encoder_attention_heads,
            "ffn_dim": config.encoder_ffn_dim,
            "total_params": total_params,
            "size_mb": total_params * 4 / 1024 / 1024,
            "device": str(self.device),
            "precision": self.precision,
        }


def load_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从 YAML 文件加载配置
    
    参数：
        config_path: 配置文件路径
        
    返回：
        Dict: 配置字典
    """
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_training_config_from_yaml(
    config_path: Union[str, Path]
) -> TrainingConfig:
    """
    从 YAML 文件创建训练配置
    
    参数：
        config_path: 配置文件路径
        
    返回：
        TrainingConfig: 训练配置
    """
    config = load_config_from_yaml(config_path)
    training_config = config.get("training", {})
    
    return TrainingConfig(
        output_dir=config.get("output", {}).get("checkpoint_dir", "outputs/checkpoints"),
        num_epochs=training_config.get("num_epochs", DEFAULT_NUM_EPOCHS),
        batch_size=training_config.get("batch_size", DEFAULT_BATCH_SIZE),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", DEFAULT_LEARNING_RATE),
        warmup_steps=training_config.get("warmup_steps", DEFAULT_WARMUP_STEPS),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        save_steps=training_config.get("save_steps", 1000),
        eval_steps=training_config.get("eval_steps", 500),
        logging_steps=training_config.get("logging_steps", 100),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        seed=training_config.get("seed", 42),
        bf16=config.get("model", {}).get("dtype", "bf16") == "bf16",
        curriculum_enabled=config.get("curriculum", {}).get("enabled", True),
    )


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="模型配置工具"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示模型信息"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.info:
        config = ModelConfig.from_pretrained(args.model)
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
