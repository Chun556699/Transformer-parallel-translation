"""
训练器模块

功能说明：
    提供翻译模型的训练功能，包括：
    - 基于 transformers Trainer 的训练流程
    - 课程学习集成
    - 混合精度训练（BF16）
    - 梯度累积与检查点
    - 训练过程可视化

依赖：
    - transformers: Trainer, TrainingArguments
    - torch: PyTorch 核心
    - tqdm: 进度条

作者：NMT Project
版本：1.0.0
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler,
)
from transformers.trainer_utils import EvalPrediction

from tqdm import tqdm

# 尝试导入本地模块
try:
    from ..data import (
        TranslationDataset,
        CurriculumSampler,
        create_dataloader,
    )
    from ..model.config import TrainingConfig, ModelConfig
except ImportError:
    pass


# ====================================
# 常量定义
# ====================================

# 默认训练配置
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5  # 微调推荐值
DEFAULT_NUM_EPOCHS = 5
DEFAULT_WARMUP_STEPS = 100


@dataclass
class TrainingState:
    """
    训练状态数据类
    
    属性：
        epoch: 当前 epoch
        global_step: 全局步数
        best_loss: 最佳损失
        best_bleu: 最佳 BLEU
        training_loss: 训练损失历史
        validation_loss: 验证损失历史
    """
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_bleu: float = 0.0
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "best_bleu": self.best_bleu,
            "training_loss": self.training_loss[-100:],  # 只保留最近 100 条
            "validation_loss": self.validation_loss,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """保存状态"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingState":
        """加载状态"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        state = cls()
        state.epoch = data.get("epoch", 0)
        state.global_step = data.get("global_step", 0)
        state.best_loss = data.get("best_loss", float('inf'))
        state.best_bleu = data.get("best_bleu", 0.0)
        state.training_loss = data.get("training_loss", [])
        state.validation_loss = data.get("validation_loss", [])
        
        return state


class TranslationTrainer:
    """
    翻译模型训练器
    
    功能说明：
        - 支持 Helsinki-NLP MarianMT 模型微调
        - 集成课程学习策略
        - 支持混合精度训练 (BF16)
        - 提供完整的训练监控
    
    参数：
        model: 翻译模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        output_dir: 输出目录
        training_config: 训练配置
        curriculum_sampler: 课程学习采样器（可选）
        logger: 日志记录器
        
    示例：
        >>> trainer = TranslationTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     output_dir="outputs/checkpoints"
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: MarianMTModel,
        tokenizer: MarianTokenizer,
        train_dataset: TranslationDataset,
        eval_dataset: Optional[TranslationDataset] = None,
        output_dir: str = "outputs/checkpoints",
        training_config: Optional[TrainingConfig] = None,
        curriculum_sampler: Optional[CurriculumSampler] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化训练器
        
        参数：
            model: 翻译模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            output_dir: 输出目录
            training_config: 训练配置
            curriculum_sampler: 课程学习采样器
            logger: 日志记录器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.config = training_config or TrainingConfig()
        self.curriculum_sampler = curriculum_sampler
        self.logger = logger or logging.getLogger(__name__)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定设备
        self.device = next(model.parameters()).device
        
        # 训练状态
        self.state = TrainingState()
        
        # 初始化优化器和调度器
        self._setup_optimizer()
        
        self.logger.info("训练器初始化完成")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  训练集大小: {len(train_dataset)}")
        self.logger.info(f"  批次大小: {self.config.batch_size}")
        self.logger.info(f"  等效批次: {self.config.effective_batch_size}")
        self.logger.info(f"  总 epoch: {self.config.num_epochs}")
    
    def _setup_optimizer(self) -> None:
        """
        设置优化器和学习率调度器
        
        使用 AdamW 优化器 + 线性预热 + 余弦退火调度。
        """
        # 过滤不需要权重衰减的参数
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # 创建 AdamW 优化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 计算总训练步数
        num_training_steps = (
            len(self.train_dataset) // self.config.batch_size
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        
        # 创建调度器（预热 + 余弦退火）
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"优化器设置完成:")
        self.logger.info(f"  学习率: {self.config.learning_rate}")
        self.logger.info(f"  预热步数: {self.config.warmup_steps}")
        self.logger.info(f"  总训练步数: {num_training_steps}")
    
    def _create_dataloader(
        self,
        dataset: TranslationDataset,
        shuffle: bool = True,
        sampler: Optional[CurriculumSampler] = None
    ) -> DataLoader:
        """
        创建数据加载器
        
        参数：
            dataset: 数据集
            shuffle: 是否打乱
            sampler: 采样器
            
        返回：
            DataLoader: 数据加载器
        """
        from ..data.dataset import collate_fn
        
        if sampler is not None:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn
            )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        执行单个 epoch 的训练
        
        参数：
            dataloader: 训练数据加载器
            epoch: 当前 epoch 编号
        
        返回：
            float: 平均训练损失
        """
        # 设置模型为训练模式
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            leave=True
        )
        
        # 梯度累积计数
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(
                enabled=self.config.bf16 or self.config.fp16,
                dtype=torch.bfloat16 if self.config.bf16 else torch.float16
            ):
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            accumulation_counter += 1
            
            # 梯度累积
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 更新参数
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                accumulation_counter = 0
                self.state.global_step += 1
            
            # 记录损失
            batch_loss = loss.item() * self.config.gradient_accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            
            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 记录学习率
            self.state.learning_rates.append(current_lr)
            
            # 定期记录
            if self.state.global_step % self.config.logging_steps == 0:
                self.state.training_loss.append(total_loss / num_batches)
        
        # 安全检查：防止除零错误
        if num_batches == 0:
            self.logger.warning(
                "训练循环没有执行任何批次！"
                "可能是因为课程学习采样器返回了空索引。"
                "请检查数据集和采样器配置。"
            )
            return 0.0
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        参数：
            dataloader: 评估数据加载器
            
        返回：
            Dict[str, float]: 评估指标
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="评估中", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 计算困惑度
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def train(self) -> TrainingState:
        """
        执行完整训练流程
        
        返回：
            TrainingState: 训练状态
        """
        self.logger.info("开始训练...")
        start_time = time.time()
        
        # 创建评估数据加载器
        eval_dataloader = None
        if self.eval_dataset:
            eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                shuffle=False
            )
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            
            # 课程学习：更新采样器的 epoch
            if self.curriculum_sampler:
                self.curriculum_sampler.set_epoch(epoch)
            
            # 创建训练数据加载器
            train_dataloader = self._create_dataloader(
                self.train_dataset,
                shuffle=True if not self.curriculum_sampler else False,
                sampler=self.curriculum_sampler
            )
            
            # 训练一个 epoch
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            self.logger.info(f"Epoch {epoch + 1} 训练损失: {train_loss:.4f}")
            
            # 评估
            if eval_dataloader:
                eval_metrics = self.evaluate(eval_dataloader)
                eval_loss = eval_metrics["eval_loss"]
                perplexity = eval_metrics["perplexity"]
                
                self.state.validation_loss.append(eval_loss)
                
                self.logger.info(
                    f"Epoch {epoch + 1} 验证损失: {eval_loss:.4f}, "
                    f"困惑度: {perplexity:.2f}"
                )
                
                # 保存最佳模型
                if eval_loss < self.state.best_loss:
                    self.state.best_loss = eval_loss
                    self._save_checkpoint("best")
                    self.logger.info(f"保存最佳模型 (loss: {eval_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % 1 == 0:  # 每个 epoch 保存一次
                self._save_checkpoint(f"epoch_{epoch + 1}")
        
        # 保存最终模型
        self._save_checkpoint("final")
        
        # 计算训练时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.logger.info(f"训练完成！总用时: {hours}小时{minutes}分钟")
        self.logger.info(f"最佳验证损失: {self.state.best_loss:.4f}")
        
        return self.state
    
    def _save_checkpoint(self, name: str) -> None:
        """
        保存检查点
        
        参数：
            name: 检查点名称
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        self.state.save(checkpoint_dir / "training_state.json")
        
        # 保存优化器状态
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        torch.save(
            self.scheduler.state_dict(),
            checkpoint_dir / "scheduler.pt"
        )
    
    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """
        加载检查点
        
        参数：
            checkpoint_dir: 检查点目录
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # 加载模型
        self.model = MarianMTModel.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # 加载训练状态
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            self.state = TrainingState.load(state_path)
        
        # 加载优化器状态
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path))
        
        self.logger.info(f"检查点加载完成: {checkpoint_dir}")


class HuggingFaceTrainer:
    """
    基于 Hugging Face Trainer 的训练器封装
    
    功能说明：
        使用 transformers.Trainer 进行训练，简化训练流程。
    
    参数：
        model: 翻译模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        training_args: 训练参数
        
    示例：
        >>> trainer = HuggingFaceTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: MarianMTModel,
        tokenizer: MarianTokenizer,
        train_dataset: TranslationDataset,
        eval_dataset: Optional[TranslationDataset] = None,
        output_dir: str = "outputs/checkpoints",
        training_config: Optional[TrainingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化训练器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config or TrainingConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建训练参数
        self.training_args = self._create_training_args(output_dir)
        
        # 创建 Trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=self._data_collator,
        )
    
    def _create_training_args(self, output_dir: str) -> TrainingArguments:
        """
        创建 TrainingArguments
        
        参数：
            output_dir: 输出目录
            
        返回：
            TrainingArguments: 训练参数
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if self.eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            logging_dir=f"{output_dir}/logs",
            report_to=["tensorboard"],
            remove_unused_columns=False,
        )
    
    def _data_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        数据整理函数
        
        参数：
            features: 特征列表
            
        返回：
            Dict: 整理后的批次数据
        """
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if "labels" in features[0]:
            labels = torch.stack([f["labels"] for f in features])
            batch["labels"] = labels
        
        return batch
    
    def train(self):
        """执行训练"""
        self.logger.info("开始训练（使用 Hugging Face Trainer）...")
        self.trainer.train()
        self.logger.info("训练完成！")
    
    def evaluate(self):
        """执行评估"""
        if self.eval_dataset:
            return self.trainer.evaluate()
        return None
    
    def save_model(self, output_dir: str):
        """保存模型"""
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# ====================================
# 便捷函数
# ====================================

def create_trainer(
    model_path: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "outputs/checkpoints",
    config_path: Optional[str] = None,
    use_curriculum: bool = True,
    direction: str = "zh2en"
) -> TranslationTrainer:
    """
    创建训练器的便捷函数
    
    参数：
        model_path: 模型路径
        train_data_path: 训练数据路径
        eval_data_path: 验证数据路径
        output_dir: 输出目录
        config_path: 配置文件路径
        use_curriculum: 是否使用课程学习
        direction: 翻译方向
        
    返回：
        TranslationTrainer: 训练器实例
    """
    from transformers import MarianMTModel, MarianTokenizer
    from ..data import TranslationDataset, TranslationTokenizer, CurriculumSampler
    from ..model.config import TrainingConfig, create_training_config_from_yaml
    
    # 加载配置
    if config_path:
        training_config = create_training_config_from_yaml(config_path)
    else:
        training_config = TrainingConfig()
    
    # 加载模型和分词器
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    # 创建分词器封装
    translation_tokenizer = TranslationTokenizer(
        model_name_or_path=model_path,
        direction=direction
    )
    
    # 确定字段名
    if direction == "zh2en":
        src_key, tgt_key = "chinese", "english"
    else:
        src_key, tgt_key = "english", "chinese"
    
    # 创建数据集
    train_dataset = TranslationDataset(
        data=train_data_path,
        tokenizer=translation_tokenizer,
        src_key=src_key,
        tgt_key=tgt_key,
        direction=direction
    )
    
    eval_dataset = None
    if eval_data_path:
        eval_dataset = TranslationDataset(
            data=eval_data_path,
            tokenizer=translation_tokenizer,
            src_key=src_key,
            tgt_key=tgt_key,
            direction=direction
        )
    
    # 创建课程学习采样器
    curriculum_sampler = None
    if use_curriculum and training_config.curriculum_enabled:
        curriculum_sampler = CurriculumSampler(
            dataset=train_dataset,
            total_epochs=training_config.num_epochs
        )
    
    # 移动模型到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 启用梯度检查点
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 创建训练器
    trainer = TranslationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        training_config=training_config,
        curriculum_sampler=curriculum_sampler
    )
    
    return trainer


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="翻译模型训练工具"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="训练数据路径"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        help="验证数据路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/checkpoints",
        help="输出目录"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="zh2en",
        choices=["zh2en", "en2zh"],
        help="翻译方向"
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="禁用课程学习"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 创建并运行训练器
    trainer = create_trainer(
        model_path=args.model,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output,
        config_path=args.config,
        use_curriculum=not args.no_curriculum,
        direction=args.direction
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
