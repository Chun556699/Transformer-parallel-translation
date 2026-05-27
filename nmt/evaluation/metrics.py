"""
多元评估指标模块

功能说明：
    提供多种翻译质量评估指标：
    - sacreBLEU：标准化 BLEU，行业基准
    - COMET-22：神经网络评估，基于 XLM-R
    - BERTScore：基于 BERT 的语义相似度
    - chrF++：字符级 F-score（对中文友好）
    - TER：翻译编辑率

评测目标：
    - BLEU ≥ 30.0
    - COMET ≥ 0.80

依赖：
    - sacrebleu: BLEU, chrF, TER
    - comet: COMET 评估
    - bert-score: BERTScore

作者：NMT Project
版本：1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np

# 评估库（可选导入）
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF, TER
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("sacrebleu 未安装")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    logging.warning("comet 未安装")

try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score 未安装")


# ====================================
# 常量定义
# ====================================

# COMET 模型
DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"

# BERTScore 模型
DEFAULT_BERTSCORE_MODEL = "bert-base-multilingual-cased"

# 目标指标
TARGET_BLEU = 30.0
TARGET_COMET = 0.80


@dataclass
class EvaluationResult:
    """
    评估结果数据类
    
    属性：
        bleu: sacreBLEU 分数
        comet: COMET 分数
        bertscore_precision: BERTScore 精确率
        bertscore_recall: BERTScore 召回率
        bertscore_f1: BERTScore F1
        chrf: chrF++ 分数
        ter: TER 分数
    """
    bleu: Optional[float] = None
    comet: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    chrf: Optional[float] = None
    ter: Optional[float] = None
    
    # 详细信息
    bleu_details: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "bleu": self.bleu,
            "comet": self.comet,
            "bertscore_precision": self.bertscore_precision,
            "bertscore_recall": self.bertscore_recall,
            "bertscore_f1": self.bertscore_f1,
            "chrf": self.chrf,
            "ter": self.ter,
        }
    
    def __str__(self) -> str:
        """格式化输出"""
        lines = ["评估结果:", "=" * 40]
        
        if self.bleu is not None:
            status = "✓" if self.bleu >= TARGET_BLEU else "✗"
            lines.append(f"  {status} BLEU: {self.bleu:.2f} (目标: ≥{TARGET_BLEU})")
        
        if self.comet is not None:
            status = "✓" if self.comet >= TARGET_COMET else "✗"
            lines.append(f"  {status} COMET: {self.comet:.4f} (目标: ≥{TARGET_COMET})")
        
        if self.bertscore_f1 is not None:
            lines.append(f"    BERTScore F1: {self.bertscore_f1:.4f}")
        
        if self.chrf is not None:
            lines.append(f"    chrF++: {self.chrf:.2f}")
        
        if self.ter is not None:
            lines.append(f"    TER: {self.ter:.2f}")
        
        lines.append("=" * 40)
        return "\n".join(lines)


class MultiMetricEvaluator:
    """
    多指标评估器
    
    功能说明：
        提供多种翻译质量评估指标的统一接口：
        - sacreBLEU
        - COMET
        - BERTScore
        - chrF++
        - TER
    
    参数：
        comet_model: COMET 模型名称
        bertscore_model: BERTScore 模型名称
        device: 计算设备
        logger: 日志记录器
        
    示例：
        >>> evaluator = MultiMetricEvaluator()
        >>> result = evaluator.evaluate(hypotheses, references)
        >>> print(result)
    """
    
    def __init__(
        self,
        comet_model: str = DEFAULT_COMET_MODEL,
        bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化评估器
        
        参数：
            comet_model: COMET 模型名称
            bertscore_model: BERTScore 模型名称
            device: 计算设备
            logger: 日志记录器
        """
        self.comet_model_name = comet_model
        self.bertscore_model_name = bertscore_model
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 延迟加载模型
        self._comet_model = None
        self._bert_scorer = None
        
        self.logger.info("评估器初始化完成")
        self.logger.info(f"  可用指标: {self._get_available_metrics()}")
    
    def _get_available_metrics(self) -> List[str]:
        """获取可用的评估指标"""
        metrics = []
        if SACREBLEU_AVAILABLE:
            metrics.extend(["bleu", "chrf", "ter"])
        if COMET_AVAILABLE:
            metrics.append("comet")
        if BERTSCORE_AVAILABLE:
            metrics.append("bertscore")
        return metrics
    
    def _load_comet_model(self):
        """加载 COMET 模型"""
        if self._comet_model is None and COMET_AVAILABLE:
            self.logger.info(f"加载 COMET 模型: {self.comet_model_name}")
            try:
                model_path = download_model(self.comet_model_name)
                self._comet_model = load_from_checkpoint(model_path)
                self.logger.info("COMET 模型加载完成")
            except Exception as e:
                self.logger.error(f"COMET 模型加载失败: {e}")
                self._comet_model = None
    
    def _load_bert_scorer(self):
        """加载 BERTScorer"""
        if self._bert_scorer is None and BERTSCORE_AVAILABLE:
            self.logger.info(f"加载 BERTScorer: {self.bertscore_model_name}")
            try:
                self._bert_scorer = BERTScorer(
                    model_type=self.bertscore_model_name,
                    device=self.device
                )
                self.logger.info("BERTScorer 加载完成")
            except Exception as e:
                self.logger.error(f"BERTScorer 加载失败: {e}")
                self._bert_scorer = None
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Tuple[float, Dict]:
        """
        计算 sacreBLEU 分数
        
        参数：
            hypotheses: 翻译结果列表
            references: 参考译文列表
            
        返回：
            Tuple[float, Dict]: (BLEU 分数, 详细信息)
        """
        if not SACREBLEU_AVAILABLE:
            self.logger.warning("sacrebleu 不可用")
            return None, {}
        
        # sacreBLEU 需要参考译文为列表的列表
        bleu = BLEU()
        result = bleu.corpus_score(hypotheses, [references])
        
        return result.score, {
            "score": result.score,
            "counts": result.counts,
            "totals": result.totals,
            "precisions": result.precisions,
            "bp": result.bp,
            "sys_len": result.sys_len,
            "ref_len": result.ref_len,
        }
    
    def compute_comet(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """
        计算 COMET 分数
        
        参数：
            sources: 源语言文本列表
            hypotheses: 翻译结果列表
            references: 参考译文列表
            
        返回：
            float: COMET 分数
        """
        if not COMET_AVAILABLE:
            self.logger.warning("COMET 不可用")
            return None
        
        self._load_comet_model()
        
        if self._comet_model is None:
            return None
        
        # 准备数据
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        
        # 计算分数
        try:
            output = self._comet_model.predict(data, batch_size=32, gpus=1)
            return output.system_score
        except Exception as e:
            self.logger.error(f"COMET 计算失败: {e}")
            return None
    
    def compute_bertscore(
        self,
        hypotheses: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Tuple[float, float, float]:
        """
        计算 BERTScore
        
        参数：
            hypotheses: 翻译结果列表
            references: 参考译文列表
            lang: 语言代码
            
        返回：
            Tuple[float, float, float]: (精确率, 召回率, F1)
        """
        if not BERTSCORE_AVAILABLE:
            self.logger.warning("BERTScore 不可用")
            return None, None, None
        
        self._load_bert_scorer()
        
        if self._bert_scorer is None:
            return None, None, None
        
        try:
            P, R, F1 = self._bert_scorer.score(hypotheses, references)
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            self.logger.error(f"BERTScore 计算失败: {e}")
            return None, None, None
    
    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """
        计算 chrF++ 分数
        
        chrF 是基于字符级 n-gram 的评估指标，
        对中文等不使用空格分词的语言更友好。
        
        参数：
            hypotheses: 翻译结果列表
            references: 参考译文列表
            
        返回：
            float: chrF++ 分数
        """
        if not SACREBLEU_AVAILABLE:
            self.logger.warning("sacrebleu 不可用")
            return None
        
        chrf = CHRF(word_order=2)  # chrF++
        result = chrf.corpus_score(hypotheses, [references])
        
        return result.score
    
    def compute_ter(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> float:
        """
        计算 TER (Translation Edit Rate)
        
        TER 衡量将翻译结果编辑为参考译文所需的最小编辑次数，
        越低越好。
        
        参数：
            hypotheses: 翻译结果列表
            references: 参考译文列表
            
        返回：
            float: TER 分数
        """
        if not SACREBLEU_AVAILABLE:
            self.logger.warning("sacrebleu 不可用")
            return None
        
        ter = TER()
        result = ter.corpus_score(hypotheses, [references])
        
        return result.score
    
    def evaluate(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        执行全面评估
        
        参数：
            hypotheses: 翻译结果列表
            references: 参考译文列表
            sources: 源语言文本列表（COMET 需要）
            metrics: 要计算的指标列表
            
        返回：
            EvaluationResult: 评估结果
        """
        if metrics is None:
            metrics = self._get_available_metrics()
        
        result = EvaluationResult()
        
        # 计算 BLEU
        if "bleu" in metrics:
            self.logger.info("计算 BLEU...")
            bleu_score, bleu_details = self.compute_bleu(hypotheses, references)
            result.bleu = bleu_score
            result.bleu_details = bleu_details
        
        # 计算 COMET
        if "comet" in metrics and sources:
            self.logger.info("计算 COMET...")
            result.comet = self.compute_comet(sources, hypotheses, references)
        
        # 计算 BERTScore
        if "bertscore" in metrics:
            self.logger.info("计算 BERTScore...")
            p, r, f1 = self.compute_bertscore(hypotheses, references)
            result.bertscore_precision = p
            result.bertscore_recall = r
            result.bertscore_f1 = f1
        
        # 计算 chrF++
        if "chrf" in metrics:
            self.logger.info("计算 chrF++...")
            result.chrf = self.compute_chrf(hypotheses, references)
        
        # 计算 TER
        if "ter" in metrics:
            self.logger.info("计算 TER...")
            result.ter = self.compute_ter(hypotheses, references)
        
        self.logger.info("评估完成")
        return result


def evaluate_translation(
    hypotheses: List[str],
    references: List[str],
    sources: Optional[List[str]] = None
) -> EvaluationResult:
    """
    便捷函数：评估翻译质量
    
    参数：
        hypotheses: 翻译结果列表
        references: 参考译文列表
        sources: 源语言文本列表
        
    返回：
        EvaluationResult: 评估结果
    """
    evaluator = MultiMetricEvaluator()
    return evaluator.evaluate(hypotheses, references, sources)


def compute_significance(
    scores_a: List[float],
    scores_b: List[float],
    num_samples: int = 1000
) -> Tuple[float, bool]:
    """
    Bootstrap 显著性检验
    
    参数：
        scores_a: 系统 A 的分数列表
        scores_b: 系统 B 的分数列表
        num_samples: 重采样次数
        
    返回：
        Tuple[float, bool]: (p 值, 是否显著)
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    observed_diff = scores_a.mean() - scores_b.mean()
    
    # Bootstrap 重采样
    count = 0
    n = len(scores_a)
    
    for _ in range(num_samples):
        # 随机重采样
        indices = np.random.randint(0, n, n)
        sample_a = scores_a[indices]
        sample_b = scores_b[indices]
        
        # 计算差异
        sample_diff = sample_a.mean() - sample_b.mean()
        
        if sample_diff >= observed_diff:
            count += 1
    
    p_value = count / num_samples
    is_significant = p_value < 0.05
    
    return p_value, is_significant


# ====================================
# 命令行接口
# ====================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="翻译评估工具"
    )
    parser.add_argument(
        "--hypotheses", "-hyp",
        type=str,
        required=True,
        help="翻译结果文件路径（每行一个）"
    )
    parser.add_argument(
        "--references", "-ref",
        type=str,
        required=True,
        help="参考译文文件路径（每行一个）"
    )
    parser.add_argument(
        "--sources", "-src",
        type=str,
        help="源语言文件路径（COMET 需要）"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["bleu", "chrf", "ter"],
        help="要计算的指标"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出结果文件路径"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 加载数据
    with open(args.hypotheses, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip() for line in f]
    
    with open(args.references, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    sources = None
    if args.sources:
        with open(args.sources, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f]
    
    # 评估
    evaluator = MultiMetricEvaluator()
    result = evaluator.evaluate(
        hypotheses=hypotheses,
        references=references,
        sources=sources,
        metrics=args.metrics
    )
    
    # 输出结果
    print(result)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
