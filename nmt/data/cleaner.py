"""
数据清洗与预处理模块

功能说明：
    提供中英翻译数据的清洗与预处理功能，包括：
    - 去除重复样本（基于 hash 去重）
    - Unicode 标准化（NFKC）
    - 清洗 HTML 标签、特殊字符
    - 语言检测与过滤（langdetect）
    - 长度过滤（中文 5-200 字符，英文 10-500 字符）
    - 格式标准化

依赖：
    - langdetect: 语言检测
    - regex: 正则表达式增强库
    - tqdm: 进度条显示

作者：NMT Project
版本：1.0.0
"""

import os
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, field
from collections import Counter
import logging

from tqdm import tqdm

# 尝试导入语言检测库
try:
    from langdetect import detect, DetectorFactory
    # 设置确定性模式，确保结果可复现
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect 未安装，语言检测功能将被禁用")


# ====================================
# 常量定义
# ====================================

# 中文字符长度范围
MIN_CHINESE_LENGTH = 5
MAX_CHINESE_LENGTH = 200

# 英文字符长度范围
MIN_ENGLISH_LENGTH = 10
MAX_ENGLISH_LENGTH = 500

# 中英长度比例范围（中文字符数 / 英文字符数）
MIN_LENGTH_RATIO = 0.1
MAX_LENGTH_RATIO = 0.8

# HTML 标签正则表达式
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# URL 正则表达式
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-._~:/?#\[\]@!$&\'()*+,;=%]*'
)

# 邮箱正则表达式
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

# 特殊字符清理（保留基本标点）
SPECIAL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

# 多余空白字符
MULTI_SPACE_PATTERN = re.compile(r'\s+')

# 中文字符范围正则
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

# 中文标点符号
CHINESE_PUNCTUATION = '，。！？、；：""''（）【】《》…—～·'

# 英文标点符号
ENGLISH_PUNCTUATION = ',.!?;:\'"()-[]<>...-~.'


@dataclass
class CleaningStats:
    """
    清洗统计信息数据类
    
    功能说明：
        记录数据清洗过程中的各项统计指标
    
    属性：
        total_samples: 原始样本总数
        duplicate_removed: 去重删除数量
        html_cleaned: HTML 清洗数量
        length_filtered: 长度过滤数量
        language_filtered: 语言过滤数量
        empty_filtered: 空白过滤数量
        final_samples: 最终保留数量
    """
    total_samples: int = 0
    duplicate_removed: int = 0
    html_cleaned: int = 0
    length_filtered: int = 0
    language_filtered: int = 0
    empty_filtered: int = 0
    ratio_filtered: int = 0
    final_samples: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式"""
        return {
            "原始样本数": self.total_samples,
            "去重删除": self.duplicate_removed,
            "HTML清洗": self.html_cleaned,
            "长度过滤": self.length_filtered,
            "语言过滤": self.language_filtered,
            "空白过滤": self.empty_filtered,
            "比例过滤": self.ratio_filtered,
            "最终保留": self.final_samples,
        }
    
    def __str__(self) -> str:
        """格式化输出统计信息"""
        lines = [
            "=" * 50,
            "数据清洗统计报告",
            "=" * 50,
        ]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value:,}")
        
        # 计算保留率
        if self.total_samples > 0:
            retention_rate = self.final_samples / self.total_samples * 100
            lines.append(f"  保留率: {retention_rate:.2f}%")
        
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class SamplePair:
    """
    翻译样本对数据类
    
    属性：
        chinese: 中文文本
        english: 英文文本
        source: 数据来源（可选）
        score: 质量评分（可选，后续筛选使用）
    """
    chinese: str
    english: str
    source: Optional[str] = None
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "chinese": self.chinese,
            "english": self.english,
        }
        if self.source:
            result["source"] = self.source
        if self.score is not None:
            result["score"] = self.score
        return result


class DataCleaner:
    """
    数据清洗器
    
    功能说明：
        对中英翻译数据进行全面清洗和预处理，支持：
        - 批量处理大规模数据
        - 自定义清洗规则
        - 详细的统计报告
    
    参数：
        min_zh_length: 中文最小长度（字符数）
        max_zh_length: 中文最大长度（字符数）
        min_en_length: 英文最小长度（字符数）
        max_en_length: 英文最大长度（字符数）
        enable_language_detection: 是否启用语言检测
        remove_urls: 是否移除 URL
        remove_emails: 是否移除邮箱地址
        normalize_unicode: 是否进行 Unicode 标准化
        
    示例：
        >>> cleaner = DataCleaner(min_zh_length=5, max_zh_length=200)
        >>> cleaned_samples = cleaner.clean_dataset(raw_samples)
        >>> print(cleaner.stats)
    """
    
    def __init__(
        self,
        min_zh_length: int = MIN_CHINESE_LENGTH,
        max_zh_length: int = MAX_CHINESE_LENGTH,
        min_en_length: int = MIN_ENGLISH_LENGTH,
        max_en_length: int = MAX_ENGLISH_LENGTH,
        min_length_ratio: float = MIN_LENGTH_RATIO,
        max_length_ratio: float = MAX_LENGTH_RATIO,
        enable_language_detection: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据清洗器
        
        参数：
            min_zh_length: 中文最小长度
            max_zh_length: 中文最大长度
            min_en_length: 英文最小长度
            max_en_length: 英文最大长度
            min_length_ratio: 最小中英长度比例
            max_length_ratio: 最大中英长度比例
            enable_language_detection: 是否启用语言检测
            remove_urls: 是否移除 URL
            remove_emails: 是否移除邮箱
            normalize_unicode: 是否标准化 Unicode
            logger: 日志记录器
        """
        # 长度配置
        self.min_zh_length = min_zh_length
        self.max_zh_length = max_zh_length
        self.min_en_length = min_en_length
        self.max_en_length = max_en_length
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        
        # 功能开关
        self.enable_language_detection = enable_language_detection and LANGDETECT_AVAILABLE
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        
        # 日志记录器
        self.logger = logger or logging.getLogger(__name__)
        
        # 统计信息
        self.stats = CleaningStats()
        
        # 用于去重的哈希集合
        self._seen_hashes: set = set()
        
        self.logger.info(f"数据清洗器初始化完成")
        self.logger.info(f"  中文长度范围: {min_zh_length}-{max_zh_length}")
        self.logger.info(f"  英文长度范围: {min_en_length}-{max_en_length}")
        self.logger.info(f"  语言检测: {'启用' if self.enable_language_detection else '禁用'}")
    
    def _compute_hash(self, chinese: str, english: str) -> str:
        """
        计算样本对的唯一哈希值
        
        参数：
            chinese: 中文文本
            english: 英文文本
            
        返回：
            str: MD5 哈希值
        """
        # 合并中英文本并计算 MD5
        combined = f"{chinese.strip()}|||{english.strip()}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Unicode 标准化（NFKC）
        
        NFKC 标准化会：
        - 将全角字符转换为半角（如 ａ → a）
        - 统一等价字符（如 ﬁ → fi）
        - 规范化变音符号
        
        参数：
            text: 输入文本
            
        返回：
            str: 标准化后的文本
        """
        if not self.normalize_unicode:
            return text
        return unicodedata.normalize('NFKC', text)
    
    def _remove_html_tags(self, text: str) -> str:
        """
        移除 HTML 标签
        
        参数：
            text: 输入文本
            
        返回：
            str: 清洗后的文本
        """
        # 移除 HTML 标签
        text = HTML_TAG_PATTERN.sub('', text)
        # 处理 HTML 实体
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text
    
    def _remove_urls_and_emails(self, text: str) -> str:
        """
        移除 URL 和邮箱地址
        
        参数：
            text: 输入文本
            
        返回：
            str: 清洗后的文本
        """
        if self.remove_urls:
            text = URL_PATTERN.sub('', text)
        if self.remove_emails:
            text = EMAIL_PATTERN.sub('', text)
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """
        移除特殊控制字符
        
        参数：
            text: 输入文本
            
        返回：
            str: 清洗后的文本
        """
        # 移除控制字符
        text = SPECIAL_CHAR_PATTERN.sub('', text)
        # 规范化空白字符
        text = MULTI_SPACE_PATTERN.sub(' ', text)
        return text.strip()
    
    def _count_chinese_chars(self, text: str) -> int:
        """
        统计中文字符数量
        
        参数：
            text: 输入文本
            
        返回：
            int: 中文字符数量
        """
        return len(CHINESE_CHAR_PATTERN.findall(text))
    
    def _detect_language(self, text: str, expected_lang: str) -> bool:
        """
        检测文本语言是否符合预期
        
        参数：
            text: 输入文本
            expected_lang: 预期语言代码（'zh-cn' 或 'en'）
            
        返回：
            bool: 是否符合预期语言
        """
        if not self.enable_language_detection:
            return True
        
        try:
            detected = detect(text)
            # 中文检测结果可能是 'zh-cn', 'zh-tw', 'ko' 等
            if expected_lang == 'zh':
                return detected in ['zh-cn', 'zh-tw', 'zh', 'ko', 'ja']
            elif expected_lang == 'en':
                return detected == 'en'
            return True
        except Exception:
            # 检测失败时保守处理，保留样本
            return True
    
    def _check_length_ratio(self, chinese: str, english: str) -> bool:
        """
        检查中英文长度比例是否合理
        
        中文和英文的长度比例应该在合理范围内，
        异常比例可能表示翻译质量问题。
        
        参数：
            chinese: 中文文本
            english: 英文文本
            
        返回：
            bool: 比例是否合理
        """
        zh_len = len(chinese)
        en_len = len(english)
        
        if en_len == 0:
            return False
        
        ratio = zh_len / en_len
        return self.min_length_ratio <= ratio <= self.max_length_ratio
    
    def clean_text(self, text: str, is_chinese: bool = True) -> str:
        """
        清洗单条文本
        
        参数：
            text: 输入文本
            is_chinese: 是否为中文文本
            
        返回：
            str: 清洗后的文本
        """
        # 1. Unicode 标准化
        text = self._normalize_unicode(text)
        
        # 2. 移除 HTML 标签
        text = self._remove_html_tags(text)
        
        # 3. 移除 URL 和邮箱
        text = self._remove_urls_and_emails(text)
        
        # 4. 移除特殊字符
        text = self._remove_special_chars(text)
        
        # 5. 规范化空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def validate_sample(
        self,
        chinese: str,
        english: str
    ) -> Tuple[bool, str]:
        """
        验证样本对是否有效
        
        参数：
            chinese: 中文文本
            english: 英文文本
            
        返回：
            Tuple[bool, str]: (是否有效, 过滤原因)
        """
        # 检查空白
        if not chinese.strip() or not english.strip():
            return False, "empty"
        
        # 检查中文长度
        zh_len = len(chinese)
        if zh_len < self.min_zh_length or zh_len > self.max_zh_length:
            return False, "zh_length"
        
        # 检查英文长度
        en_len = len(english)
        if en_len < self.min_en_length or en_len > self.max_en_length:
            return False, "en_length"
        
        # 检查长度比例
        if not self._check_length_ratio(chinese, english):
            return False, "ratio"
        
        # 语言检测
        if self.enable_language_detection:
            # 检测中文
            zh_char_count = self._count_chinese_chars(chinese)
            if zh_char_count < self.min_zh_length * 0.5:
                return False, "zh_lang"
            
            # 检测英文
            if not self._detect_language(english, 'en'):
                return False, "en_lang"
        
        return True, "valid"
    
    def clean_sample(
        self,
        chinese: str,
        english: str,
        source: Optional[str] = None
    ) -> Optional[SamplePair]:
        """
        清洗单个样本对
        
        参数：
            chinese: 中文文本
            english: 英文文本
            source: 数据来源
            
        返回：
            Optional[SamplePair]: 清洗后的样本，无效则返回 None
        """
        # 清洗文本
        cleaned_zh = self.clean_text(chinese, is_chinese=True)
        cleaned_en = self.clean_text(english, is_chinese=False)
        
        # 验证样本
        is_valid, reason = self.validate_sample(cleaned_zh, cleaned_en)
        
        if not is_valid:
            # 更新统计信息
            if reason == "empty":
                self.stats.empty_filtered += 1
            elif reason in ["zh_length", "en_length"]:
                self.stats.length_filtered += 1
            elif reason == "ratio":
                self.stats.ratio_filtered += 1
            elif reason in ["zh_lang", "en_lang"]:
                self.stats.language_filtered += 1
            return None
        
        # 去重检查
        sample_hash = self._compute_hash(cleaned_zh, cleaned_en)
        if sample_hash in self._seen_hashes:
            self.stats.duplicate_removed += 1
            return None
        
        # 记录哈希
        self._seen_hashes.add(sample_hash)
        
        return SamplePair(
            chinese=cleaned_zh,
            english=cleaned_en,
            source=source
        )
    
    def clean_dataset(
        self,
        samples: List[Dict[str, str]],
        zh_key: str = "chinese",
        en_key: str = "english",
        show_progress: bool = True
    ) -> List[SamplePair]:
        """
        清洗整个数据集
        
        参数：
            samples: 原始样本列表，每个样本为字典
            zh_key: 中文字段名
            en_key: 英文字段名
            show_progress: 是否显示进度条
            
        返回：
            List[SamplePair]: 清洗后的样本列表
        """
        # 重置统计信息
        self.stats = CleaningStats()
        self.stats.total_samples = len(samples)
        self._seen_hashes.clear()
        
        cleaned_samples = []
        
        # 设置进度条
        iterator = tqdm(
            samples,
            desc="数据清洗",
            disable=not show_progress
        )
        
        for sample in iterator:
            # 获取中英文本
            chinese = sample.get(zh_key, "")
            english = sample.get(en_key, "")
            source = sample.get("source", None)
            
            # 清洗样本
            cleaned = self.clean_sample(chinese, english, source)
            
            if cleaned is not None:
                cleaned_samples.append(cleaned)
        
        # 更新最终统计
        self.stats.final_samples = len(cleaned_samples)
        
        self.logger.info(str(self.stats))
        
        return cleaned_samples
    
    def clean_dataset_streaming(
        self,
        data_path: str | Path,
        output_path: str | Path,
        zh_key: str = "chinese",
        en_key: str = "english",
        file_format: str = "jsonl"
    ) -> CleaningStats:
        """
        流式清洗大规模数据集（节省内存）
        
        参数：
            data_path: 输入数据路径（目录或文件）
            output_path: 输出文件路径
            zh_key: 中文字段名
            en_key: 英文字段名
            file_format: 文件格式（jsonl 或 json）
            
        返回：
            CleaningStats: 清洗统计信息
        """
        data_path = Path(data_path)
        output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 重置统计信息
        self.stats = CleaningStats()
        self._seen_hashes.clear()
        
        self.logger.info(f"开始流式清洗: {data_path}")
        
        # 获取所有数据文件
        if data_path.is_dir():
            # 同时查找 .json 和 .jsonl 文件
            data_files = list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json"))
        else:
            data_files = [data_path]
        
        # 根据文件扩展名确定格式
        if data_files:
            first_file = data_files[0]
            if first_file.suffix == ".jsonl":
                file_format = "jsonl"
            else:
                file_format = "json"
        
        self.logger.info(f"发现 {len(data_files)} 个数据文件")
        
        # 打开输出文件
        with open(output_path, 'w', encoding='utf-8') as out_file:
            # 遍历所有文件
            for file_path in tqdm(data_files, desc="处理文件"):
                self._process_single_file(
                    file_path=file_path,
                    out_file=out_file,
                    zh_key=zh_key,
                    en_key=en_key,
                    file_format=file_format
                )
        
        self.logger.info(str(self.stats))
        return self.stats
    
    def _process_single_file(
        self,
        file_path: Path,
        out_file,
        zh_key: str,
        en_key: str,
        file_format: str
    ) -> None:
        """
        处理单个数据文件
        
        参数：
            file_path: 文件路径
            out_file: 输出文件句柄
            zh_key: 中文字段名
            en_key: 英文字段名
            file_format: 文件格式
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取文件开头判断实际格式
                first_chars = f.read(1024)
                f.seek(0)
                
                # 判断是JSON数组还是JSONL格式
                first_non_whitespace = ''
                for c in first_chars:
                    if c not in ' \t\n\r':
                        first_non_whitespace = c
                        break
                
                if first_non_whitespace == '[':
                    # JSON数组格式：整体加载
                    data = json.load(f)
                    if isinstance(data, list):
                        for sample in data:
                            self._process_sample(
                                sample, out_file, zh_key, en_key
                            )
                    elif isinstance(data, dict):
                        self._process_sample(
                            data, out_file, zh_key, en_key
                        )
                else:
                    # JSONL格式：逐行处理
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            self._process_sample(
                                sample, out_file, zh_key, en_key
                            )
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"处理文件失败 {file_path}: {e}")
    
    def _process_sample(
        self,
        sample: Dict,
        out_file,
        zh_key: str,
        en_key: str
    ) -> None:
        """
        处理单个样本并写入输出
        
        参数：
            sample: 样本字典
            out_file: 输出文件句柄
            zh_key: 中文字段名
            en_key: 英文字段名
        """
        self.stats.total_samples += 1
        
        # 获取文本
        chinese = sample.get(zh_key, "")
        english = sample.get(en_key, "")
        source = sample.get("source", None)
        
        # 清洗样本
        cleaned = self.clean_sample(chinese, english, source)
        
        if cleaned is not None:
            self.stats.final_samples += 1
            # 写入 JSONL 格式
            out_file.write(json.dumps(cleaned.to_dict(), ensure_ascii=False) + '\n')
    
    def reset(self) -> None:
        """
        重置清洗器状态
        
        清除哈希缓存和统计信息，用于处理新数据集
        """
        self._seen_hashes.clear()
        self.stats = CleaningStats()
        self.logger.info("清洗器状态已重置")


def load_translation2019zh(
    data_dir: str | Path,
    max_samples: Optional[int] = None,
    show_progress: bool = True
) -> Generator[Dict[str, str], None, None]:
    """
    加载 translation2019zh 数据集
    
    translation2019zh 数据集包含约 516 万条中英翻译对，
    每个文件为 JSON 格式，包含 chinese 和 english 字段。
    
    参数：
        data_dir: 数据目录路径
        max_samples: 最大加载样本数（用于测试）
        show_progress: 是否显示进度条
        
    返回：
        Generator: 样本生成器，每个样本为字典
        
    示例：
        >>> for sample in load_translation2019zh("mydata/translation2019zh"):
        ...     print(sample["chinese"], sample["english"])
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 查找所有 JSON 文件
    json_files = sorted(data_dir.glob("*.json"))
    
    if not json_files:
        # 尝试查找 JSONL 文件
        json_files = sorted(data_dir.glob("*.jsonl"))
    
    if not json_files:
        raise FileNotFoundError(f"未找到数据文件: {data_dir}")
    
    sample_count = 0
    
    # 遍历文件
    iterator = tqdm(json_files, desc="加载数据", disable=not show_progress)
    
    for file_path in iterator:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试作为 JSON 数组加载
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                            sample_count += 1
                            if max_samples and sample_count >= max_samples:
                                return
                    elif isinstance(data, dict):
                        yield data
                        sample_count += 1
                except json.JSONDecodeError:
                    # 可能是 JSONL 格式
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                                sample_count += 1
                                if max_samples and sample_count >= max_samples:
                                    return
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logging.warning(f"加载文件失败 {file_path}: {e}")
            continue


# ====================================
# 命令行接口
# ====================================

def main():
    """
    命令行入口函数
    
    使用方式：
        python cleaner.py --input mydata/translation2019zh --output outputs/cleaned_data.jsonl
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="中英翻译数据清洗工具"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入数据路径（目录或文件）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--min-zh-length",
        type=int,
        default=MIN_CHINESE_LENGTH,
        help=f"中文最小长度（默认: {MIN_CHINESE_LENGTH}）"
    )
    parser.add_argument(
        "--max-zh-length",
        type=int,
        default=MAX_CHINESE_LENGTH,
        help=f"中文最大长度（默认: {MAX_CHINESE_LENGTH}）"
    )
    parser.add_argument(
        "--min-en-length",
        type=int,
        default=MIN_ENGLISH_LENGTH,
        help=f"英文最小长度（默认: {MIN_ENGLISH_LENGTH}）"
    )
    parser.add_argument(
        "--max-en-length",
        type=int,
        default=MAX_ENGLISH_LENGTH,
        help=f"英文最大长度（默认: {MAX_ENGLISH_LENGTH}）"
    )
    parser.add_argument(
        "--no-lang-detect",
        action="store_true",
        help="禁用语言检测"
    )
    parser.add_argument(
        "--zh-key",
        type=str,
        default="chinese",
        help="中文字段名（默认: chinese）"
    )
    parser.add_argument(
        "--en-key",
        type=str,
        default="english",
        help="英文字段名（默认: english）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # 创建清洗器
    cleaner = DataCleaner(
        min_zh_length=args.min_zh_length,
        max_zh_length=args.max_zh_length,
        min_en_length=args.min_en_length,
        max_en_length=args.max_en_length,
        enable_language_detection=not args.no_lang_detect
    )
    
    # 执行流式清洗
    stats = cleaner.clean_dataset_streaming(
        data_path=args.input,
        output_path=args.output,
        zh_key=args.zh_key,
        en_key=args.en_key
    )
    
    print(stats)


if __name__ == "__main__":
    main()
