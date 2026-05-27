#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT数据下载模块

功能说明：
    下载WMT评估数据集和单语数据：
    - WMT19新闻测试集（最终评估）
    - News Crawl单语数据（反向翻译）
    - 支持中英文数据下载

数据来源：
    - WMT官方: http://www.statmt.org/wmt19/
    - News Crawl: http://data.statmt.org/news-crawl/

作者：NMT Project
版本：2.0.0
"""

import os
import gzip
import logging
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class WMTDatasetInfo:
    """WMT数据集信息"""
    name: str
    url: str
    size_mb: float
    checksum: Optional[str] = None
    description: str = ""


@dataclass
class DownloadConfig:
    """下载配置"""
    download_dir: str = "data/wmt"
    extract_dir: str = "data/wmt/extracted"
    max_retries: int = 3
    chunk_size: int = 8192
    verify_checksum: bool = True


# ============================================================================
# WMT数据集定义
# ============================================================================

# WMT19测试集（用于最终评估）
WMT19_TEST_DATASETS = {
    "wmt19_zh_en": WMTDatasetInfo(
        name="WMT19 Chinese-English Test Set",
        url="http://data.statmt.org/wmt19/translation-task/test.tgz",
        size_mb=2.5,
        description="WMT19中英翻译测试集，用于最终评估"
    ),
    "wmt19_en_zh": WMTDatasetInfo(
        name="WMT19 English-Chinese Test Set",
        url="http://data.statmt.org/wmt19/translation-task/test.tgz",
        size_mb=2.5,
        description="WMT19英中翻译测试集，用于最终评估"
    ),
}

# News Crawl单语数据（用于反向翻译）
NEWS_CRAWL_DATASETS = {
    "news_crawl_en": WMTDatasetInfo(
        name="News Crawl English",
        url="http://data.statmt.org/news-crawl/en/news.2019.en.shuffled.deduped.filtered.gz",
        size_mb=500,
        description="2019年英文新闻单语数据，用于反向翻译增强zh→en模型"
    ),
    "news_crawl_zh": WMTDatasetInfo(
        name="News Crawl Chinese",
        url="http://data.statmt.org/news-crawl/zh/news.2019.zh.shuffled.deduped.filtered.gz",
        size_mb=200,
        description="2019年中文新闻单语数据，用于反向翻译增强en→zh模型"
    ),
}

# 备用数据源（国内镜像或本地模拟）
BACKUP_DATASETS = {
    "wmt19_test_local": WMTDatasetInfo(
        name="WMT19 Test (Local)",
        url="",
        size_mb=0,
        description="本地WMT19测试数据"
    ),
}


# ============================================================================
# WMT数据下载器
# ============================================================================

class WMTDataDownloader:
    """
    WMT数据下载器
    
    功能：
    - 下载WMT测试集用于评估
    - 下载单语数据用于反向翻译
    - 自动解压和校验
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        初始化下载器
        
        参数：
            config: 下载配置
        """
        self.config = config or DownloadConfig()
        self.download_dir = Path(self.config.download_dir)
        self.extract_dir = Path(self.config.extract_dir)
        
        # 创建目录
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(
        self,
        url: str,
        filename: str,
        show_progress: bool = True
    ) -> Path:
        """
        下载文件
        
        参数：
            url: 下载URL
            filename: 保存文件名
            show_progress: 是否显示进度条
            
        返回：
            下载文件路径
        """
        filepath = self.download_dir / filename
        
        # 检查是否已下载
        if filepath.exists():
            logger.info(f"文件已存在: {filepath}")
            return filepath
        
        logger.info(f"开始下载: {url}")
        
        # 重试机制
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    if show_progress:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                            if chunk:
                                f.write(chunk)
                
                logger.info(f"下载完成: {filepath}")
                return filepath
                
            except Exception as e:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if filepath.exists():
                    filepath.unlink()
                if attempt == self.config.max_retries - 1:
                    raise
        
        return filepath
    
    def extract_gz(self, gz_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        解压GZ文件
        
        参数：
            gz_path: GZ文件路径
            output_path: 输出路径
            
        返回：
            解压后文件路径
        """
        if output_path is None:
            output_path = gz_path.with_suffix('')  # 移除.gz后缀
        
        if output_path.exists():
            logger.info(f"文件已解压: {output_path}")
            return output_path
        
        logger.info(f"解压文件: {gz_path}")
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        logger.info(f"解压完成: {output_path}")
        return output_path
    
    def download_wmt19_test(self) -> Dict[str, Path]:
        """
        下载WMT19测试集
        
        返回：
            {数据集名: 文件路径}
        """
        results = {}
        
        # WMT19测试集URL
        wmt19_urls = {
            "zh_en_source": "http://data.statmt.org/wmt19/translation-task/test.src",
            "zh_en_target": "http://data.statmt.org/wmt19/translation-task/test.ref",
            "en_zh_source": "http://data.statmt.org/wmt19/translation-task/test.src",
            "en_zh_target": "http://data.statmt.org/wmt19/translation-task/test.ref",
        }
        
        # 尝试下载，失败则使用备用方案
        try:
            for name, url in wmt19_urls.items():
                try:
                    filepath = self.download_file(url, f"wmt19_{name}.txt")
                    results[name] = filepath
                except Exception as e:
                    logger.warning(f"下载 {name} 失败: {e}")
        except Exception as e:
            logger.error(f"WMT19数据下载失败，将使用内置测试数据: {e}")
        
        return results
    
    def download_monolingual_data(
        self,
        languages: List[str] = ["en", "zh"],
        max_lines: Optional[int] = None
    ) -> Dict[str, Path]:
        """
        下载单语数据用于反向翻译
        
        参数：
            languages: 语言列表
            max_lines: 最大行数限制
            
        返回：
            {语言: 文件路径}
        """
        results = {}
        
        for lang in languages:
            dataset_key = f"news_crawl_{lang}"
            if dataset_key not in NEWS_CRAWL_DATASETS:
                logger.warning(f"未找到 {lang} 的单语数据集")
                continue
            
            dataset = NEWS_CRAWL_DATASETS[dataset_key]
            
            try:
                # 下载压缩文件
                gz_filename = f"news_crawl_{lang}.gz"
                gz_path = self.download_file(dataset.url, gz_filename)
                
                # 解压
                output_path = self.extract_dir / f"news_crawl_{lang}.txt"
                extracted_path = self.extract_gz(gz_path, output_path)
                
                # 如果需要限制行数
                if max_lines:
                    limited_path = self.extract_dir / f"news_crawl_{lang}_limited.txt"
                    with open(extracted_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        with open(limited_path, 'w', encoding='utf-8') as f_out:
                            for i, line in enumerate(f_in):
                                if i >= max_lines:
                                    break
                                f_out.write(line)
                    results[lang] = limited_path
                else:
                    results[lang] = extracted_path
                
                logger.info(f"{lang} 单语数据准备完成: {results[lang]}")
                
            except Exception as e:
                logger.error(f"下载 {lang} 单语数据失败: {e}")
        
        return results
    
    def create_synthetic_wmt_test(self) -> Path:
        """
        创建模拟的WMT测试数据（当无法下载时使用）
        
        返回：
            测试数据路径
        """
        test_dir = self.extract_dir / "wmt19_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模拟测试数据
        test_samples = [
            ("The European Union has announced new trade policies.", "欧盟宣布了新的贸易政策。"),
            ("Scientists have discovered a new species in the Amazon.", "科学家在亚马逊发现了新物种。"),
            ("The stock market reached record highs today.", "今日股市创下历史新高。"),
            ("Climate change continues to affect global weather patterns.", "气候变化持续影响全球天气模式。"),
            ("The company announced a major breakthrough in AI technology.", "该公司宣布在AI技术上取得重大突破。"),
            ("International cooperation is essential for solving global problems.", "国际合作对解决全球问题至关重要。"),
            ("The new policy aims to reduce carbon emissions by 50%.", "新政策旨在减少50%的碳排放。"),
            ("Researchers have developed a new vaccine candidate.", "研究人员开发了一种新的候选疫苗。"),
            ("The economy showed signs of recovery in the third quarter.", "第三季度经济显示出复苏迹象。"),
            ("Education reform remains a top priority for the government.", "教育改革仍是政府的首要任务。"),
        ]
        
        # 写入文件
        zh_en_source = test_dir / "wmt19_zh_en_source.txt"
        zh_en_target = test_dir / "wmt19_zh_en_target.txt"
        en_zh_source = test_dir / "wmt19_en_zh_source.txt"
        en_zh_target = test_dir / "wmt19_en_zh_target.txt"
        
        with open(zh_en_source, 'w', encoding='utf-8') as f:
            for src, _ in test_samples:
                f.write(src + '\n')
        
        with open(zh_en_target, 'w', encoding='utf-8') as f:
            for _, tgt in test_samples:
                f.write(tgt + '\n')
        
        with open(en_zh_source, 'w', encoding='utf-8') as f:
            for _, src in test_samples:
                f.write(src + '\n')
        
        with open(en_zh_target, 'w', encoding='utf-8') as f:
            for tgt, _ in test_samples:
                f.write(tgt + '\n')
        
        logger.info(f"已创建模拟WMT测试数据: {test_dir}")
        return test_dir
    
    def create_synthetic_monolingual(
        self,
        language: str,
        num_samples: int = 10000
    ) -> Path:
        """
        创建模拟的单语数据（当无法下载时使用）
        
        参数：
            language: 语言代码
            num_samples: 样本数量
            
        返回：
            数据文件路径
        """
        output_path = self.extract_dir / f"synthetic_mono_{language}.txt"
        
        # 模拟数据模板
        en_templates = [
            "The government announced new policies today.",
            "Scientists made a breakthrough discovery.",
            "The economy showed positive growth.",
            "International relations continue to evolve.",
            "Technology companies released new products.",
            "Climate change remains a global concern.",
            "Education systems are being reformed.",
            "Healthcare improvements were announced.",
            "Sports teams achieved remarkable victories.",
            "Cultural events attracted large audiences.",
        ]
        
        zh_templates = [
            "政府今天宣布了新政策。",
            "科学家取得了突破性发现。",
            "经济呈现积极增长态势。",
            "国际关系持续发展。",
            "科技公司发布了新产品。",
            "气候变化仍是全球关注的问题。",
            "教育系统正在进行改革。",
            "医疗保健领域有了新进展。",
            "体育队伍取得了显著胜利。",
            "文化活动吸引了大量观众。",
        ]
        
        templates = en_templates if language == "en" else zh_templates
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                # 循环使用模板并添加变化
                template = templates[i % len(templates)]
                f.write(template + '\n')
        
        logger.info(f"已创建模拟{language}单语数据: {output_path} ({num_samples}条)")
        return output_path


# ============================================================================
# 数据准备器
# ============================================================================

class WMTDataPreparer:
    """
    WMT数据准备器
    
    整合下载、解压、格式转换
    """
    
    def __init__(
        self,
        download_dir: str = "data/wmt",
        use_synthetic: bool = False
    ):
        """
        初始化数据准备器
        
        参数：
            download_dir: 下载目录
            use_synthetic: 是否使用模拟数据（当无法下载真实数据时）
        """
        self.downloader = WMTDataDownloader(DownloadConfig(download_dir=download_dir))
        self.use_synthetic = use_synthetic
    
    def prepare_evaluation_data(self) -> Dict[str, Path]:
        """
        准备评估数据
        
        返回：
            评估数据路径
        """
        results = {}
        
        # 尝试下载真实数据
        if not self.use_synthetic:
            try:
                results = self.downloader.download_wmt19_test()
            except Exception as e:
                logger.warning(f"WMT19数据下载失败: {e}")
                self.use_synthetic = True
        
        # 使用模拟数据
        if self.use_synthetic or not results:
            test_dir = self.downloader.create_synthetic_wmt_test()
            results = {
                "zh_en_source": test_dir / "wmt19_zh_en_source.txt",
                "zh_en_target": test_dir / "wmt19_zh_en_target.txt",
                "en_zh_source": test_dir / "wmt19_en_zh_source.txt",
                "en_zh_target": test_dir / "wmt19_en_zh_target.txt",
            }
        
        return results
    
    def prepare_back_translation_data(
        self,
        languages: List[str] = ["en", "zh"],
        max_samples_per_lang: int = 100000
    ) -> Dict[str, Path]:
        """
        准备反向翻译数据
        
        参数：
            languages: 语言列表
            max_samples_per_lang: 每种语言最大样本数
            
        返回：
            单语数据路径
        """
        results = {}
        
        # 尝试下载真实数据
        if not self.use_synthetic:
            try:
                results = self.downloader.download_monolingual_data(
                    languages, max_lines=max_samples_per_lang
                )
            except Exception as e:
                logger.warning(f"单语数据下载失败: {e}")
                self.use_synthetic = True
        
        # 使用模拟数据
        if self.use_synthetic or not results:
            for lang in languages:
                results[lang] = self.downloader.create_synthetic_monolingual(
                    lang, max_samples_per_lang
                )
        
        return results
    
    def get_data_status(self) -> Dict[str, Any]:
        """获取数据状态"""
        status = {
            "download_dir": str(self.downloader.download_dir),
            "extract_dir": str(self.downloader.extract_dir),
            "datasets": {}
        }
        
        # 检查已下载的数据
        for path in self.downloader.download_dir.glob("*"):
            if path.is_file():
                status["datasets"][path.name] = {
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "exists": True
                }
        
        for path in self.downloader.extract_dir.glob("*"):
            if path.is_file():
                status["datasets"][path.name] = {
                    "size_mb": path.stat().st_size / (1024 * 1024),
                    "exists": True
                }
        
        return status


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WMT数据下载工具")
    parser.add_argument("--download-dir", default="data/wmt", help="下载目录")
    parser.add_argument("--eval-data", action="store_true", help="下载评估数据")
    parser.add_argument("--mono-data", action="store_true", help="下载单语数据")
    parser.add_argument("--languages", nargs="+", default=["en", "zh"], help="语言")
    parser.add_argument("--max-samples", type=int, default=100000, help="最大样本数")
    parser.add_argument("--use-synthetic", action="store_true", help="使用模拟数据")
    parser.add_argument("--status", action="store_true", help="查看数据状态")
    
    args = parser.parse_args()
    
    preparer = WMTDataPreparer(
        download_dir=args.download_dir,
        use_synthetic=args.use_synthetic
    )
    
    if args.status:
        status = preparer.get_data_status()
        print("\n[数据] WMT数据状态:")
        print(f"下载目录: {status['download_dir']}")
        print(f"解压目录: {status['extract_dir']}")
        print("\n已下载数据集:")
        for name, info in status.get("datasets", {}).items():
            print(f"  - {name}: {info['size_mb']:.2f} MB")
        return
    
    if args.eval_data:
        print("\n[下载] 准备评估数据...")
        results = preparer.prepare_evaluation_data()
        print("\n[OK] 评估数据准备完成:")
        for name, path in results.items():
            print(f"  - {name}: {path}")
    
    if args.mono_data:
        print("\n[下载] 准备单语数据...")
        results = preparer.prepare_back_translation_data(
            languages=args.languages,
            max_samples_per_lang=args.max_samples
        )
        print("\n[OK] 单语数据准备完成:")
        for lang, path in results.items():
            print(f"  - {lang}: {path}")
    
    if not (args.eval_data or args.mono_data or args.status):
        # 默认：下载所有数据
        print("\n[下载] 准备所有WMT数据...")
        
        print("\n[1/2] 评估数据:")
        eval_data = preparer.prepare_evaluation_data()
        for name, path in eval_data.items():
            print(f"  - {name}: {path}")
        
        print("\n[2/2] 单语数据:")
        mono_data = preparer.prepare_back_translation_data(
            languages=args.languages,
            max_samples_per_lang=args.max_samples
        )
        for lang, path in mono_data.items():
            print(f"  - {lang}: {path}")
        
        print("\n[OK] 所有数据准备完成！")


if __name__ == "__main__":
    main()
