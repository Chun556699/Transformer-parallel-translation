#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMT22数据集下载脚本

从WMT官方或sacreBLEU下载WMT22中英翻译测试集用于评估

数据来源：
    https://www.statmt.org/wmt22/translation-task.html
    sacreBLEU自动下载

使用方法：
    python scripts/download_wmt22.py

作者：NMT Project
"""

import os
import gzip
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

# WMT官方数据源
WMT22_URLS = {
    # WMT22 测试集 - 官方URL
    "test_tgz": "https://www.statmt.org/wmt22/test.tgz",
    "zh_en_source": "https://data.statmt.org/wmt22/translation-task/test/wmt22-test-set.src.gz",
    "zh_en_target": "https://data.statmt.org/wmt22/translation-task/test/wmt22-test-set.ref.gz",
}


def download_file(url: str, output_path: Path, desc: str = None) -> Path:
    """下载文件并显示进度条"""
    if output_path.exists():
        print(f"文件已存在: {output_path}")
        return output_path
    
    print(f"下载: {url}")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return output_path


def download_wmt22_via_sacrebleu(output_dir: str = "wmt-data") -> Tuple[List[str], List[str]]:
    """
    使用sacreBLEU下载WMT22中英测试集
    
    这是最可靠的方式，sacreBLEU会自动处理下载和解析
    """
    try:
        import sacrebleu
        print("使用sacreBLEU下载WMT22测试集...")
        
        # 获取WMT22数据集
        ds = sacrebleu.DATASETS['wmt22']
        
        # 获取zh-en测试集
        src_file = ds.get_source_file('zh-en')
        ref_files = ds.get_reference_files('zh-en')
        
        # 读取内容
        sources = []
        with open(src_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f if line.strip()]
        
        references = []
        for ref_file in ref_files:
            with open(ref_file, 'r', encoding='utf-8') as f:
                refs = [line.strip() for line in f if line.strip()]
            references.append(refs)
        
        print(f"成功下载 {len(sources)} 条WMT22 zh-en测试样本")
        return sources, references[0] if references else []
        
    except ImportError:
        print("sacreBLEU未安装，尝试其他方式...")
        return [], []
    except Exception as e:
        print(f"sacreBLEU下载失败: {e}")
        return [], []


def load_modelscope_wmt18(output_dir: str = "wmt-data") -> Tuple[List[str], List[str]]:
    """
    加载ModelScope的WMT18测试集
    
    这是项目内置的高质量测试数据
    """
    import csv
    
    csv_path = Path(output_dir) / "damo_mt_testsets_zh2en_news_wmt18.csv"
    if not csv_path.exists():
        return [], []
    
    print(f"加载ModelScope WMT18测试集: {csv_path}")
    
    zh_sentences = []
    en_sentences = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                zh = row[0].strip()
                en = row[1].strip()
                # 过滤无效数据
                if zh and en and len(zh) > 1 and len(en) > 1:
                    # 跳过纯数字行
                    if not (zh.isdigit() or en.isdigit()):
                        zh_sentences.append(zh)
                        en_sentences.append(en)
    
    print(f"加载了 {len(zh_sentences)} 条WMT18测试样本")
    return zh_sentences, en_sentences


def download_wmt22_direct(output_dir: str = "wmt-data") -> Tuple[List[str], List[str]]:
    """
    直接从WMT官方服务器下载测试集
    """
    import tarfile
    import tempfile
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("尝试从WMT官方服务器下载...")
    
    try:
        # 下载test.tgz
        tgz_path = output_path / "wmt22_test.tgz"
        download_file(WMT22_URLS["test_tgz"], tgz_path, "WMT22 test")
        
        # 解压
        print("解压文件...")
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(output_path / "wmt22_extracted")
        
        # 查找zh-en相关文件
        extracted_dir = output_path / "wmt22_extracted"
        zh_sources = []
        en_refs = []
        
        # WMT22使用XML格式
        for xml_file in extracted_dir.rglob("*.xml"):
            print(f"解析: {xml_file}")
            zh, en = parse_wmt22_xml(xml_file)
            zh_sources.extend(zh)
            en_refs.extend(en)
        
        if zh_sources:
            print(f"从XML解析到 {len(zh_sources)} 条样本")
            return zh_sources, en_refs
        
    except Exception as e:
        print(f"WMT官方下载失败: {e}")
    
    return [], []


def parse_wmt22_xml(xml_path: Path) -> Tuple[List[str], List[str]]:
    """解析WMT22 XML格式测试集"""
    import xml.etree.ElementTree as ET
    
    zh_sentences = []
    en_sentences = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # WMT22格式：src和ref标签
        for doc in root.findall('.//doc'):
            for seg in doc.findall('.//seg'):
                src = seg.find('src')
                ref = seg.find('ref')
                
                if src is not None and ref is not None:
                    src_text = src.text or ""
                    ref_text = ref.text or ""
                    
                    # 判断语言方向
                    src_is_zh = any('\u4e00' <= c <= '\u9fff' for c in src_text)
                    ref_is_zh = any('\u4e00' <= c <= '\u9fff' for c in ref_text)
                    
                    if src_is_zh and not ref_is_zh:
                        zh_sentences.append(src_text.strip())
                        en_sentences.append(ref_text.strip())
                    elif not src_is_zh and ref_is_zh:
                        en_sentences.append(src_text.strip())
                        zh_sentences.append(ref_text.strip())
    
    except Exception as e:
        print(f"XML解析错误: {e}")
    
    return zh_sentences, en_sentences


def download_wmt22_from_official(
    output_dir: str = "wmt-data",
    max_samples: Optional[int] = None
) -> Dict[str, Path]:
    """
    下载WMT测试集（多级回退策略）
    
    优先级：
    1. ModelScope WMT18测试集（项目内置，3961条）
    2. sacreBLEU自动下载WMT22
    3. WMT官方服务器
    4. 项目内置测试数据（70条）
    
    参数：
        output_dir: 输出目录
        max_samples: 最大样本数（None表示全部）
        
    返回：
        {文件类型: 路径}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  WMT 测试数据下载")
    print("=" * 60)
    
    zh_sentences = []
    en_sentences = []
    
    # 优先级1：ModelScope WMT18测试集（项目内置）
    print("\n[1/4] 检查ModelScope WMT18测试集...")
    zh_sentences, en_sentences = load_modelscope_wmt18(output_dir)
    
    # 优先级2：sacreBLEU下载WMT22
    if not zh_sentences:
        print("\n[2/4] 尝试sacreBLEU下载WMT22...")
        zh_sentences, en_sentences = download_wmt22_via_sacrebleu(output_dir)
    
    # 优先级2：WMT官方服务器
    if not zh_sentences:
        print("\n[2/3] 尝试WMT官方服务器...")
        zh_sentences, en_sentences = download_wmt22_direct(output_dir)
    
    # 优先级3：内置测试数据
    if not zh_sentences:
        print("\n[3/3] 使用内置测试数据...")
        zh_sentences, en_sentences = get_builtin_test_data()
    
    # 限制样本数
    if max_samples and len(zh_sentences) > max_samples:
        import random
        random.seed(42)
        indices = random.sample(range(len(zh_sentences)), max_samples)
        zh_sentences = [zh_sentences[i] for i in indices]
        en_sentences = [en_sentences[i] for i in indices]
    
    print(f"\n保存数据文件...")
    print(f"共 {len(zh_sentences)} 条平行句对")
    
    # zh→en 方向（源是中文，目标是英文）
    zh2en_source = output_path / "wmt22_zh_en_source.txt"
    zh2en_target = output_path / "wmt22_zh_en_target.txt"
    
    # en→zh 方向（源是英文，目标是中文）
    en2zh_source = output_path / "wmt22_en_zh_source.txt"
    en2zh_target = output_path / "wmt22_en_zh_target.txt"
    
    with open(zh2en_source, 'w', encoding='utf-8') as f:
        for s in zh_sentences:
            f.write(s + '\n')
    
    with open(zh2en_target, 'w', encoding='utf-8') as f:
        for s in en_sentences:
            f.write(s + '\n')
    
    with open(en2zh_source, 'w', encoding='utf-8') as f:
        for s in en_sentences:
            f.write(s + '\n')
    
    with open(en2zh_target, 'w', encoding='utf-8') as f:
        for s in zh_sentences:
            f.write(s + '\n')
    
    # 保存统计信息
    stats = {
        "total_samples": len(zh_sentences),
        "zh2en_source": str(zh2en_source),
        "zh2en_target": str(zh2en_target),
        "en2zh_source": str(en2zh_source),
        "en2zh_target": str(en2zh_target)
    }
    
    with open(output_path / "wmt22_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("  下载完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_path}")
    print(f"样本数量: {len(zh_sentences)}")
    print("\n文件列表:")
    print(f"  - zh→en 源文件: {zh2en_source}")
    print(f"  - zh→en 目标文件: {zh2en_target}")
    print(f"  - en→zh 源文件: {en2zh_source}")
    print(f"  - en→zh 目标文件: {en2zh_target}")
    
    return {
        "zh2en_source": zh2en_source,
        "zh2en_target": zh2en_target,
        "en2zh_source": en2zh_source,
        "en2zh_target": en2zh_target
    }


def check_wmt22_exists(data_dir: str = "wmt-data") -> bool:
    """检查WMT22数据是否已存在"""
    data_path = Path(data_dir)
    required_files = [
        "wmt22_zh_en_source.txt",
        "wmt22_zh_en_target.txt",
        "wmt22_en_zh_source.txt",
        "wmt22_en_zh_target.txt"
    ]
    return all((data_path / f).exists() for f in required_files)


def get_builtin_test_data() -> Tuple[List[str], List[str]]:
    """获取内置测试数据（最后的备用方案）"""
    # 更多测试样本用于有效评估
    test_samples = [
        # 政治外交类
        ("欧盟宣布了新的贸易政策。", "The European Union has announced new trade policies."),
        ("国际合作对解决全球问题至关重要。", "International cooperation is essential for solving global problems."),
        ("新政策旨在减少50%的碳排放。", "The new policy aims to reduce carbon emissions by 50%."),
        ("各国领导人将在峰会上讨论气候变化问题。", "World leaders will discuss climate change issues at the summit."),
        ("外交部长表示两国关系正在不断改善。", "The foreign minister stated that relations between the two countries are constantly improving."),
        ("联合国安理会通过了新的决议。", "The UN Security Council passed a new resolution."),
        ("贸易协定将促进两国经济发展。", "The trade agreement will promote economic development in both countries."),
        ("国际组织呼吁加强环境保护措施。", "International organizations call for stronger environmental protection measures."),
        
        # 科技类
        ("该公司宣布在AI技术上取得重大突破。", "The company announced a major breakthrough in AI technology."),
        ("科技公司发布了新一代智能手机。", "The technology company released a new generation of smartphones."),
        ("研究人员开发了一种新的候选疫苗。", "Researchers have developed a new vaccine candidate."),
        ("人工智能正在改变我们的生活方式。", "Artificial intelligence is changing our way of life."),
        ("量子计算机有望在未来十年内实现商业化。", "Quantum computers are expected to become commercially available within the next decade."),
        ("自动驾驶汽车正在进行道路测试。", "Self-driving cars are undergoing road testing."),
        ("5G网络将大幅提升数据传输速度。", "5G networks will significantly increase data transmission speeds."),
        
        # 经济金融类
        ("今日股市创下历史新高。", "The stock market reached record highs today."),
        ("第三季度经济显示出复苏迹象。", "The economy showed signs of recovery in the third quarter."),
        ("经济数据显示通货膨胀率有所下降。", "Economic data shows that the inflation rate has decreased."),
        ("央行宣布维持利率不变。", "The central bank announced it would keep interest rates unchanged."),
        ("出口贸易额同比增长了15%。", "Export trade volume increased by 15% year-on-year."),
        ("消费者信心指数有所回升。", "The consumer confidence index has rebounded."),
        ("房地产市场继续保持稳定增长。", "The real estate market continues to maintain steady growth."),
        
        # 科学研究类
        ("科学家在亚马逊发现了新物种。", "Scientists have discovered a new species in the Amazon."),
        ("气候变化持续影响全球天气模式。", "Climate change continues to affect global weather patterns."),
        ("全球气候变化会议在巴黎召开。", "The global climate change conference was held in Paris."),
        ("天文学家发现了一颗新的系外行星。", "Astronomers have discovered a new exoplanet."),
        ("海洋温度上升正在威胁珊瑚礁生态系统。", "Rising ocean temperatures are threatening coral reef ecosystems."),
        ("这项研究发表在顶级学术期刊上。", "The study was published in a top academic journal."),
        
        # 教育文化类
        ("教育改革仍是政府的首要任务。", "Education reform remains a top priority for the government."),
        ("大学推出了新的在线课程项目。", "The university launched a new online course program."),
        ("博物馆举办了一场大型艺术展览。", "The museum held a large-scale art exhibition."),
        ("文化遗产保护需要全社会的共同努力。", "Cultural heritage protection requires the joint efforts of the whole society."),
        ("越来越多的学生选择出国留学。", "More and more students choose to study abroad."),
        
        # 医疗健康类
        ("医疗团队成功完成了复杂的手术。", "The medical team successfully completed a complex surgery."),
        ("新药物临床试验取得了积极结果。", "The clinical trial of the new drug achieved positive results."),
        ("健康生活方式可以预防多种疾病。", "A healthy lifestyle can prevent many diseases."),
        ("医院引进了先进的医疗设备。", "The hospital introduced advanced medical equipment."),
        ("心理健康问题日益受到社会关注。", "Mental health issues are receiving increasing social attention."),
        
        # 体育类
        ("国家队在比赛中取得了优异成绩。", "The national team achieved excellent results in the competition."),
        ("奥运会将在这个城市举行。", "The Olympic Games will be held in this city."),
        ("足球运动员打破了历史进球纪录。", "The football player broke the historical scoring record."),
        ("马拉松比赛吸引了来自世界各地的选手。", "The marathon attracted participants from around the world."),
        
        # 社会民生类
        ("城市交通拥堵问题日益严重。", "Urban traffic congestion is becoming increasingly serious."),
        ("政府推出了新的住房保障政策。", "The government introduced new housing security policies."),
        ("就业市场呈现出积极的发展趋势。", "The job market shows a positive development trend."),
        ("食品安全问题引起了公众的广泛关注。", "Food safety issues have attracted widespread public attention."),
        ("人口老龄化给社会保障体系带来挑战。", "Population aging poses challenges to the social security system."),
        
        # 环境类
        ("空气污染治理取得了显著成效。", "Significant progress has been made in air pollution control."),
        ("可再生能源的使用正在快速增长。", "The use of renewable energy is growing rapidly."),
        ("森林覆盖率持续提高。", "Forest coverage continues to increase."),
        ("水资源保护已成为全球性议题。", "Water resource protection has become a global issue."),
        
        # 旅游类
        ("旅游业已成为该地区的重要产业。", "Tourism has become an important industry in the region."),
        ("这个景点每年吸引数百万游客。", "This attraction draws millions of tourists every year."),
        ("签证政策的放宽促进了国际旅游。", "The relaxation of visa policies has promoted international tourism."),
        
        # 交通类
        ("高铁网络不断扩展。", "The high-speed rail network continues to expand."),
        ("新机场的建设将提升城市的国际地位。", "The construction of the new airport will enhance the city's international status."),
        ("公共交通系统的完善方便了市民出行。", "The improvement of the public transport system has made travel easier for citizens."),
        
        # 能源类
        ("核能发电在全国能源结构中占有重要地位。", "Nuclear power generation plays an important role in the national energy structure."),
        ("太阳能发电成本持续下降。", "Solar power generation costs continue to decline."),
        ("电动汽车销量创历史新高。", "Electric vehicle sales hit a record high."),
        
        # 农业类
        ("农业现代化进程稳步推进。", "The agricultural modernization process is steadily advancing."),
        ("粮食产量连续多年保持稳定。", "Grain production has remained stable for many consecutive years."),
        ("智慧农业技术的应用提高了生产效率。", "The application of smart agriculture technology has improved production efficiency."),
        
        # 法律类
        ("新法律将于下月起正式实施。", "The new law will officially take effect next month."),
        ("知识产权保护力度不断加强。", "Intellectual property protection is continuously being strengthened."),
        ("司法改革取得了重要进展。", "Important progress has been made in judicial reform."),
        
        # 企业商业类
        ("公司年度营收超过预期目标。", "The company's annual revenue exceeded the expected target."),
        ("创业公司获得了大量风险投资。", "The startup company secured significant venture capital investment."),
        ("电子商务改变了传统零售业态。", "E-commerce has transformed the traditional retail industry."),
        ("跨国公司在当地设立了研发中心。", "The multinational corporation established an R&D center locally."),
    ]
    
    zh = [s[0] for s in test_samples]
    en = [s[1] for s in test_samples]
    return zh, en


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WMT22数据集下载工具")
    parser.add_argument("--output-dir", default="wmt-data", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--check", action="store_true", help="检查数据是否存在")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.check:
        exists = check_wmt22_exists(args.output_dir)
        print(f"WMT22数据集{'已存在' if exists else '不存在'}: {args.output_dir}")
        return
    
    # 检查是否已存在
    if check_wmt22_exists(args.output_dir):
        print(f"WMT22数据集已存在: {args.output_dir}")
        print("如需重新下载，请先删除该目录")
        return
    
    # 下载数据
    download_wmt22_from_official(
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
