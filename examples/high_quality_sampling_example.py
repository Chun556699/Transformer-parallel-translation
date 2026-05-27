#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量数据抽样使用示例

功能说明：
   如何从520万样本中抽取30万高质量样本，确保不会毁坏和永久性的遗忘 / Helsinki-NLP

使用方法：
    python examples/high_quality_sampling_example.py

作者：NMT Project
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.data import HighQualitySampler, sample_translation_dataset

def main():
    """主函数 -高质量数据抽样"""
    
    print("=" * 70)
    print("高质量数据抽样示例")
    print("=" * 70)
    print()
    
    #配置参数
    config = {
        'input_path': 'mydata/translation2019zh/translation2019zh_train.json',
        'output_path': 'outputs/data/high_quality_samples.jsonl',
        'target_samples': 300000,        #目样本数
        'quality_threshold': 0.75,       #高质量阈值（高于标准0.7）
        'labse_model': 'sentence-transformers/LaBSE',
        'device': 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'
    }
    
    print("配置参数：")
    print(f"  输入数据: {config['input_path']}")
    print(f"  输出路径: {config['output_path']}")
    print(f"  目标样本: {config['target_samples']:,}")
    print(f" 阈值: {config['quality_threshold']}")
    print(f" 计算设备: {config['device']}")
    print()
    
    #检查输入文件
    if not os.path.exists(config['input_path']):
        print("[错误]错误：输入数据文件不存在")
        print(f"   请确保 {config['input_path']}存在")
        print("   或修改 input_path 参数指向正确的数据文件")
        return 1
    
    print("[OK] 输入文件检查通过")
    print()
    
    #执行高质量抽样
    print("开始高质量数据抽样...")
    print("-" * 50)
    
    try:
        stats = sample_translation_dataset(
            input_path=config['input_path'],
            output_path=config['output_path'],
            target_samples=config['target_samples'],
            quality_threshold=config['quality_threshold'],
            labse_model=config['labse_model'],
            device=config['device']
        )
        
        print("-" * 50)
        print("[OK]抽完成！")
        print()
        
        # 输出统计信息
        print("统计摘要：")
        print(f"  输入样本总数: {stats.total_input:,}")
        print(f" 质过滤后: {stats.quality_filtered:,}")
        print(f"  最终高质量样本: {stats.final_samples:,}")
        print(f" 抽比例: {stats.final_samples/stats.total_input*100:.2f}%")
        print()
        
        print("质量分布：")
        print(f" 分布: {dict(stats.difficulty_distribution)}")
        print(f"  长度分布: {dict(stats.length_distribution)}")
        print(f" 分布: {dict(stats.domain_distribution)}")
        print()
        
        print("后续步骤：")
        print("1. 使用抽样后的数据进行模型训练")
        print("2.训脚本会自动使用高质量样本")
        print("3.保持 quality_threshold ≥ 0.75确保性能")
        print()
        
        print("输出文件：")
        print(f"  - 高质量样本: {config['output_path']}")
        print(f"  -统计信息: {config['output_path'].replace('.jsonl', '.stats.json')}")
        print()
        
        return 0
        
    except Exception as e:
        print("[错误]抽失败！")
        print(f"   错误信息: {str(e)}")
        print()
        print("常见问题排查：")
        print("1.检查输入文件路径是否正确")
        print("2.确保有足够磁盘空间")
        print("3.检查CUDA环境（如果使用GPU）")
        print("4.确保安装了 sentence-transformers")
        return 1

def advanced_example():
    """高级示例：自定义抽样策略"""
    
    print("=" * 70)
    print("高级抽样策略示例")
    print("=" * 70)
    print()
    
    # 创建自定义抽样器
    sampler = HighQualitySampler(
        target_samples=50000,      # 更少的样本
        quality_threshold=0.80,    # 更高的质量要求
        seed=12345                 #固定随机种子确保可重现
    )
    
    print("自定义参数：")
    print(f" 目样本数: {sampler.target_samples:,}")
    print(f" 质量阈值: {sampler.quality_threshold}")
    print(f" 随种子: {sampler.seed}")
    print()
    
    #可以在这里添加更多自定义逻辑
    # 例如：特定领域的数据筛选、自定义分层策略等

if __name__ == "__main__":
    print("选择运行模式：")
    print("1.基本抽样示例")
    print("2.高级抽样示例")
    print()
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "2":
        advanced_example()
        exit_code = 0
    else:
        exit_code = main()
    
    print("=" * 70)
    if exit_code == 0:
        print("[成功] 示例运行成功！")
    else:
        print("[警告]  示例运行失败，请检查错误信息")
    print("=" * 70)
    
    sys.exit(exit_code)