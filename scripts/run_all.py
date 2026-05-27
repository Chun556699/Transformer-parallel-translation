#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流程一键脚本

功能说明：
    - 执行完整的 NMT系统构建流程
    -包括数据准备、训练、压缩、评估、部署
    -支持断点续跑

使用方法：
    python scripts/run_all.py [--start-from STEP] [--stop-at STEP] [--skip-deploy]

参数说明：
    --start-from    : 从指定步骤开始 (data, train, compress, evaluate, deploy)
    --stop-at       : 在指定步骤停止
    --skip-deploy   :跳过部署步骤

作者：NMT翻系统
"""

import os
import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到 Python路
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.utils import setup_logger

# ============================================================================
#配置
# ============================================================================

logger = setup_logger(__name__, level=logging.INFO)

#步定义
STEPS = [
    {"name": "data", "display": "数据准备", "script": "data_prep.py"},
    {"name": "train", "display": "模型训练", "script": "train.py"},
    {"name": "compress", "display": "模型压缩", "script": "compress.py"},
    {"name": "evaluate", "display": "模型评估", "script": "evaluate.py"},
    {"name": "deploy", "display": "服务部署", "script": "deploy.py"}
]

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT完整构建流程")
    parser.add_argument("--start-from", choices=[s["name"] for s in STEPS],
                       default="data", help="从指定步骤开始")
    parser.add_argument("--stop-at", choices=[s["name"] for s in STEPS],
                       default="deploy", help="在指定步骤停止")
    parser.add_argument("--skip-deploy", action="store_true", help="跳过部署步骤")
    
    args = parser.parse_args()
    
    #计算执行范围
    start_idx = next(i for i, s in enumerate(STEPS) if s["name"] == args.start_from)
    stop_idx = next(i for i, s in enumerate(STEPS) if s["name"] == args.stop_at)
    
    if args.skip_deploy and args.stop_at == "deploy":
        stop_idx = next(i for i, s in enumerate(STEPS) if s["name"] == "evaluate")
    
    if start_idx > stop_idx:
        logger.error(f"StartFrom ({args.start_from}) 不能在 StopAt ({args.stop_at}) 之后")
        sys.exit(1)
    
    #执行流程
    total_start_time = time.time()
    
    print_header("NMT翻系统 - 完整构建流程")
    
    print("执行计划：")
    for i in range(start_idx, stop_idx + 1):
        step = STEPS[i]
        print(f"  [{i + 1}] {step['display']}")
    print()
    
    #执行步骤
    results = {}
    step_number = 1
    total_steps = stop_idx - start_idx + 1
    
    for i in range(start_idx, stop_idx + 1):
        step = STEPS[i]
        step_start_time = time.time()
        
        print_step_header(step_number, total_steps, step["display"])
        
        try:
            #执行脚本
            script_path = project_root / "scripts" / step["script"]
            cmd = [sys.executable, str(script_path)]
            
            print(f"执行: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                raise RuntimeError("脚本执行失败")
            
            elapsed = time.time() - step_start_time
            results[step["name"]] = {
                "status": "成功",
                "time": format_time(elapsed)
            }
            
            print(f"\n{step['display']}完成，耗时: {format_time(elapsed)}")
            
        except Exception as e:
            elapsed = time.time() - step_start_time
            results[step["name"]] = {
                "status": "失败",
                "time": format_time(elapsed),
                "error": str(e)
            }
            
            print(f"\n{step['display']}失: {e}")
            print("流程中断，请检查错误后重新运行")
            print(f"可使用 --start-from {step['name']} 从当前步骤继续")
            break
        
        step_number += 1
    
    #汇总报告
    total_elapsed = time.time() - total_start_time
    print_header("构建流程完成")
    
    print("执行结果：")
    print("-" * 50)
    
    for step in STEPS:
        if step["name"] in results:
            result = results[step["name"]]
            color = "\033[32m" if result["status"] == "成功" else "\033[31m"
            status_icon = "[OK]" if result["status"] == "成功" else "[FAIL]"
            reset = "\033[0m"
            
            print(f"  {step['display']}: {color}{status_icon}{reset} ({result['time']})")
    
    print("-" * 50)
    print(f"\n总耗时: {format_time(total_elapsed)}")
    
    #检查是否全部成功
    all_success = all(r["status"] == "成功" for r in results.values())
    
    if all_success:
        print("\n所有步骤执行成功！")
        print("\n服务访问地址：")
        print("  -前端界面: http://localhost:3000")
        print("  - API文档: http://localhost:8000/docs")
    else:
        print("\n部分步骤执行失败，请检查错误日志")

# ============================================================================
#辅助函数
# ============================================================================

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_step_header(step_number: int, total_steps: int, step_name: str):
    """打印步骤标题"""
    print("\n" + "-" * 70)
    print(f" 步骤 {step_number}/{total_steps} : {step_name}")
    print("-" * 70)

def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ============================================================================
#入点
# ============================================================================

if __name__ == "__main__":
    main()