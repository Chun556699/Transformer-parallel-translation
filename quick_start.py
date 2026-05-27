#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMT系统快速启动脚本

功能说明：
    一键式操作界面，整合所有核心功能：
    - 数据准备（清洗、高质量抽样）
    -模型训练（中英双向）
    -模型压缩（剪枝、量化）
    -模型评估（多指标评测）
    - 服务部署（前后端启动）
    - 系统状态监控

使用方法：
    python quick_start.py

作者：NMT Project
"""

import os
import sys
import argparse
import logging
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

#导入，避免循环依赖
def setup_logger(name, level=logging.INFO):
    """设置日志"""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ============================================================================
# 配置和常量
# ============================================================================

logger = setup_logger(__name__, level=logging.INFO)

# 默认路径配置
DEFAULT_PATHS = {
    'data_input': 'mydata/translation2019zh',
    'data_output': 'outputs/data',
    'model_output': 'outputs/models',
    'compressed_output': 'outputs/compressed',
    'evaluation_output': 'outputs/evaluation'
}

#系统状态文件
STATUS_FILE = project_root / 'outputs' / 'system_status.json'

# ============================================================================
#系统状态管理
# ============================================================================

class SystemStatus:
    """系统状态管理器"""
    
    def __init__(self):
        self.status_file = STATUS_FILE
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_status()
    
    def _load_status(self) -> Dict[str, Any]:
        """加载状态文件"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"状态文件加载失败: {e}")
        return {'steps': {}}
    
    def _save_status(self):
        """保存状态文件"""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"状态文件保存失败: {e}")
    
    def update_step(self, step: str, status: str, details: Optional[Dict] = None):
        """更新步骤状态"""
        if 'steps' not in self.data:
            self.data['steps'] = {}
        
        self.data['steps'][step] = {
            'status': status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'details': details or {}
        }
        self._save_status()
    
    def get_step_status(self, step: str) -> Optional[Dict]:
        """获取步骤状态"""
        return self.data.get('steps', {}).get(step)
    
    def is_step_completed(self, step: str) -> bool:
        """检查步骤是否已完成"""
        step_data = self.get_step_status(step)
        return step_data and step_data.get('status') == 'completed'

# ============================================================================
#快操作操作类
# ============================================================================

class QuickNMT:
    """NMT快速操作类"""
    
    def __init__(self):
        self.status = SystemStatus()
        self.project_root = project_root
    
    def run_command(self, cmd: List[str], cwd: Optional[str] = None, 
                   capture_output: bool = False) -> subprocess.CompletedProcess:
        """运行命令"""
        if cwd is None:
            cwd = str(self.project_root)
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        if capture_output:
            return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        else:
            return subprocess.run(cmd, cwd=cwd)
    
    def check_environment(self) -> bool:
        """检查运行环境"""
        print("[检查] 检查运行环境...")
            
        # 检查Python版本
        python_version = sys.version_info
        if python_version < (3, 10):
            print("[错误] Python版本过低，需要3.10+")
            return False
        print(f"[OK] Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
        # 检查PyTorch
        try:
            import torch
            print(f"[OK] PyTorch版本: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"[OK] CUDA可用: {torch.cuda.get_device_name(0)}")
            else:
                print("[警告] CUDA不可用，将使用CPU模式")
        except ImportError:
            print("[错误] PyTorch未安装")
            return False
            
        # 检查Transformers
        try:
            import transformers
            print(f"[OK] Transformers版本: {transformers.__version__}")
        except ImportError:
            print("[错误] Transformers未安装")
            return False
            
        # 检查数据目录
        data_path = Path(DEFAULT_PATHS['data_input'])
        if not data_path.exists():
            print(f"[警告] 数据目录不存在: {data_path}")
            print("   请确保数据文件已准备就绪")
        else:
            print(f"[OK] 数据目录: {data_path}")
            
        # 检查Helsinki-NLP模型
        zh_en_model = Path("models/Helsinki-NLP-opus-mt-zh-en")
        en_zh_model = Path("models/Helsinki-NLP-opus-mt-en-zh")
        if zh_en_model.exists() and en_zh_model.exists():
            print(f"[OK] Helsinki-NLP模型: 已就绪")
        else:
            print(f"[警告] Helsinki-NLP模型不完整，请检查models目录")
            
        print("[OK] 环境检查完成\n")
        return True
        
    def download_wmt_data(self, use_synthetic: bool = False) -> bool:
        """下载WMT数据（评估集 + 反向翻译单语数据）"""
        step_name = "wmt_data_download"
            
        if self.status.is_step_completed(step_name):
            print("[OK] WMT数据已下载，跳过")
            return True
            
        print("[下载] 下载WMT数据...")
        start_time = time.time()
            
        cmd = [
            sys.executable, "scripts/download_wmt_data.py",
            "--download-dir", "data/wmt",
            "--eval-data",
            "--mono-data",
            "--languages", "en", "zh",
            "--max-samples", "100000"
        ]
            
        if use_synthetic:
            cmd.append("--use-synthetic")
            
        try:
            result = self.run_command(cmd)
            if result.returncode == 0:
                elapsed = time.time() - start_time
                self.status.update_step(
                    step_name,
                    'completed',
                    {'elapsed_time': elapsed, 'use_synthetic': use_synthetic}
                )
                print(f"[OK] WMT数据下载完成 (耗时: {elapsed:.1f}秒)")
                return True
            else:
                print("[错误] WMT数据下载失败")
                return False
        except Exception as e:
            print(f"[错误] WMT数据下载异常: {e}")
            return False
    
    def quick_data_prep(self, use_sampling: bool = True) -> bool:
        """快速数据准备"""
        step_name = "data_preparation"
        
        if self.status.is_step_completed(step_name):
            print("[OK] 数据准备已完成，跳过")
            return True
        
        print("[启动] 开始数据准备...")
        total_start = time.time()
        
        try:
            # 步骤1: 高质量数据抽样
            if use_sampling:
                print("[数据] 使用高质量数据抽样 (30万样本)...")
                sample_cmd = [
                    sys.executable, "-m", "nmt.data.high_quality_sampler",
                    "--input", "mydata/translation2019zh/translation2019zh_train.json",
                    "--output", "outputs/data/sampled/sampled_data.jsonl",
                    "--target-samples", "300000",
                    "--quality-threshold", "0.75",
                    "--labse-model", "sentence-transformers/LaBSE"
                ]
                result = self.run_command(sample_cmd)
                if result.returncode != 0:
                    print("[错误] 数据抽样失败")
                    return False
            
            # 步骤2: 构建数据集
            print("[数据] 构建训练/验证/测试数据集...")
            dataset_cmd = [
                sys.executable, "-m", "nmt.data.dataset",
                "--input", "outputs/data/sampled/sampled_data.jsonl",
                "--output", "outputs/data/dataset",
                "--split"
            ]
            result = self.run_command(dataset_cmd)
            if result.returncode != 0:
                print("[错误] 数据集构建失败")
                return False
            
            elapsed = time.time() - total_start
            self.status.update_step(
                step_name, 
                'completed', 
                {'elapsed_time': elapsed, 'sampling': use_sampling}
            )
            print(f"[OK] 数据准备完成 (耗时: {elapsed:.1f}秒)")
            return True
            
        except Exception as e:
            print(f"[错误] 数据准备异常: {e}")
            return False
    
    def quick_train(self, direction: str = "both", epochs: int = 5) -> bool:
        """快速模型训练"""
        start_time = time.time()
        
        # 训练方向列表
        if direction == "both":
            directions = ["zh2en", "en2zh"]
        else:
            directions = [direction]
        
        success_all = True
        
        for d in directions:
            step_name = f"train_{d}"
            
            if self.status.is_step_completed(step_name):
                print(f"[OK] {d}方向训练已完成，跳过")
                continue
            
            print(f"[启动] 开始{d}方向模型训练...")
            
            # 选择正确的模型路径
            if d == "zh2en":
                model_path = "models/Helsinki-NLP-opus-mt-zh-en"
                output_dir = "outputs/models/zh2en"
            else:
                model_path = "models/Helsinki-NLP-opus-mt-en-zh"
                output_dir = "outputs/models/en2zh"
            
            # 使用正确的训练命令
            cmd = [
                sys.executable, "-m", "nmt.training.trainer",
                "--direction", d,
                "--model", model_path,
                "--train-data", "outputs/data/dataset/train.jsonl",
                "--output", output_dir,
                "--config", "configs/training_config.yaml",
                "--eval-data", "outputs/data/dataset/val.jsonl"
            ]
            
            try:
                result = self.run_command(cmd)
                if result.returncode == 0:
                    elapsed = time.time() - start_time
                    self.status.update_step(
                        step_name,
                        'completed',
                        {'elapsed_time': elapsed, 'epochs': epochs}
                    )
                    print(f"[OK] {d}方向训练完成 (耗时: {elapsed:.1f}秒)")
                else:
                    print(f"[错误] {d}方向训练失败")
                    success_all = False
            except Exception as e:
                print(f"[错误] {d}方向训练异常: {e}")
                success_all = False
        
        return success_all
    
    def quick_compress(self, quantize_mode: str = "int8") -> bool:
        """快速模型压缩"""
        step_name = "model_compression"
        
        if self.status.is_step_completed(step_name):
            print("[OK]模型压缩已完成，跳过")
            return True
        
        print("[启动] 开始模型压缩...")
        start_time = time.time()
        
        cmd = [
            sys.executable, "scripts/compress.py",
            "--quantize", quantize_mode,
            "--output-dir", DEFAULT_PATHS['compressed_output']
        ]
        
        try:
            result = self.run_command(cmd)
            if result.returncode == 0:
                elapsed = time.time() - start_time
                self.status.update_step(
                    step_name,
                    'completed',
                    {'elapsed_time': elapsed, 'quantize_mode': quantize_mode}
                )
                print(f"[OK]模型压缩完成 (耗时: {elapsed:.1f}秒)")
                return True
            else:
                print("[错误]模型压缩失败")
                return False
        except Exception as e:
            print(f"[错误] 模型压缩异常: {e}")
            return False
    
    def quick_evaluate(self, direction: str = "both") -> bool:
        """快速模型评估"""
        step_name = "model_evaluation"
        
        if self.status.is_step_completed(step_name):
            print("[OK] 模型评估已完成，跳过")
            return True
        
        print("[启动] 开始模型评估...")
        start_time = time.time()
        
        # 评估双向模型
        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--direction", direction,
            "--max-samples", "1000"
        ]
        
        try:
            result = self.run_command(cmd)
            if result.returncode == 0:
                elapsed = time.time() - start_time
                self.status.update_step(
                    step_name,
                    'completed',
                    {'elapsed_time': elapsed}
                )
                print(f"[OK] 模型评估完成 (耗时: {elapsed:.1f}秒)")
                return True
            else:
                print("[错误] 模型评估失败")
                return False
        except Exception as e:
            print(f"[错误] 模型评估异常: {e}")
            return False
    
    def quick_deploy(self, mode: str = "dev") -> bool:
        """快速服务部署"""
        print(f"[启动]启动{mode}模式服务...")
        
        cmd = [
            sys.executable, "scripts/deploy.py",
            "--mode", mode
        ]
        
        try:
            print(f"启动命令: {' '.join(cmd)}")
            print("[提示]按 Ctrl+C停服务")
            print("[服务] 前端访问: http://localhost:3000")
            print("[服务] API文档: http://localhost:8000/docs")
            print()
            
            result = self.run_command(cmd)
            if result.returncode == 0:
                print("[OK] 服务已停止")
                return True
            else:
                print("[错误] 服务启动失败")
                return False
        except KeyboardInterrupt:
            print("\n[停止] 服务已手动停止")
            return True
        except Exception as e:
            print(f"[错误] 服务异常: {e}")
            return False
    
    def show_status(self):
        """显示系统状态"""
        print("=" * 60)
        print("[数据] NMT系统状态")
        print("=" * 60)
        
        steps = self.status.data.get('steps', {})
        if not steps:
            print("[警告] 暂执行记录")
            return
        
        for step_name, step_info in steps.items():
            status = step_info.get('status', 'unknown')
            timestamp = step_info.get('timestamp', 'N/A')
            details = step_info.get('details', {})
            
            status_icon = "[OK]" if status == 'completed' else "[错误]"
            print(f"{status_icon} {step_name}")
            print(f"  状态: {status}")
            print(f"   时间: {timestamp}")
            if details:
                print(f"   详情: {details}")
            print()
    
    def quick_full_pipeline(self, skip_deploy: bool = False, use_synthetic_wmt: bool = False):
        """快速完整流程"""
        print("[启动] 启动完整NMT流程...")
        print("=" * 60)
        
        pipeline_steps = [
            ("环境检查", self.check_environment),
            ("数据准备(抽样+构建)", lambda: self.quick_data_prep(use_sampling=True)),
            ("模型训练(双向)", lambda: self.quick_train("both", epochs=5)),
            ("模型压缩", lambda: self.quick_compress("int8")),
            ("模型评估", lambda: self.quick_evaluate("both")),
        ]
        
        if not skip_deploy:
            pipeline_steps.append(("服务部署", lambda: self.quick_deploy("dev")))
        
        total_start = time.time()
        success_count = 0
        
        for step_name, step_func in pipeline_steps:
            print(f"\n[步骤] 步骤: {step_name}")
            print("-" * 40)
            
            try:
                if step_func():
                    success_count += 1
                else:
                    print(f"[错误] 步骤失败: {step_name}")
                    break
            except Exception as e:
                print(f"[错误] 步骤异常: {step_name} - {e}")
                break
        
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print(f"[完成] 完整流程完成")
        print(f"[OK] 成功步骤: {success_count}/{len(pipeline_steps)}")
        print(f"[时间] 总耗时: {total_elapsed:.1f}秒")
        print("=" * 60)

# ============================================================================
# 主函数和命令行接口
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NMT系统快速启动工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                      # 交互式菜单
  %(prog)s --status             # 查看系统状态
  %(prog)s --full               # 完整流程(双向训练)
  %(prog)s --data               # 数据准备(抽样+构建)
  %(prog)s --train --direction both  # 训练双向模型
  %(prog)s --evaluate           # 模型评估
  %(prog)s --deploy             # 部署服务
        """
    )
    
    # 操作选项
    parser.add_argument("--status", action="store_true", help="查看系统状态")
    parser.add_argument("--check", action="store_true", help="检查环境")
    parser.add_argument("--full", action="store_true", help="完整流程")
    parser.add_argument("--data", action="store_true", help="数据准备")
    parser.add_argument("--train", action="store_true", help="模型训练")
    parser.add_argument("--compress", action="store_true", help="模型压缩")
    parser.add_argument("--evaluate", action="store_true", help="模型评估")
    parser.add_argument("--deploy", action="store_true", help="服务部署")
    
    # 参数选项
    parser.add_argument("--direction", choices=["zh2en", "en2zh", "both"], 
                       default="both", help="训练方向")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--quantize", choices=["int8", "int4"], default="int8", 
                       help="量化精度")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev", 
                       help="部署模式")
    parser.add_argument("--skip-deploy", action="store_true", help="跳过部署")
    parser.add_argument("--no-sampling", action="store_true", help="不使用高质量抽样")
    
    args = parser.parse_args()
    
    # 初始化快速操作器
    nmt = QuickNMT()
    
    # 根据参数执行相应操作
    if args.status:
        nmt.show_status()
        return 0
    
    if args.check:
        success = nmt.check_environment()
        return 0 if success else 1
    
    if args.full:
        nmt.quick_full_pipeline(skip_deploy=args.skip_deploy)
        return 0
    
    # 单步操作
    if args.data:
        success = nmt.quick_data_prep(use_sampling=not args.no_sampling)
        return 0 if success else 1
    
    if args.train:
        success = nmt.quick_train(args.direction, args.epochs)
        return 0 if success else 1
    
    if args.compress:
        success = nmt.quick_compress(args.quantize)
        return 0 if success else 1
    
    if args.evaluate:
        success = nmt.quick_evaluate()
        return 0 if success else 1
    
    if args.deploy:
        success = nmt.quick_deploy(args.mode)
        return 0 if success else 1
    
    # 交互式菜单
    interactive_menu(nmt)
    return 0

def interactive_menu(nmt: QuickNMT):
    """交互式菜单"""
    while True:
        print("\n" + "=" * 60)
        print(" NMT系统快速启动菜单")
        print("=" * 60)
        print("1.  查看系统状态")
        print("2.  检查运行环境")
        print("3.  数据准备(抽样+构建)")
        print("4.  模型训练(双向)")
        print("5.  模型压缩")
        print("6.  模型评估")
        print("7.  服务部署")
        print("8.  完整流程")
        print("0.  退出")
        print("=" * 60)
        
        try:
            choice = input("请选择操作 (0-8): ").strip()
            
            if choice == "0":
                print("[退出] 再见！")
                break
            elif choice == "1":
                nmt.show_status()
            elif choice == "2":
                nmt.check_environment()
            elif choice == "3":
                use_sampling = input("使用高质量抽样? (y/n, 默认y): ").strip().lower() != 'n'
                nmt.quick_data_prep(use_sampling=use_sampling)
            elif choice == "4":
                direction = input("训练方向 (zh2en/en2zh/both, 默认both): ").strip()
                if not direction:
                    direction = "both"
                epochs = input("训练轮数 (默认5): ").strip()
                epochs = int(epochs) if epochs else 5
                nmt.quick_train(direction, epochs)
            elif choice == "5":
                quantize = input("量化精度 (int8/int4, 默认int8): ").strip()
                if not quantize:
                    quantize = "int8"
                nmt.quick_compress(quantize)
            elif choice == "6":
                direction = input("评估方向 (zh2en/en2zh/both, 默认both): ").strip()
                if not direction:
                    direction = "both"
                nmt.quick_evaluate(direction)
            elif choice == "7":
                mode = input("部署模式 (dev/prod, 默认dev): ").strip()
                if not mode:
                    mode = "dev"
                nmt.quick_deploy(mode)
            elif choice == "8":
                skip_deploy = input("跳过部署? (y/n, 默认n): ").strip().lower() == 'y'
                nmt.quick_full_pipeline(skip_deploy=skip_deploy)
            else:
                print("[错误] 无效选择")
                
        except KeyboardInterrupt:
            print("\n\n[退出] 再见！")
            break
        except Exception as e:
            print(f"[错误] 操作异常: {e}")

if __name__ == "__main__":
    exit(main())