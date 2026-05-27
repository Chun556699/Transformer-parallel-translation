#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署脚本

功能说明：
    - 启动 FastAPI后端服务
    -启动 React前端服务
    - Docker容器部署（可选）
    - 服务健康检查

使用方法：
    python scripts/deploy.py [--mode MODE] [--backend-port PORT] [--frontend-port PORT]

参数说明：
    --mode          :模式 (dev:开发, prod:生产，默认: dev)
    --backend-port  :后端端口 (默认: 8000)
    --frontend-port :前端端口 (默认: 3000)
    --backend-only  :仅启动后端
    --frontend-only :仅启动前端
    --docker        : 使用 Docker部署
    --model-path    :模型路径 (默认: outputs/compressed)

作者：NMT翻系统
"""

import os
import sys
import argparse
import logging
import time
import subprocess
import signal
import threading
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python路
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmt.utils import setup_logger

# ============================================================================
#配置
# ============================================================================

# 默认模型路径（优先使用量化模型）
DEFAULT_MODEL_PATH = "outputs/quantized"

logger = setup_logger(__name__, level=logging.INFO)

# ============================================================================
#全局变量
# ============================================================================

processes: List[subprocess.Popen] = []
stop_event = threading.Event()

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NMT服务部署")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev",
                       help="部署模式")
    parser.add_argument("--backend-port", type=int, default=8000, help="后端端口")
    parser.add_argument("--frontend-port", type=int, default=3000, help="前端端口")
    parser.add_argument("--backend-only", action="store_true", help="仅启动后端")
    parser.add_argument("--frontend-only", action="store_true", help="仅启动前端")
    parser.add_argument("--docker", action="store_true", help="使用 Docker部署")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                       help="模型路径")
    
    args = parser.parse_args()
    
    print_header("NMT服务部署")
    
    #注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    #检查模型
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"警告:模型路径不存在: {model_path}")
        print("将使用默认预训练模型")
    
    #设置环境变量
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["PYTHONPATH"] = str(project_root)
    
    try:
        if args.docker:
            deploy_docker(args.backend_port, args.frontend_port)
        else:
            deploy_local(args)
    except Exception as e:
        logger.error(f"部署失败: {e}")
        sys.exit(1)
    finally:
        cleanup()

# ============================================================================
#部署函数
# ============================================================================

def deploy_local(args):
    """本地部署"""
    print("部署配置：")
    print(f"  -模式: {args.mode}")
    print(f"  -后端端口: {args.backend_port}")
    print(f"  -前端端口: {args.frontend_port}")
    print(f"  -模型路径: {args.model_path}")
    print()
    
    #检查端口占用
    if not args.frontend_only:
        if is_port_in_use(args.backend_port):
            print(f"端口 {args.backend_port}已被占用")
            if not handle_port_conflict(args.backend_port):
                sys.exit(1)
    
    if not args.backend_only:
        if is_port_in_use(args.frontend_port):
            print(f"端口 {args.frontend_port}已被占用")
            if not handle_port_conflict(args.frontend_port):
                sys.exit(1)
    
    #启动服务
    if not args.frontend_only:
        start_backend(args.backend_port, args.mode)
    
    if not args.backend_only:
        start_frontend(args.frontend_port, args.mode)
    
    #等待服务
    print_header("部署完成")
    print("\n服务访问地址：")
    if not args.frontend_only:
        print(f"  - 后端 API: http://localhost:{args.backend_port}")
        print(f"  - API文档: http://localhost:{args.backend_port}/docs")
    if not args.backend_only:
        print(f"  -前端界面: http://localhost:{args.frontend_port}")
    
    print("\n按 Ctrl+C所有服务\n")
    
    #保持运行
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass

def deploy_docker(backend_port: int, frontend_port: int):
    """Docker部署"""
    print_header("Docker部署")
    
    #检查 docker-compose.yml
    docker_compose_path = project_root / "docker-compose.yml"
    if not docker_compose_path.exists():
        print_step("生成 docker-compose.yml...")
        generate_docker_compose(backend_port, frontend_port)
    
    #启动容器
    print_step("启动 Docker容器...")
    cmd = ["docker-compose", "up", "-d", "--build"]
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError("Docker部署失败")
    
    print_step("Docker容器启动成功")
    print(f"\n服务信息:")
    print(f"  -后端: http://localhost:{backend_port}")
    print(f"  -前端: http://localhost:{frontend_port}")

# ============================================================================
#辅助函数
# ============================================================================

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_step(message):
    """打印步骤信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True

def handle_port_conflict(port: int) -> bool:
    """处理端口冲突"""
    try:
        choice = input("是否停止占用进程？(y/n): ").strip().lower()
        if choice == 'y':
            kill_process_on_port(port)
            time.sleep(2)
            return True
        else:
            print(f"端口 {port}，无法启动服务")
            return False
    except (EOFError, KeyboardInterrupt):
        return False

def kill_process_on_port(port: int):
    """杀端口上的进程"""
    import psutil
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                print(f"正在停止端口 {port} 上的进程 ({process.name()})...")
                process.terminate()
                process.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
            break

def start_backend(port: int, mode: str):
    """启动后端服务"""
    print_header("启动后端服务")
    
    backend_dir = project_root / "web" / "backend"
    
    if mode == "prod":
        print_step("生产模式启动 (Gunicorn + Uvicorn)")
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", str(port), "--workers", "4"
        ]
    else:
        print_step("开发模式启动 (Uvicorn + 热重载)")
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", str(port), "--reload"
        ]
    
    print(f"命令: {' '.join(cmd)}")
    print(f"工作目录: {backend_dir}")
    
    #启动后端进程
    process = subprocess.Popen(cmd, cwd=str(backend_dir))
    processes.append(process)
    
    #等待服务就绪
    if wait_for_service(f"http://localhost:{port}/api/health", "后端服务"):
        print(f"\n后端服务信息:")
        print(f"  -地址: http://localhost:{port}")
        print(f"  - API文档: http://localhost:{port}/docs")
        print(f"  -进程 ID: {process.pid}")

def start_frontend(port: int, mode: str):
    """启动前端服务"""
    print_header("启动前端服务")
    
    frontend_dir = project_root / "web" / "frontend"
    
    # 检查依赖
    if not (frontend_dir / "node_modules").exists():
        print_step("安装前端依赖...")
        cmd = "npm install"
        result = subprocess.run(cmd, cwd=str(frontend_dir), shell=True)
        if result.returncode != 0:
            raise RuntimeError("前端依赖安装失败")
    
    # 启动命令
    if mode == "prod":
        print_step("生产模式：构建并启动")
        # 构建
        build_cmd = "npm run build"
        result = subprocess.run(build_cmd, cwd=str(frontend_dir), shell=True)
        if result.returncode != 0:
            raise RuntimeError("前端构建失败")
        
        # 使用 serve 启动
        cmd = f"npx serve -s dist -l {port}"
    else:
        print_step("开发模式启动 (Vite)")
        cmd = f"npm run dev -- --port {port}"
    
    print(f"命令: {cmd}")
    
    # 启动前端进程
    process = subprocess.Popen(cmd, cwd=str(frontend_dir), shell=True)
    processes.append(process)
    
    # 等待服务就绪
    time.sleep(5)  # 给 Vite 一些启动时间
    
    print(f"\n前端服务信息:")
    print(f"  - 地址: http://localhost:{port}")
    print(f"  - 进程 ID: {process.pid}")

def wait_for_service(url: str, service_name: str, timeout: int = 120) -> bool:
    """等待服务启动"""
    import requests
    
    print(f"等待 {service_name}启动...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"{service_name}已就绪")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    print(f"{service_name}启动超时")
    return False

def generate_docker_compose(backend_port: int, frontend_port: int):
    """生成 docker-compose.yml"""
    content = f"""version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{backend_port}:8000"
    volumes:
      - ./outputs/compressed:/app/models:ro
    environment:
      - MODEL_PATH=/app/models
      - PYTHONPATH=/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  frontend:
    build:
      context: ./web/frontend
      dockerfile: Dockerfile
    ports:
      - "{frontend_port}:80"
    depends_on:
      - backend
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
"""
    
    docker_compose_path = project_root / "docker-compose.yml"
    with open(docker_compose_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print_step("docker-compose.yml已生成")

def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n收到停止信号，正在关闭服务...")
    stop_event.set()

def cleanup():
    """清理进程"""
    print("正在停止所有服务...")
    for process in processes:
        if process.poll() is None:  #进程仍在运行
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.warning(f"停止进程失败: {e}")
    
    processes.clear()
    print("服务已停止")

# ============================================================================
#入点
# ============================================================================

if __name__ == "__main__":
    main()