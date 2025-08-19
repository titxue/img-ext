#!/usr/bin/env python3
"""
FastAPI服务启动脚本

提供便捷的方式启动图像特征提取API服务
支持不同的配置选项和环境设置
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'torchvision',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"错误: 缺少必要的依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_feature_extractor():
    """检查特征提取器是否可用"""
    try:
        from feature_extractor import FeatureExtractor
        # 尝试初始化特征提取器
        extractor = FeatureExtractor()
        print(f"✅ 特征提取器初始化成功，设备: {extractor.device}")
        return True
    except Exception as e:
        print(f"❌ 特征提取器初始化失败: {str(e)}")
        return False

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True, 
                log_level: str = "info", workers: int = 1):
    """启动FastAPI服务器"""
    import uvicorn
    
    print(f"正在启动图像特征提取API服务...")
    print(f"服务地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"ReDoc文档: http://{host}:{port}/redoc")
    print(f"健康检查: http://{host}:{port}/health")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            workers=workers if not reload else 1,  # reload模式下只能使用1个worker
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动服务时发生错误: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="图像特征提取API服务启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start_api_server.py                    # 使用默认配置启动
  python start_api_server.py --port 8080       # 指定端口
  python start_api_server.py --host 127.0.0.1  # 仅本地访问
  python start_api_server.py --no-reload       # 生产模式（不自动重载）
  python start_api_server.py --workers 4       # 多进程模式（生产环境）
        """
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="禁用自动重载 (生产模式)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="日志级别 (默认: info)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数量 (仅在--no-reload模式下有效，默认: 1)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查环境和依赖，不启动服务"
    )
    
    args = parser.parse_args()
    
    print("图像特征提取API服务启动器")
    print("=" * 50)
    
    # 检查依赖
    print("检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 所有依赖包已安装")
    
    # 检查特征提取器
    print("\n检查特征提取器...")
    if not check_feature_extractor():
        print("\n建议:")
        print("1. 确保PyTorch已正确安装")
        print("2. 如果使用Mac M系列芯片，确保安装了MPS支持的PyTorch版本")
        print("3. 检查CUDA环境（如果使用GPU）")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ 环境检查完成，所有组件正常")
        return
    
    # 启动服务器
    print("\n启动API服务器...")
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level,
        workers=args.workers
    )

if __name__ == "__main__":
    main()