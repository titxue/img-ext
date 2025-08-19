#!/usr/bin/env python3
"""
FastAPI客户端使用示例

本示例展示如何使用requests库调用图像特征提取API服务
包括单张图像和批量图像的特征提取功能
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time

class FeatureExtractionClient:
    """图像特征提取API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """
        检查API服务健康状态
        
        Returns:
            Dict: 健康检查结果
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        获取服务器设备信息
        
        Returns:
            Dict: 设备信息
        """
        try:
            response = self.session.get(f"{self.base_url}/device-info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def extract_single_features(self, image_path: str) -> Dict[str, Any]:
        """
        提取单张图像的特征
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Dict: 特征提取结果
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"图像文件不存在: {image_path}"}
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = self.session.post(
                    f"{self.base_url}/extract-features",
                    files=files
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def extract_batch_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """
        批量提取图像特征
        
        Args:
            image_paths: 图像文件路径列表
            batch_size: 批处理大小
            
        Returns:
            Dict: 批量特征提取结果
        """
        # 验证所有文件存在
        valid_paths = []
        for path in image_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                print(f"警告: 图像文件不存在，跳过: {path}")
        
        if not valid_paths:
            return {"error": "没有有效的图像文件"}
        
        try:
            files = []
            for path in valid_paths:
                with open(path, 'rb') as f:
                    files.append(('files', (Path(path).name, f.read(), 'image/jpeg')))
            
            data = {'batch_size': batch_size}
            response = self.session.post(
                f"{self.base_url}/extract-features-batch",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def demo_single_image():
    """单张图像特征提取演示"""
    print("=== 单张图像特征提取演示 ===")
    
    client = FeatureExtractionClient()
    
    # 健康检查
    health = client.health_check()
    print(f"服务健康状态: {health.get('status', 'unknown')}")
    
    # 获取设备信息
    device_info = client.get_device_info()
    if 'error' not in device_info:
        print(f"服务器设备: {device_info.get('device_info', {}).get('device', 'unknown')}")
    
    # 示例图像路径（请根据实际情况修改）
    image_path = "tests/test_data/sample_image.jpg"
    
    # 如果示例图像不存在，创建一个简单的测试图像
    if not Path(image_path).exists():
        print(f"示例图像不存在: {image_path}")
        print("请将图像文件放置在指定路径，或修改image_path变量")
        return
    
    # 提取特征
    print(f"正在提取图像特征: {image_path}")
    result = client.extract_single_features(image_path)
    
    if 'error' in result:
        print(f"错误: {result['error']}")
    else:
        print(f"提取成功!")
        print(f"特征维度: {len(result.get('features', []))}")
        print(f"处理时间: {result.get('processing_time', 0):.3f}秒")
        print(f"设备: {result.get('device_info', {}).get('device', 'unknown')}")
        print(f"消息: {result.get('message', '')}")

def demo_batch_images():
    """批量图像特征提取演示"""
    print("\n=== 批量图像特征提取演示 ===")
    
    client = FeatureExtractionClient()
    
    # 示例图像路径列表（请根据实际情况修改）
    image_paths = [
        "tests/test_data/sample_image.jpg",
        "tests/test_data/sample_image2.jpg",
        "tests/test_data/sample_image3.jpg"
    ]
    
    # 检查文件是否存在
    existing_paths = [path for path in image_paths if Path(path).exists()]
    if not existing_paths:
        print("没有找到示例图像文件")
        print("请将图像文件放置在tests/test_data/目录下")
        return
    
    print(f"正在批量提取{len(existing_paths)}张图像的特征...")
    result = client.extract_batch_features(existing_paths, batch_size=16)
    
    if 'error' in result:
        print(f"错误: {result['error']}")
    else:
        print(f"批量提取成功!")
        print(f"处理图像数量: {result.get('count', 0)}")
        print(f"特征矩阵形状: [{result.get('count', 0)}, {len(result.get('features', [[]])[0]) if result.get('features') else 0}]")
        print(f"处理时间: {result.get('processing_time', 0):.3f}秒")
        print(f"设备: {result.get('device_info', {}).get('device', 'unknown')}")
        print(f"消息: {result.get('message', '')}")

def demo_performance_comparison():
    """性能对比演示"""
    print("\n=== 性能对比演示 ===")
    
    client = FeatureExtractionClient()
    
    # 创建测试图像列表
    test_images = ["tests/test_data/sample_image.jpg"] * 10  # 重复使用同一张图像
    existing_images = [path for path in test_images if Path(path).exists()]
    
    if not existing_images:
        print("没有找到测试图像")
        return
    
    print(f"使用{len(existing_images)}张图像进行性能测试...")
    
    # 单张处理性能
    start_time = time.time()
    single_results = []
    for image_path in existing_images:
        result = client.extract_single_features(image_path)
        if 'error' not in result:
            single_results.append(result)
    single_time = time.time() - start_time
    
    # 批量处理性能
    start_time = time.time()
    batch_result = client.extract_batch_features(existing_images)
    batch_time = time.time() - start_time
    
    print(f"\n性能对比结果:")
    print(f"单张处理: {single_time:.3f}秒 ({len(single_results)}张图像)")
    print(f"批量处理: {batch_time:.3f}秒 ({batch_result.get('count', 0)}张图像)")
    
    if single_time > 0 and batch_time > 0:
        speedup = single_time / batch_time
        print(f"批量处理加速比: {speedup:.2f}x")

def main():
    """主函数"""
    print("图像特征提取API客户端演示")
    print("=" * 50)
    
    try:
        # 单张图像演示
        demo_single_image()
        
        # 批量图像演示
        demo_batch_images()
        
        # 性能对比演示
        demo_performance_comparison()
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {str(e)}")
    
    print("\n演示完成!")
    print("\n使用说明:")
    print("1. 确保FastAPI服务正在运行 (python app.py)")
    print("2. 将测试图像放置在tests/test_data/目录下")
    print("3. 运行此脚本查看API调用示例")

if __name__ == "__main__":
    main()