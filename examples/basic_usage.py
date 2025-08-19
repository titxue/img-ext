#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础使用示例 - Mac M系列芯片优化版图像特征提取器

本示例展示如何使用FeatureExtractor进行图像特征提取，
特别针对Mac M系列芯片的MPS加速进行了优化。
"""

import sys
import os
import numpy as np
from PIL import Image
import torch
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor


def create_sample_image(size=(224, 224)):
    """创建示例图像"""
    # 创建一个渐变色的示例图像
    width, height = size
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # 创建彩色渐变效果
            image_array[i, j, 0] = int(255 * i / height)  # 红色通道
            image_array[i, j, 1] = int(255 * j / width)   # 绿色通道
            image_array[i, j, 2] = int(255 * (i + j) / (height + width))  # 蓝色通道
    
    return Image.fromarray(image_array)


def demonstrate_device_detection():
    """演示设备检测功能"""
    print("=== 设备检测演示 ===")
    
    # 自动设备检测
    extractor = FeatureExtractor(device='auto')
    device_info = extractor.get_device_info()
    
    print(f"检测到的设备: {device_info['device']}")
    print(f"设备类型: {device_info['device_type']}")
    print(f"MPS可用: {device_info['mps_available']}")
    print(f"CUDA可用: {device_info['cuda_available']}")
    
    if device_info['device_type'] == 'mps':
        print("🚀 Mac M系列芯片MPS加速已启用！")
        print(f"MPS构建状态: {device_info.get('mps_built', 'Unknown')}")
    elif device_info['device_type'] == 'cuda':
        print(f"🔥 CUDA加速已启用！")
        print(f"CUDA设备数量: {device_info.get('cuda_device_count', 'Unknown')}")
        print(f"CUDA设备名称: {device_info.get('cuda_device_name', 'Unknown')}")
    else:
        print("💻 使用CPU计算")
    
    print()
    return extractor


def demonstrate_single_image_extraction(extractor):
    """演示单张图像特征提取"""
    print("=== 单张图像特征提取演示 ===")
    
    # 创建示例图像
    sample_image = create_sample_image()
    print(f"创建示例图像: {sample_image.size}, 模式: {sample_image.mode}")
    
    # 提取特征并计时
    start_time = time.time()
    features = extractor.extract_features(sample_image)
    extraction_time = time.time() - start_time
    
    print(f"特征提取完成！")
    print(f"特征维度: {features.shape}")
    print(f"特征类型: {features.dtype}")
    print(f"提取时间: {extraction_time:.4f}秒")
    print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
    print(f"特征均值: {features.mean():.4f}")
    print(f"特征标准差: {features.std():.4f}")
    
    # 显示前10个特征值
    print(f"前10个特征值: {features[:10].tolist()}")
    print()
    
    return features


def demonstrate_batch_extraction(extractor):
    """演示批量图像特征提取"""
    print("=== 批量图像特征提取演示 ===")
    
    # 创建多张示例图像
    batch_size = 8
    sample_images = []
    
    for i in range(batch_size):
        # 创建不同的图像（通过改变尺寸和颜色）
        size = (224 + i * 10, 224 + i * 10)
        image = create_sample_image(size)
        # 调整回标准尺寸
        image = image.resize((224, 224))
        sample_images.append(image)
    
    print(f"创建了{batch_size}张示例图像")
    
    # 批量提取特征并计时
    start_time = time.time()
    batch_features = extractor.extract_features_batch(sample_images, batch_size=4)
    extraction_time = time.time() - start_time
    
    print(f"批量特征提取完成！")
    print(f"批量特征维度: {batch_features.shape}")
    print(f"批量提取时间: {extraction_time:.4f}秒")
    print(f"平均每张图像时间: {extraction_time/batch_size:.4f}秒")
    
    # 分析批量特征
    print(f"批量特征范围: [{batch_features.min():.4f}, {batch_features.max():.4f}]")
    print(f"批量特征均值: {batch_features.mean():.4f}")
    
    # 计算特征之间的相似性
    similarities = torch.cosine_similarity(batch_features[0:1], batch_features[1:], dim=1)
    print(f"第一张图像与其他图像的余弦相似度: {similarities.tolist()}")
    print()
    
    return batch_features


def demonstrate_different_formats(extractor):
    """演示不同图像格式的处理"""
    print("=== 不同图像格式处理演示 ===")
    
    # PIL图像
    pil_image = create_sample_image()
    pil_features = extractor.extract_features(pil_image)
    print(f"PIL图像特征提取: {pil_features.shape}")
    
    # Numpy数组
    numpy_image = np.array(pil_image)
    numpy_features = extractor.extract_features(numpy_image)
    print(f"Numpy数组特征提取: {numpy_features.shape}")
    
    # 验证结果一致性
    similarity = torch.cosine_similarity(pil_features.unsqueeze(0), numpy_features.unsqueeze(0))
    print(f"PIL和Numpy结果相似度: {similarity.item():.6f}")
    
    # 不同颜色模式
    rgba_image = pil_image.convert('RGBA')
    rgba_features = extractor.extract_features(rgba_image)
    print(f"RGBA图像特征提取: {rgba_features.shape}")
    
    gray_image = pil_image.convert('L')
    gray_features = extractor.extract_features(gray_image)
    print(f"灰度图像特征提取: {gray_features.shape}")
    
    print()


def demonstrate_performance_comparison():
    """演示不同设备的性能对比"""
    print("=== 性能对比演示 ===")
    
    # 创建测试图像
    test_images = [create_sample_image() for _ in range(10)]
    
    devices_to_test = ['auto', 'cpu']
    
    # 如果MPS可用，添加到测试列表
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices_to_test.insert(0, 'mps')
    
    # 如果CUDA可用，添加到测试列表
    if torch.cuda.is_available():
        devices_to_test.insert(-1, 'cuda')
    
    results = {}
    
    for device in devices_to_test:
        try:
            print(f"测试设备: {device}")
            extractor = FeatureExtractor(device=device)
            
            # 单张图像测试
            start_time = time.time()
            single_features = extractor.extract_features(test_images[0])
            single_time = time.time() - start_time
            
            # 批量测试
            start_time = time.time()
            batch_features = extractor.extract_features_batch(test_images, batch_size=5)
            batch_time = time.time() - start_time
            
            results[device] = {
                'single_time': single_time,
                'batch_time': batch_time,
                'avg_time': batch_time / len(test_images),
                'actual_device': str(extractor.device)
            }
            
            print(f"  实际使用设备: {extractor.device}")
            print(f"  单张图像时间: {single_time:.4f}秒")
            print(f"  批量处理时间: {batch_time:.4f}秒")
            print(f"  平均每张时间: {batch_time/len(test_images):.4f}秒")
            
        except Exception as e:
            print(f"  设备{device}测试失败: {e}")
            results[device] = None
        
        print()
    
    # 性能总结
    print("性能总结:")
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        fastest_device = min(valid_results.keys(), key=lambda x: valid_results[x]['avg_time'])
        print(f"最快设备: {fastest_device} ({valid_results[fastest_device]['avg_time']:.4f}秒/张)")
        
        if len(valid_results) > 1:
            slowest_device = max(valid_results.keys(), key=lambda x: valid_results[x]['avg_time'])
            speedup = valid_results[slowest_device]['avg_time'] / valid_results[fastest_device]['avg_time']
            print(f"相比最慢设备({slowest_device})提速: {speedup:.2f}x")
    
    print()


def main():
    """主函数"""
    print("🚀 Mac M系列芯片优化版图像特征提取器演示")
    print("=" * 50)
    
    try:
        # 设备检测演示
        extractor = demonstrate_device_detection()
        
        # 单张图像特征提取演示
        single_features = demonstrate_single_image_extraction(extractor)
        
        # 批量特征提取演示
        batch_features = demonstrate_batch_extraction(extractor)
        
        # 不同格式处理演示
        demonstrate_different_formats(extractor)
        
        # 性能对比演示
        demonstrate_performance_comparison()
        
        print("🎉 所有演示完成！")
        print("\n💡 使用提示:")
        print("1. 在Mac M系列芯片上，MPS加速可显著提升性能")
        print("2. 批量处理比单张处理更高效")
        print("3. 支持多种图像格式和输入方式")
        print("4. 特征向量可用于图像相似度计算、聚类等任务")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()