#!/usr/bin/env python3
"""
真实图片测试脚本
使用从网站爬取的真实产品图片测试特征提取器性能
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import time
import os
from PIL import Image
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractor import FeatureExtractor

class TestRealImages:
    """真实图片测试类"""
    
    @pytest.fixture(scope="class")
    def extractor(self):
        """创建特征提取器实例"""
        return FeatureExtractor()
    
    @pytest.fixture(scope="class")
    def real_images_dir(self):
        """获取真实图片目录"""
        images_dir = Path(__file__).parent / "test_data" / "real_images"
        if not images_dir.exists():
            pytest.skip("真实图片目录不存在，请先运行爬虫脚本")
        return images_dir
    
    @pytest.fixture(scope="class")
    def image_files(self, real_images_dir):
        """获取所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in real_images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            pytest.skip("没有找到图片文件")
        
        return image_files[:20]  # 限制测试图片数量
    
    def test_device_detection(self):
        """测试设备检测功能"""
        extractor = FeatureExtractor()
        device_info = extractor.get_device_info()
        
        print(f"设备信息: {device_info}")
        
        # 验证设备类型
        assert device_info['device_type'] in ['cuda', 'cpu', 'mps'], f"未知设备类型: {device_info['device_type']}"
        
        # 验证设备可用性检测
        if device_info['device_type'] == 'cuda':
            assert device_info['cuda_available'] == True
        elif device_info['device_type'] == 'mps':
            assert device_info['mps_available'] == True
        else:
            assert device_info['device_type'] == 'cpu'
    
    def test_single_real_image_extraction(self, extractor, image_files):
        """测试单张真实图片特征提取"""
        test_image = image_files[0]
        print(f"\n测试图片: {test_image.name}")
        
        # 加载图片
        image = Image.open(test_image)
        print(f"图片尺寸: {image.size}")
        print(f"图片模式: {image.mode}")
        
        # 提取特征
        start_time = time.time()
        features = extractor.extract_features(image)
        extraction_time = time.time() - start_time
        
        # 验证特征
        assert isinstance(features, np.ndarray), "特征应该是numpy数组"
        assert features.shape == (512,), f"特征维度应该是(512,)，实际是{features.shape}"
        assert not np.isnan(features).any(), "特征中不应该包含NaN值"
        assert not np.isinf(features).any(), "特征中不应该包含无穷值"
        
        print(f"特征提取时间: {extraction_time:.3f}秒")
        print(f"特征统计: 均值={features.mean():.4f}, 标准差={features.std():.4f}")
        print(f"特征范围: [{features.min():.4f}, {features.max():.4f}]")
    
    def test_batch_real_images_extraction(self, extractor, image_files):
        """测试批量真实图片特征提取"""
        # 选择前10张图片进行批量测试
        test_images = image_files[:10]
        print(f"\n批量测试 {len(test_images)} 张图片")
        
        # 加载图片
        images = []
        for img_path in test_images:
            try:
                image = Image.open(img_path)
                # 确保图片是RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"跳过损坏的图片 {img_path.name}: {e}")
        
        if not images:
            pytest.skip("没有有效的图片可以测试")
        
        # 批量提取特征
        start_time = time.time()
        features = extractor.extract_features_batch(images)
        extraction_time = time.time() - start_time
        
        # 验证特征
        assert isinstance(features, np.ndarray), "批量特征应该是numpy数组"
        assert features.shape == (len(images), 512), f"批量特征维度应该是({len(images)}, 512)，实际是{features.shape}"
        assert not np.isnan(features).any(), "批量特征中不应该包含NaN值"
        assert not np.isinf(features).any(), "批量特征中不应该包含无穷值"
        
        print(f"批量特征提取时间: {extraction_time:.3f}秒")
        print(f"平均每张图片: {extraction_time/len(images):.3f}秒")
        print(f"批量特征统计: 均值={features.mean():.4f}, 标准差={features.std():.4f}")
    
    def test_different_image_formats(self, extractor, real_images_dir):
        """测试不同格式的图片"""
        format_stats = {}
        
        for img_path in real_images_dir.iterdir():
            if not img_path.is_file():
                continue
                
            try:
                image = Image.open(img_path)
                format_name = image.format or 'Unknown'
                
                # 提取特征
                features = extractor.extract_features(image)
                
                # 统计格式信息
                if format_name not in format_stats:
                    format_stats[format_name] = {'count': 0, 'sizes': []}
                
                format_stats[format_name]['count'] += 1
                format_stats[format_name]['sizes'].append(image.size)
                
                # 验证特征
                assert features.shape == (512,), f"格式{format_name}的特征维度错误"
                assert not np.isnan(features).any(), f"格式{format_name}的特征包含NaN"
                
            except Exception as e:
                print(f"处理图片 {img_path.name} 时出错: {e}")
        
        # 打印格式统计
        print("\n图片格式统计:")
        for format_name, stats in format_stats.items():
            print(f"  {format_name}: {stats['count']} 张")
            if stats['sizes']:
                avg_width = sum(s[0] for s in stats['sizes']) / len(stats['sizes'])
                avg_height = sum(s[1] for s in stats['sizes']) / len(stats['sizes'])
                print(f"    平均尺寸: {avg_width:.0f}x{avg_height:.0f}")
        
        assert len(format_stats) > 0, "应该至少处理一种图片格式"
    
    def test_feature_consistency(self, extractor, image_files):
        """测试特征提取的一致性"""
        if len(image_files) < 2:
            pytest.skip("需要至少2张图片来测试一致性")
        
        test_image = image_files[0]
        image = Image.open(test_image)
        
        # 多次提取同一张图片的特征
        features_list = []
        for i in range(3):
            features = extractor.extract_features(image)
            features_list.append(features)
        
        # 验证一致性
        for i in range(1, len(features_list)):
            np.testing.assert_array_almost_equal(
                features_list[0], features_list[i], decimal=6,
                err_msg="同一张图片的特征提取结果应该一致"
            )
        
        print(f"\n特征一致性测试通过: {len(features_list)} 次提取结果完全一致")
    
    def test_performance_benchmark(self, extractor, image_files):
        """性能基准测试"""
        if len(image_files) < 10:
            pytest.skip("需要至少10张图片进行性能测试")
        
        test_images = image_files[:10]
        images = []
        
        # 加载图片
        for img_path in test_images:
            try:
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images.append(image)
            except Exception:
                continue
        
        if len(images) < 5:
            pytest.skip("有效图片数量不足")
        
        # 单张图片性能测试
        single_times = []
        for image in images[:5]:
            start_time = time.time()
            extractor.extract_features(image)
            single_times.append(time.time() - start_time)
        
        # 批量处理性能测试
        start_time = time.time()
        extractor.extract_features_batch(images)
        batch_time = time.time() - start_time
        
        # 性能统计
        avg_single_time = np.mean(single_times)
        total_single_time = sum(single_times)
        
        print(f"\n性能基准测试结果:")
        print(f"  单张图片平均时间: {avg_single_time:.3f}秒")
        print(f"  {len(images)}张图片单独处理总时间: {total_single_time:.3f}秒")
        print(f"  {len(images)}张图片批量处理时间: {batch_time:.3f}秒")
        print(f"  批量处理效率提升: {total_single_time/batch_time:.2f}x")
        
        # 性能断言
        assert avg_single_time < 2.0, "单张图片处理时间不应超过2秒"
        # 批量处理性能取决于设备和数据量，仅作信息展示
        if batch_time < total_single_time:
            print("  ✅ 批量处理比单独处理更快")
        else:
            print("  ℹ️  在当前设备和数据量下，批量处理优势不明显")

if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])