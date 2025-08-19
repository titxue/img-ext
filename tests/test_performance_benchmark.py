#!/usr/bin/env python3
"""
性能基准测试模块

测试内容:
- 处理时间基准测试
- 内存使用监控
- 批量vs单张处理对比
- 不同批次大小性能对比
- 设备性能对比 (MPS vs CPU)
"""

import time
import psutil
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import FeatureExtractor


class PerformanceBenchmark:
    """
    性能基准测试类
    """
    
    def __init__(self, test_images_dir: str):
        """
        初始化性能测试
        
        Args:
            test_images_dir: 测试图片目录
        """
        self.test_images_dir = Path(test_images_dir)
        self.test_images = self._load_test_images()
        self.results = {}
        
        print(f"加载了 {len(self.test_images)} 张测试图片")
    
    def _load_test_images(self) -> List[str]:
        """
        加载测试图片路径
        
        Returns:
            List[str]: 图片路径列表
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        
        for ext in image_extensions:
            images.extend(self.test_images_dir.glob(f'*{ext}'))
            images.extend(self.test_images_dir.glob(f'*{ext.upper()}'))
        
        return [str(img) for img in images]
    
    def _get_memory_usage(self) -> float:
        """
        获取当前内存使用量 (MB)
        
        Returns:
            float: 内存使用量
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _measure_time_and_memory(self, func, *args, **kwargs) -> Tuple[Any, float, float, float]:
        """
        测量函数执行时间和内存使用
        
        Args:
            func: 要测量的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Tuple[Any, float, float, float]: (结果, 执行时间, 开始内存, 结束内存)
        """
        # 强制垃圾回收
        gc.collect()
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        
        return result, execution_time, start_memory, end_memory
    
    def test_single_image_performance(self, extractor: FeatureExtractor, num_runs: int = 10) -> Dict[str, Any]:
        """
        测试单张图片处理性能
        
        Args:
            extractor: 特征提取器
            num_runs: 运行次数
            
        Returns:
            Dict[str, Any]: 性能测试结果
        """
        print(f"\n🔍 测试单张图片处理性能 (设备: {extractor.device})")
        
        times = []
        memory_usage = []
        
        # 选择前num_runs张图片进行测试
        test_images = self.test_images[:min(num_runs, len(self.test_images))]
        
        for i, img_path in enumerate(test_images):
            print(f"  处理图片 {i+1}/{len(test_images)}: {Path(img_path).name}")
            
            _, exec_time, start_mem, end_mem = self._measure_time_and_memory(
                extractor.extract_features, img_path
            )
            
            times.append(exec_time)
            memory_usage.append(end_mem - start_mem)
        
        results = {
            'device': str(extractor.device),
            'num_images': len(test_images),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'times': times,
            'memory_usage': memory_usage
        }
        
        print(f"  ✅ 平均处理时间: {results['avg_time']:.4f}s ± {results['std_time']:.4f}s")
        print(f"  📊 内存使用: {results['avg_memory']:.2f}MB ± {results['std_memory']:.2f}MB")
        
        return results
    
    def test_batch_performance(self, extractor: FeatureExtractor, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        测试批量处理性能
        
        Args:
            extractor: 特征提取器
            batch_sizes: 批次大小列表
            
        Returns:
            Dict[str, Any]: 批量处理性能结果
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        print(f"\n🔍 测试批量处理性能 (设备: {extractor.device})")
        
        results = {
            'device': str(extractor.device),
            'batch_results': {}
        }
        
        # 使用前32张图片进行批量测试
        test_images = self.test_images[:min(32, len(self.test_images))]
        
        for batch_size in batch_sizes:
            print(f"  测试批次大小: {batch_size}")
            
            _, exec_time, start_mem, end_mem = self._measure_time_and_memory(
                extractor.extract_features_batch, test_images, batch_size
            )
            
            # 计算每张图片的平均处理时间
            time_per_image = exec_time / len(test_images)
            memory_usage = end_mem - start_mem
            
            results['batch_results'][batch_size] = {
                'total_time': exec_time,
                'time_per_image': time_per_image,
                'memory_usage': memory_usage,
                'throughput': len(test_images) / exec_time  # 图片/秒
            }
            
            print(f"    ⏱️  总时间: {exec_time:.4f}s")
            print(f"    📈 每张图片: {time_per_image:.4f}s")
            print(f"    🚀 吞吐量: {results['batch_results'][batch_size]['throughput']:.2f} 图片/秒")
            print(f"    💾 内存使用: {memory_usage:.2f}MB")
        
        return results
    
    def compare_single_vs_batch(self, extractor: FeatureExtractor, num_images: int = 20) -> Dict[str, Any]:
        """
        对比单张处理vs批量处理性能
        
        Args:
            extractor: 特征提取器
            num_images: 测试图片数量
            
        Returns:
            Dict[str, Any]: 对比结果
        """
        print(f"\n🔍 对比单张vs批量处理性能 (设备: {extractor.device})")
        
        test_images = self.test_images[:min(num_images, len(self.test_images))]
        
        # 测试单张处理
        print("  测试单张处理...")
        single_start_time = time.time()
        single_start_mem = self._get_memory_usage()
        
        for img_path in test_images:
            extractor.extract_features(img_path)
        
        single_end_time = time.time()
        single_end_mem = self._get_memory_usage()
        
        single_total_time = single_end_time - single_start_time
        single_memory = single_end_mem - single_start_mem
        
        # 测试批量处理
        print("  测试批量处理...")
        _, batch_total_time, batch_start_mem, batch_end_mem = self._measure_time_and_memory(
            extractor.extract_features_batch, test_images, 16
        )
        
        batch_memory = batch_end_mem - batch_start_mem
        
        # 计算性能提升
        time_speedup = single_total_time / batch_total_time
        memory_ratio = batch_memory / single_memory if single_memory > 0 else 1
        
        results = {
            'device': str(extractor.device),
            'num_images': len(test_images),
            'single_processing': {
                'total_time': single_total_time,
                'time_per_image': single_total_time / len(test_images),
                'memory_usage': single_memory
            },
            'batch_processing': {
                'total_time': batch_total_time,
                'time_per_image': batch_total_time / len(test_images),
                'memory_usage': batch_memory
            },
            'performance_gain': {
                'time_speedup': time_speedup,
                'memory_ratio': memory_ratio
            }
        }
        
        print(f"  ✅ 单张处理总时间: {single_total_time:.4f}s")
        print(f"  ✅ 批量处理总时间: {batch_total_time:.4f}s")
        print(f"  🚀 性能提升: {time_speedup:.2f}x")
        print(f"  💾 内存比率: {memory_ratio:.2f}x")
        
        return results
    
    def run_comprehensive_benchmark(self, devices: List[str] = None) -> Dict[str, Any]:
        """
        运行全面的性能基准测试
        
        Args:
            devices: 要测试的设备列表
            
        Returns:
            Dict[str, Any]: 完整的基准测试结果
        """
        if devices is None:
            devices = ['auto']  # 默认使用自动设备选择
        
        print("🚀 开始全面性能基准测试")
        print(f"📊 测试图片数量: {len(self.test_images)}")
        print(f"🖥️  测试设备: {devices}")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'test_images_count': len(self.test_images),
            'devices_tested': [],
            'results': {}
        }
        
        for device in devices:
            print(f"\n{'='*50}")
            print(f"🖥️  测试设备: {device}")
            print(f"{'='*50}")
            
            try:
                # 创建特征提取器
                extractor = FeatureExtractor(device=device)
                device_name = str(extractor.device)
                all_results['devices_tested'].append(device_name)
                
                # 设备信息
                device_info = extractor.get_device_info()
                print(f"📋 设备信息: {device_info}")
                
                # 运行各项测试
                device_results = {
                    'device_info': device_info,
                    'single_image_performance': self.test_single_image_performance(extractor),
                    'batch_performance': self.test_batch_performance(extractor),
                    'single_vs_batch_comparison': self.compare_single_vs_batch(extractor)
                }
                
                all_results['results'][device_name] = device_results
                
            except Exception as e:
                print(f"❌ 设备 {device} 测试失败: {e}")
                continue
        
        # 保存结果
        self.results = all_results
        return all_results
    
    def save_results(self, output_path: str = None) -> str:
        """
        保存测试结果到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 测试结果已保存到: {output_path}")
        return output_path


if __name__ == "__main__":
    # 运行性能基准测试
    test_images_dir = "test_data/real_images"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试图片目录不存在: {test_images_dir}")
        exit(1)
    
    # 创建基准测试实例
    benchmark = PerformanceBenchmark(test_images_dir)
    
    # 运行测试
    results = benchmark.run_comprehensive_benchmark(['auto', 'cpu'])
    
    # 保存结果
    benchmark.save_results()
    
    print("\n🎉 性能基准测试完成!")