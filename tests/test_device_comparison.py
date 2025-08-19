#!/usr/bin/env python3
"""
设备性能对比测试模块

对比内容:
- MPS vs CPU 处理时间对比
- 内存使用对比
- 批量处理性能对比
- 特征质量一致性验证
- 设备切换开销测试
"""

import time
import psutil
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import FeatureExtractor


class DevicePerformanceComparator:
    """
    设备性能对比器
    """
    
    def __init__(self, test_images_dir: str):
        """
        初始化设备性能对比器
        
        Args:
            test_images_dir: 测试图片目录
        """
        self.test_images_dir = Path(test_images_dir)
        self.test_images = self._load_test_images()
        self.results = {}
        
        # 检查可用设备
        self.available_devices = self._check_available_devices()
        
        print(f"加载了 {len(self.test_images)} 张测试图片")
        print(f"可用设备: {', '.join(self.available_devices)}")
    
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
        
        return [str(img) for img in images[:20]]  # 限制为20张图片以加快测试
    
    def _check_available_devices(self) -> List[str]:
        """
        检查可用的设备
        
        Returns:
            List[str]: 可用设备列表
        """
        devices = ['cpu']
        
        # 检查CUDA
        if torch.cuda.is_available():
            devices.append('cuda')
        
        # 检查MPS (Mac M-series)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        return devices
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前内存使用情况
        
        Returns:
            Dict[str, float]: 内存使用信息
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存 (MB)
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存 (MB)
            'percent': process.memory_percent()        # 内存使用百分比
        }
    
    def benchmark_single_image_performance(self, image_path: str) -> Dict[str, Any]:
        """
        单张图片性能基准测试
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict[str, Any]: 各设备性能结果
        """
        print(f"\n🖼️  单张图片性能测试: {Path(image_path).name}")
        
        results = {}
        
        for device in self.available_devices:
            print(f"  测试设备: {device}")
            
            # 创建特征提取器
            extractor = FeatureExtractor(device=device)
            
            # 预热 (避免首次运行的初始化开销)
            _ = extractor.extract_features(image_path)
            
            # 记录初始内存
            initial_memory = self._get_memory_usage()
            
            # 多次运行取平均值
            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                features = extractor.extract_features(image_path)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # 记录峰值内存
            peak_memory = self._get_memory_usage()
            
            results[device] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'features_shape': features.shape,
                'memory_usage': {
                    'initial': initial_memory,
                    'peak': peak_memory,
                    'delta_rss_mb': peak_memory['rss_mb'] - initial_memory['rss_mb']
                }
            }
            
            print(f"    平均时间: {results[device]['mean_time']:.4f}s ± {results[device]['std_time']:.4f}s")
            print(f"    内存增量: {results[device]['memory_usage']['delta_rss_mb']:.2f}MB")
        
        return results
    
    def benchmark_batch_performance(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        批量处理性能基准测试
        
        Args:
            batch_sizes: 批量大小列表
            
        Returns:
            Dict[str, Any]: 各设备批量处理性能结果
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        
        print(f"\n📦 批量处理性能测试")
        
        results = {}
        
        for device in self.available_devices:
            print(f"  测试设备: {device}")
            device_results = {}
            
            extractor = FeatureExtractor(device=device)
            
            for batch_size in batch_sizes:
                if batch_size > len(self.test_images):
                    continue
                
                print(f"    批量大小: {batch_size}")
                
                # 选择测试图片
                test_batch = self.test_images[:batch_size]
                
                # 预热
                _ = extractor.extract_features_batch(test_batch, batch_size=batch_size)
                
                # 记录初始内存
                initial_memory = self._get_memory_usage()
                
                # 多次运行取平均值
                times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    features = extractor.extract_features_batch(test_batch, batch_size=batch_size)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                # 记录峰值内存
                peak_memory = self._get_memory_usage()
                
                device_results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'throughput': batch_size / np.mean(times),  # 图片/秒
                    'time_per_image': np.mean(times) / batch_size,
                    'features_shape': features.shape,
                    'memory_usage': {
                        'initial': initial_memory,
                        'peak': peak_memory,
                        'delta_rss_mb': peak_memory['rss_mb'] - initial_memory['rss_mb']
                    }
                }
                
                print(f"      平均时间: {device_results[f'batch_{batch_size}']['mean_time']:.4f}s")
                print(f"      吞吐量: {device_results[f'batch_{batch_size}']['throughput']:.2f} 图片/秒")
                print(f"      单图时间: {device_results[f'batch_{batch_size}']['time_per_image']:.4f}s")
            
            results[device] = device_results
        
        return results
    
    def test_feature_consistency(self, num_test_images: int = 5) -> Dict[str, Any]:
        """
        测试不同设备间特征一致性
        
        Args:
            num_test_images: 测试图片数量
            
        Returns:
            Dict[str, Any]: 特征一致性测试结果
        """
        print(f"\n🔍 特征一致性测试 (测试 {num_test_images} 张图片)")
        
        if len(self.available_devices) < 2:
            print("  ⚠️  只有一个设备可用，跳过一致性测试")
            return {'skipped': True, 'reason': 'Only one device available'}
        
        test_images = self.test_images[:num_test_images]
        device_features = {}
        
        # 在每个设备上提取特征
        for device in self.available_devices:
            print(f"  在 {device} 上提取特征...")
            extractor = FeatureExtractor(device=device)
            features = extractor.extract_features_batch(test_images, batch_size=4)
            device_features[device] = features
        
        # 计算设备间特征相似性
        consistency_results = {}
        device_pairs = []
        
        for i, device1 in enumerate(self.available_devices):
            for j, device2 in enumerate(self.available_devices[i+1:], i+1):
                pair_name = f"{device1}_vs_{device2}"
                device_pairs.append((device1, device2))
                
                features1 = device_features[device1]
                features2 = device_features[device2]
                
                # 计算余弦相似度
                similarities = []
                for k in range(len(features1)):
                    sim = cosine_similarity([features1[k]], [features2[k]])[0][0]
                    similarities.append(sim)
                
                # 计算L2距离
                l2_distances = []
                for k in range(len(features1)):
                    dist = np.linalg.norm(features1[k] - features2[k])
                    l2_distances.append(dist)
                
                consistency_results[pair_name] = {
                    'cosine_similarities': similarities,
                    'mean_cosine_similarity': np.mean(similarities),
                    'std_cosine_similarity': np.std(similarities),
                    'min_cosine_similarity': np.min(similarities),
                    'l2_distances': l2_distances,
                    'mean_l2_distance': np.mean(l2_distances),
                    'std_l2_distance': np.std(l2_distances),
                    'max_l2_distance': np.max(l2_distances)
                }
                
                print(f"  {pair_name}:")
                print(f"    平均余弦相似度: {consistency_results[pair_name]['mean_cosine_similarity']:.6f} ± {consistency_results[pair_name]['std_cosine_similarity']:.6f}")
                print(f"    平均L2距离: {consistency_results[pair_name]['mean_l2_distance']:.6f} ± {consistency_results[pair_name]['std_l2_distance']:.6f}")
        
        return {
            'device_pairs': device_pairs,
            'consistency_results': consistency_results,
            'num_test_images': num_test_images
        }
    
    def test_device_switching_overhead(self) -> Dict[str, Any]:
        """
        测试设备切换开销
        
        Returns:
            Dict[str, Any]: 设备切换开销测试结果
        """
        print("\n🔄 设备切换开销测试")
        
        if len(self.available_devices) < 2:
            print("  ⚠️  只有一个设备可用，跳过切换开销测试")
            return {'skipped': True, 'reason': 'Only one device available'}
        
        test_image = self.test_images[0]
        switching_results = []
        
        # 测试设备间切换的开销
        for i in range(len(self.available_devices)):
            for j in range(len(self.available_devices)):
                if i == j:
                    continue
                
                device1 = self.available_devices[i]
                device2 = self.available_devices[j]
                
                print(f"  测试切换: {device1} -> {device2}")
                
                # 在第一个设备上初始化
                extractor1 = FeatureExtractor(device=device1)
                _ = extractor1.extract_features(test_image)  # 预热
                
                # 记录切换时间
                switch_start = time.perf_counter()
                extractor2 = FeatureExtractor(device=device2)
                _ = extractor2.extract_features(test_image)  # 首次运行
                switch_end = time.perf_counter()
                
                switch_time = switch_end - switch_start
                
                # 测试切换后的稳定性能
                stable_times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    _ = extractor2.extract_features(test_image)
                    end_time = time.perf_counter()
                    stable_times.append(end_time - start_time)
                
                switching_results.append({
                    'from_device': device1,
                    'to_device': device2,
                    'switch_time': switch_time,
                    'stable_mean_time': np.mean(stable_times),
                    'stable_std_time': np.std(stable_times)
                })
                
                print(f"    切换时间: {switch_time:.4f}s")
                print(f"    稳定后时间: {np.mean(stable_times):.4f}s ± {np.std(stable_times):.4f}s")
        
        return {
            'switching_results': switching_results,
            'mean_switch_time': np.mean([r['switch_time'] for r in switching_results]),
            'max_switch_time': np.max([r['switch_time'] for r in switching_results])
        }
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        生成性能对比摘要
        
        Returns:
            Dict[str, Any]: 性能摘要
        """
        if not self.results:
            return {'error': 'No results available. Run comprehensive comparison first.'}
        
        summary = {
            'available_devices': self.available_devices,
            'test_images_count': len(self.test_images)
        }
        
        # 单张图片性能摘要
        if 'single_image_performance' in self.results:
            single_perf = self.results['single_image_performance']
            fastest_device = min(single_perf.keys(), key=lambda d: single_perf[d]['mean_time'])
            slowest_device = max(single_perf.keys(), key=lambda d: single_perf[d]['mean_time'])
            
            summary['single_image_summary'] = {
                'fastest_device': fastest_device,
                'fastest_time': single_perf[fastest_device]['mean_time'],
                'slowest_device': slowest_device,
                'slowest_time': single_perf[slowest_device]['mean_time'],
                'speedup_ratio': single_perf[slowest_device]['mean_time'] / single_perf[fastest_device]['mean_time']
            }
        
        # 批量处理性能摘要
        if 'batch_performance' in self.results:
            batch_perf = self.results['batch_performance']
            best_throughput = {}
            
            for device in batch_perf:
                device_throughputs = [batch_perf[device][batch]['throughput'] 
                                    for batch in batch_perf[device]]
                best_throughput[device] = max(device_throughputs)
            
            fastest_batch_device = max(best_throughput.keys(), key=lambda d: best_throughput[d])
            
            summary['batch_processing_summary'] = {
                'best_throughput_device': fastest_batch_device,
                'best_throughput': best_throughput[fastest_batch_device],
                'throughput_by_device': best_throughput
            }
        
        # 特征一致性摘要
        if 'feature_consistency' in self.results and not self.results['feature_consistency'].get('skipped'):
            consistency = self.results['feature_consistency']['consistency_results']
            avg_similarities = [result['mean_cosine_similarity'] for result in consistency.values()]
            
            summary['consistency_summary'] = {
                'mean_similarity': np.mean(avg_similarities),
                'min_similarity': np.min(avg_similarities),
                'consistency_score': np.mean(avg_similarities)  # 越接近1越好
            }
        
        return summary
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """
        运行全面的设备性能对比
        
        Returns:
            Dict[str, Any]: 完整的对比结果
        """
        print("🚀 开始全面设备性能对比")
        print(f"📊 测试图片数量: {len(self.test_images)}")
        print(f"🖥️  可用设备: {', '.join(self.available_devices)}")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'available_devices': self.available_devices,
            'test_images_count': len(self.test_images),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'torch_version': torch.__version__
            }
        }
        
        # 运行各项测试
        if self.test_images:
            all_results['single_image_performance'] = self.benchmark_single_image_performance(self.test_images[0])
        
        all_results['batch_performance'] = self.benchmark_batch_performance()
        all_results['feature_consistency'] = self.test_feature_consistency()
        all_results['device_switching'] = self.test_device_switching_overhead()
        
        # 生成摘要
        self.results = all_results
        all_results['performance_summary'] = self.generate_performance_summary()
        
        return all_results
    
    def save_results(self, output_path: str = None) -> str:
        """
        保存对比结果到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"device_performance_comparison_{timestamp}.json"
        
        # 处理numpy数组，转换为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 对比结果已保存到: {output_path}")
        return output_path


if __name__ == "__main__":
    # 运行设备性能对比
    test_images_dir = "test_data/real_images"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试图片目录不存在: {test_images_dir}")
        exit(1)
    
    # 创建对比器
    comparator = DevicePerformanceComparator(test_images_dir)
    
    # 运行对比
    results = comparator.run_comprehensive_comparison()
    
    # 保存结果
    comparator.save_results()
    
    print("\n🎉 设备性能对比完成!")