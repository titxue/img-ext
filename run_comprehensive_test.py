#!/usr/bin/env python3
"""
综合测试运行器

运行所有测试模块并生成完整的HTML测试报告，包括：
- 性能基准测试
- 特征质量分析
- 设备性能对比
- 可视化图表生成
- HTML报告生成
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests'))

# 导入测试模块
try:
    from tests.test_performance_benchmark import PerformanceBenchmark
    from tests.test_feature_quality import FeatureQualityAnalyzer
    from tests.test_device_comparison import DevicePerformanceComparator
    from tests.test_visualization import TestVisualizationGenerator
    from tests.test_report_generator import TestReportGenerator
    from feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有测试模块都已正确创建")
    sys.exit(1)


class ComprehensiveTestRunner:
    """
    综合测试运行器
    """
    
    def __init__(self, test_images_dir: str = "tests/test_data/real_images",
                 output_dir: str = "test_results"):
        """
        初始化测试运行器
        
        Args:
            test_images_dir: 测试图片目录
            output_dir: 输出目录
        """
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.charts_dir = self.output_dir / "charts"
        self.data_dir = self.output_dir / "data"
        self.charts_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        
        print(f"🚀 综合测试运行器初始化完成")
        print(f"📁 测试图片目录: {self.test_images_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"🖥️  当前设备: {self.feature_extractor.get_device_info()}")
    
    def check_prerequisites(self) -> bool:
        """
        检查测试前提条件
        
        Returns:
            bool: 是否满足测试条件
        """
        print("\n🔍 检查测试前提条件...")
        
        # 检查测试图片目录
        if not self.test_images_dir.exists():
            print(f"❌ 测试图片目录不存在: {self.test_images_dir}")
            return False
        
        # 检查测试图片数量
        image_files = list(self.test_images_dir.glob("*.jpg")) + list(self.test_images_dir.glob("*.png"))
        if len(image_files) < 5:
            print(f"❌ 测试图片数量不足: {len(image_files)} < 5")
            return False
        
        print(f"✅ 找到 {len(image_files)} 张测试图片")
        
        # 检查依赖库
        try:
            import torch
            import torchvision
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import sklearn
            import psutil
            import jinja2
            print("✅ 所有依赖库检查通过")
        except ImportError as e:
            print(f"❌ 缺少依赖库: {e}")
            return False
        
        return True
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """
        运行性能基准测试
        
        Returns:
            Dict[str, Any]: 性能测试结果
        """
        print("\n📊 运行性能基准测试...")
        
        try:
            benchmark = PerformanceBenchmark(
                feature_extractor=self.feature_extractor,
                test_images_dir=str(self.test_images_dir)
            )
            
            # 运行基准测试
            results = benchmark.run_comprehensive_benchmark()
            
            # 保存结果
            results_file = self.data_dir / "performance_benchmark.json"
            benchmark.save_results(str(results_file))
            
            print(f"✅ 性能基准测试完成，结果保存到: {results_file}")
            return results
            
        except Exception as e:
            print(f"❌ 性能基准测试失败: {e}")
            return {}
    
    def run_quality_analysis(self) -> Dict[str, Any]:
        """
        运行特征质量分析
        
        Returns:
            Dict[str, Any]: 质量分析结果
        """
        print("\n🔬 运行特征质量分析...")
        
        try:
            analyzer = FeatureQualityAnalyzer(
                feature_extractor=self.feature_extractor,
                test_images_dir=str(self.test_images_dir)
            )
            
            # 运行质量分析
            results = analyzer.run_comprehensive_analysis()
            
            # 保存结果
            results_file = self.data_dir / "feature_quality.json"
            analyzer.save_results(str(results_file))
            
            print(f"✅ 特征质量分析完成，结果保存到: {results_file}")
            return results
            
        except Exception as e:
            print(f"❌ 特征质量分析失败: {e}")
            return {}
    
    def run_device_comparison(self) -> Dict[str, Any]:
        """
        运行设备性能对比
        
        Returns:
            Dict[str, Any]: 设备对比结果
        """
        print("\n🖥️ 运行设备性能对比...")
        
        try:
            comparator = DevicePerformanceComparator(
                test_images_dir=str(self.test_images_dir)
            )
            
            # 运行设备对比
            results = comparator.run_comprehensive_comparison()
            
            # 保存结果
            results_file = self.data_dir / "device_comparison.json"
            comparator.save_results(str(results_file))
            
            print(f"✅ 设备性能对比完成，结果保存到: {results_file}")
            return results
            
        except Exception as e:
            print(f"❌ 设备性能对比失败: {e}")
            return {}
    
    def generate_visualizations(self, benchmark_data: Dict[str, Any],
                              quality_data: Dict[str, Any],
                              device_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        生成可视化图表
        
        Args:
            benchmark_data: 性能基准数据
            quality_data: 质量分析数据
            device_data: 设备对比数据
            
        Returns:
            Dict[str, List[str]]: 生成的图表文件路径
        """
        print("\n📈 生成可视化图表...")
        
        try:
            visualizer = TestVisualizationGenerator(output_dir=str(self.charts_dir))
            
            chart_paths = {
                'performance_benchmark': [],
                'feature_quality': [],
                'device_comparison': []
            }
            
            # 生成性能基准图表
            if benchmark_data:
                print("  📊 生成性能基准图表...")
                perf_charts = visualizer.plot_performance_benchmark(benchmark_data)
                chart_paths['performance_benchmark'] = perf_charts
                print(f"    ✅ 生成了 {len(perf_charts)} 个性能图表")
            
            # 生成质量分析图表
            if quality_data:
                print("  🔬 生成质量分析图表...")
                quality_charts = visualizer.plot_feature_quality_analysis(quality_data)
                chart_paths['feature_quality'] = quality_charts
                print(f"    ✅ 生成了 {len(quality_charts)} 个质量图表")
            
            # 生成设备对比图表
            if device_data:
                print("  🖥️ 生成设备对比图表...")
                device_charts = visualizer.plot_device_comparison(device_data)
                chart_paths['device_comparison'] = device_charts
                print(f"    ✅ 生成了 {len(device_charts)} 个设备对比图表")
            
            total_charts = sum(len(charts) for charts in chart_paths.values())
            print(f"✅ 可视化图表生成完成，共生成 {total_charts} 个图表")
            
            return chart_paths
            
        except Exception as e:
            print(f"❌ 可视化图表生成失败: {e}")
            return {'performance_benchmark': [], 'feature_quality': [], 'device_comparison': []}
    
    def generate_html_report(self, benchmark_data: Dict[str, Any],
                           quality_data: Dict[str, Any],
                           device_data: Dict[str, Any],
                           chart_paths: Dict[str, List[str]]) -> str:
        """
        生成HTML测试报告
        
        Args:
            benchmark_data: 性能基准数据
            quality_data: 质量分析数据
            device_data: 设备对比数据
            chart_paths: 图表文件路径
            
        Returns:
            str: 生成的报告文件路径
        """
        print("\n📄 生成HTML测试报告...")
        
        try:
            generator = TestReportGenerator(output_dir=str(self.output_dir))
            
            report_path = generator.generate_report(
                benchmark_data=benchmark_data,
                quality_data=quality_data,
                device_data=device_data,
                chart_paths=chart_paths,
                output_filename="comprehensive_test_report.html"
            )
            
            print(f"✅ HTML测试报告生成完成: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ HTML测试报告生成失败: {e}")
            return ""
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        运行完整的综合测试
        
        Returns:
            Dict[str, Any]: 完整测试结果
        """
        print("🎯 开始运行综合测试套件")
        print("=" * 60)
        
        start_time = time.time()
        
        # 检查前提条件
        if not self.check_prerequisites():
            print("❌ 前提条件检查失败，测试终止")
            return {}
        
        # 运行各项测试
        benchmark_data = self.run_performance_benchmark()
        quality_data = self.run_quality_analysis()
        device_data = self.run_device_comparison()
        
        # 生成可视化图表
        chart_paths = self.generate_visualizations(benchmark_data, quality_data, device_data)
        
        # 生成HTML报告
        report_path = self.generate_html_report(benchmark_data, quality_data, device_data, chart_paths)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 汇总结果
        summary = {
            'test_completed': True,
            'total_time': total_time,
            'benchmark_data': benchmark_data,
            'quality_data': quality_data,
            'device_data': device_data,
            'chart_paths': chart_paths,
            'report_path': report_path,
            'output_dir': str(self.output_dir)
        }
        
        print("\n" + "=" * 60)
        print(f"🎉 综合测试完成！总耗时: {total_time:.1f}秒")
        print(f"📊 测试数据保存在: {self.data_dir}")
        print(f"📈 图表保存在: {self.charts_dir}")
        if report_path:
            print(f"📄 HTML报告: {report_path}")
        print("=" * 60)
        
        return summary


def main():
    """
    主函数
    """
    print("🔍 图像特征提取器 - 综合测试套件")
    print("Version: 1.0")
    print("Author: AI Assistant")
    print()
    
    # 创建测试运行器
    runner = ComprehensiveTestRunner()
    
    # 运行综合测试
    results = runner.run_comprehensive_test()
    
    if results.get('test_completed', False):
        print("\n✅ 所有测试已成功完成！")
        
        # 显示关键统计信息
        if results.get('benchmark_data'):
            benchmark = results['benchmark_data']
            if 'single_image_results' in benchmark:
                print(f"📊 测试图片数量: {len(benchmark['single_image_results'])}")
        
        if results.get('chart_paths'):
            total_charts = sum(len(charts) for charts in results['chart_paths'].values())
            print(f"📈 生成图表数量: {total_charts}")
        
        if results.get('report_path'):
            print(f"📄 测试报告: {results['report_path']}")
            print("\n💡 提示: 可以在浏览器中打开HTML报告查看详细结果")
    else:
        print("\n❌ 测试过程中出现错误，请检查日志信息")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())