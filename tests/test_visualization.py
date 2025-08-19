#!/usr/bin/env python3
"""
可视化图表生成模块

生成图表:
- 性能基准测试图表
- 特征质量分析图表
- 设备性能对比图表
- 特征分布和聚类可视化
- 降维可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class TestVisualizationGenerator:
    """
    测试可视化图表生成器
    """
    
    def __init__(self, output_dir: str = "test_results/visualizations"):
        """
        初始化可视化生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        self.colors = sns.color_palette("husl", 10)
        self.figsize_large = (12, 8)
        self.figsize_medium = (10, 6)
        self.figsize_small = (8, 6)
        
        print(f"可视化输出目录: {self.output_dir}")
    
    def plot_performance_benchmark(self, benchmark_data: Dict[str, Any]) -> List[str]:
        """
        绘制性能基准测试图表
        
        Args:
            benchmark_data: 性能基准测试数据
            
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        print("\n📊 生成性能基准测试图表")
        
        generated_plots = []
        
        # 1. 单张图片处理时间对比
        if 'single_image_results' in benchmark_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
            
            single_results = benchmark_data['single_image_results']
            images = list(single_results.keys())
            times = [single_results[img]['processing_time'] for img in images]
            memory_usage = [single_results[img]['memory_usage']['delta_rss_mb'] for img in images]
            
            # 处理时间柱状图
            bars1 = ax1.bar(range(len(images)), times, color=self.colors[0], alpha=0.7)
            ax1.set_xlabel('测试图片')
            ax1.set_ylabel('处理时间 (秒)')
            ax1.set_title('单张图片处理时间')
            ax1.set_xticks(range(len(images)))
            ax1.set_xticklabels([Path(img).stem for img in images], rotation=45, ha='right')
            
            # 添加数值标签
            for bar, time_val in zip(bars1, times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
            
            # 内存使用柱状图
            bars2 = ax2.bar(range(len(images)), memory_usage, color=self.colors[1], alpha=0.7)
            ax2.set_xlabel('测试图片')
            ax2.set_ylabel('内存增量 (MB)')
            ax2.set_title('单张图片内存使用')
            ax2.set_xticks(range(len(images)))
            ax2.set_xticklabels([Path(img).stem for img in images], rotation=45, ha='right')
            
            # 添加数值标签
            for bar, mem_val in zip(bars2, memory_usage):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{mem_val:.1f}MB', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plot_path = self.output_dir / "single_image_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 单张图片性能图表: {plot_path}")
        
        # 2. 批量处理性能对比
        if 'batch_results' in benchmark_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
            
            batch_results = benchmark_data['batch_results']
            batch_sizes = list(batch_results.keys())
            processing_times = [batch_results[bs]['processing_time'] for bs in batch_sizes]
            throughputs = [batch_results[bs]['throughput'] for bs in batch_sizes]
            
            # 处理时间曲线
            ax1.plot(batch_sizes, processing_times, marker='o', linewidth=2, markersize=8, color=self.colors[2])
            ax1.set_xlabel('批量大小')
            ax1.set_ylabel('处理时间 (秒)')
            ax1.set_title('批量处理时间')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bs, time_val in zip(batch_sizes, processing_times):
                ax1.annotate(f'{time_val:.3f}s', (bs, time_val), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            # 吞吐量曲线
            ax2.plot(batch_sizes, throughputs, marker='s', linewidth=2, markersize=8, color=self.colors[3])
            ax2.set_xlabel('批量大小')
            ax2.set_ylabel('吞吐量 (图片/秒)')
            ax2.set_title('批量处理吞吐量')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bs, throughput in zip(batch_sizes, throughputs):
                ax2.annotate(f'{throughput:.1f}', (bs, throughput), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plot_path = self.output_dir / "batch_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 批量处理性能图表: {plot_path}")
        
        return generated_plots
    
    def plot_feature_quality_analysis(self, quality_data: Dict[str, Any]) -> List[str]:
        """
        绘制特征质量分析图表
        
        Args:
            quality_data: 特征质量分析数据
            
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        print("\n📊 生成特征质量分析图表")
        
        generated_plots = []
        
        # 1. 特征分布统计
        if 'distribution_analysis' in quality_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            dist_data = quality_data['distribution_analysis']
            overall_stats = dist_data['overall_stats']
            
            # 整体统计柱状图
            stats_names = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Q25', 'Q75']
            stats_values = [overall_stats['mean'], overall_stats['std'], overall_stats['min'], 
                          overall_stats['max'], overall_stats['median'], 
                          overall_stats['q25'], overall_stats['q75']]
            
            bars = ax1.bar(stats_names, stats_values, color=self.colors[:len(stats_names)], alpha=0.7)
            ax1.set_title('特征整体统计分布')
            ax1.set_ylabel('数值')
            
            # 添加数值标签
            for bar, val in zip(bars, stats_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 特征维度方差分布
            if 'dimension_stats' in dist_data:
                dim_stats = dist_data['dimension_stats']
                ax2.hist(dim_stats['std_per_dim'], bins=30, color=self.colors[1], alpha=0.7, edgecolor='black')
                ax2.set_title('特征维度标准差分布')
                ax2.set_xlabel('标准差')
                ax2.set_ylabel('频次')
            
            # 稀疏性分析
            if 'sparsity' in dist_data:
                sparsity = dist_data['sparsity']
                labels = ['非零值', '零值', '近零值']
                sizes = [1 - sparsity['zero_ratio'], 
                        sparsity['zero_ratio'] - sparsity['near_zero_ratio'],
                        sparsity['near_zero_ratio']]
                colors_pie = [self.colors[0], self.colors[1], self.colors[2]]
                
                ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax3.set_title('特征稀疏性分布')
            
            # 正态性检验结果
            if 'normality_tests' in dist_data:
                normality_tests = dist_data['normality_tests']
                dimensions = [t['dimension'] for t in normality_tests]
                p_values = [t['p_value'] for t in normality_tests]
                is_normal = [t['is_normal'] for t in normality_tests]
                
                colors_norm = [self.colors[0] if normal else self.colors[1] for normal in is_normal]
                ax4.bar(dimensions, p_values, color=colors_norm, alpha=0.7)
                ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='显著性水平 (0.05)')
                ax4.set_title('特征维度正态性检验')
                ax4.set_xlabel('特征维度')
                ax4.set_ylabel('P值')
                ax4.legend()
            
            plt.tight_layout()
            plot_path = self.output_dir / "feature_distribution_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 特征分布分析图表: {plot_path}")
        
        # 2. 相似性分析热力图
        if 'similarity_analysis' in quality_data:
            sim_data = quality_data['similarity_analysis']
            
            if 'cosine_similarity' in sim_data and 'matrix' in sim_data['cosine_similarity']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # 余弦相似度矩阵热力图
                cosine_matrix = np.array(sim_data['cosine_similarity']['matrix'])
                sns.heatmap(cosine_matrix, annot=False, cmap='viridis', ax=ax1, 
                           cbar_kws={'label': '余弦相似度'})
                ax1.set_title('图片间余弦相似度矩阵')
                ax1.set_xlabel('图片索引')
                ax1.set_ylabel('图片索引')
                
                # 相似度分布直方图
                mask = ~np.eye(cosine_matrix.shape[0], dtype=bool)
                similarities = cosine_matrix[mask]
                
                ax2.hist(similarities, bins=30, color=self.colors[2], alpha=0.7, edgecolor='black')
                ax2.axvline(np.mean(similarities), color='red', linestyle='--', 
                           label=f'平均值: {np.mean(similarities):.3f}')
                ax2.set_title('余弦相似度分布')
                ax2.set_xlabel('余弦相似度')
                ax2.set_ylabel('频次')
                ax2.legend()
                
                plt.tight_layout()
                plot_path = self.output_dir / "similarity_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(str(plot_path))
                print(f"  ✅ 相似性分析图表: {plot_path}")
        
        # 3. 聚类分析
        if 'clustering_analysis' in quality_data:
            cluster_data = quality_data['clustering_analysis']
            
            if 'clustering_results' in cluster_data:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
                
                clustering_results = cluster_data['clustering_results']
                n_clusters = [r['n_clusters'] for r in clustering_results]
                silhouette_scores = [r['silhouette_score'] for r in clustering_results]
                calinski_scores = [r['calinski_harabasz_score'] for r in clustering_results]
                
                # 轮廓系数曲线
                ax1.plot(n_clusters, silhouette_scores, marker='o', linewidth=2, 
                        markersize=8, color=self.colors[0], label='轮廓系数')
                ax1.set_xlabel('聚类数量')
                ax1.set_ylabel('轮廓系数')
                ax1.set_title('聚类数量 vs 轮廓系数')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # 标记最佳点
                best_idx = np.argmax(silhouette_scores)
                ax1.scatter(n_clusters[best_idx], silhouette_scores[best_idx], 
                           color='red', s=100, zorder=5)
                ax1.annotate(f'最佳: {n_clusters[best_idx]}', 
                           (n_clusters[best_idx], silhouette_scores[best_idx]),
                           textcoords="offset points", xytext=(10,10), ha='left')
                
                # Calinski-Harabasz指数曲线
                ax2.plot(n_clusters, calinski_scores, marker='s', linewidth=2, 
                        markersize=8, color=self.colors[1], label='Calinski-Harabasz指数')
                ax2.set_xlabel('聚类数量')
                ax2.set_ylabel('Calinski-Harabasz指数')
                ax2.set_title('聚类数量 vs Calinski-Harabasz指数')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # 标记最佳点
                best_idx = np.argmax(calinski_scores)
                ax2.scatter(n_clusters[best_idx], calinski_scores[best_idx], 
                           color='red', s=100, zorder=5)
                ax2.annotate(f'最佳: {n_clusters[best_idx]}', 
                           (n_clusters[best_idx], calinski_scores[best_idx]),
                           textcoords="offset points", xytext=(10,10), ha='left')
                
                plt.tight_layout()
                plot_path = self.output_dir / "clustering_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(str(plot_path))
                print(f"  ✅ 聚类分析图表: {plot_path}")
        
        # 4. 降维可视化
        if 'dimensionality_analysis' in quality_data:
            dim_data = quality_data['dimensionality_analysis']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # PCA方差解释比
            if 'pca' in dim_data:
                pca_data = dim_data['pca']
                
                # 累积方差解释比
                cumulative_var = pca_data['cumulative_variance_ratio']
                ax1.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                        marker='o', linewidth=2, color=self.colors[0])
                ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95%方差')
                ax1.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99%方差')
                ax1.set_xlabel('主成分数量')
                ax1.set_ylabel('累积方差解释比')
                ax1.set_title('PCA累积方差解释比')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 前20个主成分的方差解释比
                explained_var = pca_data['explained_variance_ratio'][:20]
                ax2.bar(range(1, len(explained_var) + 1), explained_var, 
                       color=self.colors[1], alpha=0.7)
                ax2.set_xlabel('主成分')
                ax2.set_ylabel('方差解释比')
                ax2.set_title('前20个主成分方差解释比')
                
                # PCA 2D可视化
                if 'features_2d' in pca_data:
                    pca_2d = np.array(pca_data['features_2d'])
                    scatter = ax3.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                        c=range(len(pca_2d)), cmap='viridis', alpha=0.7)
                    ax3.set_xlabel('第一主成分')
                    ax3.set_ylabel('第二主成分')
                    ax3.set_title('PCA 2D投影')
                    plt.colorbar(scatter, ax=ax3, label='图片索引')
            
            # t-SNE 2D可视化
            if 'tsne' in dim_data and 'features_2d' in dim_data['tsne']:
                tsne_2d = np.array(dim_data['tsne']['features_2d'])
                scatter = ax4.scatter(tsne_2d[:, 0], tsne_2d[:, 1], 
                                    c=range(len(tsne_2d)), cmap='plasma', alpha=0.7)
                ax4.set_xlabel('t-SNE维度1')
                ax4.set_ylabel('t-SNE维度2')
                ax4.set_title('t-SNE 2D投影')
                plt.colorbar(scatter, ax=ax4, label='图片索引')
            
            plt.tight_layout()
            plot_path = self.output_dir / "dimensionality_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 降维分析图表: {plot_path}")
        
        return generated_plots
    
    def plot_device_comparison(self, comparison_data: Dict[str, Any]) -> List[str]:
        """
        绘制设备性能对比图表
        
        Args:
            comparison_data: 设备对比数据
            
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        print("\n📊 生成设备性能对比图表")
        
        generated_plots = []
        
        # 1. 单张图片性能对比
        if 'single_image_performance' in comparison_data:
            single_perf = comparison_data['single_image_performance']
            devices = list(single_perf.keys())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
            
            # 处理时间对比
            mean_times = [single_perf[device]['mean_time'] for device in devices]
            std_times = [single_perf[device]['std_time'] for device in devices]
            
            bars1 = ax1.bar(devices, mean_times, yerr=std_times, capsize=5, 
                           color=self.colors[:len(devices)], alpha=0.7)
            ax1.set_ylabel('处理时间 (秒)')
            ax1.set_title('单张图片处理时间对比')
            
            # 添加数值标签
            for bar, time_val in zip(bars1, mean_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time_val:.4f}s', ha='center', va='bottom')
            
            # 内存使用对比
            memory_deltas = [single_perf[device]['memory_usage']['delta_rss_mb'] for device in devices]
            bars2 = ax2.bar(devices, memory_deltas, color=self.colors[:len(devices)], alpha=0.7)
            ax2.set_ylabel('内存增量 (MB)')
            ax2.set_title('单张图片内存使用对比')
            
            # 添加数值标签
            for bar, mem_val in zip(bars2, memory_deltas):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{mem_val:.1f}MB', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.output_dir / "device_single_image_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 单张图片设备对比图表: {plot_path}")
        
        # 2. 批量处理性能对比
        if 'batch_performance' in comparison_data:
            batch_perf = comparison_data['batch_performance']
            devices = list(batch_perf.keys())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
            
            # 收集所有批量大小
            all_batch_sizes = set()
            for device in devices:
                for batch_key in batch_perf[device].keys():
                    batch_size = batch_perf[device][batch_key]['batch_size']
                    all_batch_sizes.add(batch_size)
            
            batch_sizes = sorted(list(all_batch_sizes))
            
            # 处理时间对比
            for i, device in enumerate(devices):
                times = []
                throughputs = []
                
                for bs in batch_sizes:
                    batch_key = f'batch_{bs}'
                    if batch_key in batch_perf[device]:
                        times.append(batch_perf[device][batch_key]['mean_time'])
                        throughputs.append(batch_perf[device][batch_key]['throughput'])
                    else:
                        times.append(np.nan)
                        throughputs.append(np.nan)
                
                ax1.plot(batch_sizes, times, marker='o', linewidth=2, 
                        markersize=6, label=device, color=self.colors[i])
                ax2.plot(batch_sizes, throughputs, marker='s', linewidth=2, 
                        markersize=6, label=device, color=self.colors[i])
            
            ax1.set_xlabel('批量大小')
            ax1.set_ylabel('处理时间 (秒)')
            ax1.set_title('批量处理时间对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('批量大小')
            ax2.set_ylabel('吞吐量 (图片/秒)')
            ax2.set_title('批量处理吞吐量对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "device_batch_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 批量处理设备对比图表: {plot_path}")
        
        # 3. 特征一致性对比
        if 'feature_consistency' in comparison_data and not comparison_data['feature_consistency'].get('skipped'):
            consistency_data = comparison_data['feature_consistency']['consistency_results']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_large)
            
            pairs = list(consistency_data.keys())
            cosine_similarities = [consistency_data[pair]['mean_cosine_similarity'] for pair in pairs]
            l2_distances = [consistency_data[pair]['mean_l2_distance'] for pair in pairs]
            
            # 余弦相似度对比
            bars1 = ax1.bar(pairs, cosine_similarities, color=self.colors[:len(pairs)], alpha=0.7)
            ax1.set_ylabel('平均余弦相似度')
            ax1.set_title('设备间特征余弦相似度')
            ax1.set_xticklabels(pairs, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, sim_val in zip(bars1, cosine_similarities):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{sim_val:.6f}', ha='center', va='bottom', fontsize=9)
            
            # L2距离对比
            bars2 = ax2.bar(pairs, l2_distances, color=self.colors[:len(pairs)], alpha=0.7)
            ax2.set_ylabel('平均L2距离')
            ax2.set_title('设备间特征L2距离')
            ax2.set_xticklabels(pairs, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, dist_val in zip(bars2, l2_distances):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{dist_val:.6f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plot_path = self.output_dir / "device_consistency_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            print(f"  ✅ 设备一致性对比图表: {plot_path}")
        
        # 4. 性能摘要雷达图
        if 'performance_summary' in comparison_data:
            summary = comparison_data['performance_summary']
            
            if 'single_image_summary' in summary and 'batch_processing_summary' in summary:
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                # 准备雷达图数据
                categories = ['单图处理速度', '批量吞吐量', '内存效率', '特征一致性']
                
                devices = comparison_data['available_devices']
                device_scores = {}
                
                for device in devices:
                    scores = []
                    
                    # 单图处理速度 (越快越好，取倒数归一化)
                    if device in comparison_data['single_image_performance']:
                        single_time = comparison_data['single_image_performance'][device]['mean_time']
                        speed_score = 1 / single_time
                        scores.append(speed_score)
                    else:
                        scores.append(0)
                    
                    # 批量吞吐量
                    if device in summary['batch_processing_summary']['throughput_by_device']:
                        throughput = summary['batch_processing_summary']['throughput_by_device'][device]
                        scores.append(throughput)
                    else:
                        scores.append(0)
                    
                    # 内存效率 (内存使用越少越好)
                    if device in comparison_data['single_image_performance']:
                        memory_usage = comparison_data['single_image_performance'][device]['memory_usage']['delta_rss_mb']
                        memory_score = max(0, 100 - memory_usage)  # 简单的内存效率评分
                        scores.append(memory_score)
                    else:
                        scores.append(0)
                    
                    # 特征一致性 (如果有的话)
                    if 'consistency_summary' in summary:
                        consistency_score = summary['consistency_summary']['consistency_score'] * 100
                        scores.append(consistency_score)
                    else:
                        scores.append(100)  # 单设备情况下假设完全一致
                    
                    device_scores[device] = scores
                
                # 归一化分数到0-100范围
                all_scores = [score for scores in device_scores.values() for score in scores]
                if all_scores:
                    max_score = max(all_scores)
                    min_score = min(all_scores)
                    score_range = max_score - min_score if max_score != min_score else 1
                    
                    for device in device_scores:
                        device_scores[device] = [(score - min_score) / score_range * 100 
                                               for score in device_scores[device]]
                
                # 绘制雷达图
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # 闭合图形
                
                for i, device in enumerate(devices):
                    values = device_scores[device] + device_scores[device][:1]  # 闭合图形
                    ax.plot(angles, values, 'o-', linewidth=2, label=device, color=self.colors[i])
                    ax.fill(angles, values, alpha=0.25, color=self.colors[i])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 100)
                ax.set_title('设备性能综合对比', size=16, pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
                ax.grid(True)
                
                plt.tight_layout()
                plot_path = self.output_dir / "device_performance_radar.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(str(plot_path))
                print(f"  ✅ 设备性能雷达图: {plot_path}")
        
        return generated_plots
    
    def generate_all_visualizations(self, benchmark_data: Dict[str, Any] = None,
                                  quality_data: Dict[str, Any] = None,
                                  comparison_data: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        生成所有可视化图表
        
        Args:
            benchmark_data: 性能基准测试数据
            quality_data: 特征质量分析数据
            comparison_data: 设备对比数据
            
        Returns:
            Dict[str, List[str]]: 各类图表的文件路径
        """
        print("🚀 开始生成所有可视化图表")
        
        all_plots = {
            'performance_benchmark': [],
            'feature_quality': [],
            'device_comparison': []
        }
        
        # 生成性能基准测试图表
        if benchmark_data:
            all_plots['performance_benchmark'] = self.plot_performance_benchmark(benchmark_data)
        
        # 生成特征质量分析图表
        if quality_data:
            all_plots['feature_quality'] = self.plot_feature_quality_analysis(quality_data)
        
        # 生成设备对比图表
        if comparison_data:
            all_plots['device_comparison'] = self.plot_device_comparison(comparison_data)
        
        # 统计生成的图表数量
        total_plots = sum(len(plots) for plots in all_plots.values())
        print(f"\n🎉 可视化图表生成完成! 共生成 {total_plots} 个图表")
        print(f"📁 输出目录: {self.output_dir}")
        
        return all_plots
    
    def save_plot_index(self, all_plots: Dict[str, List[str]]) -> str:
        """
        保存图表索引文件
        
        Args:
            all_plots: 所有图表路径
            
        Returns:
            str: 索引文件路径
        """
        index_data = {
            'timestamp': datetime.now().isoformat(),
            'total_plots': sum(len(plots) for plots in all_plots.values()),
            'plots_by_category': all_plots,
            'output_directory': str(self.output_dir)
        }
        
        index_path = self.output_dir / "plot_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 图表索引已保存到: {index_path}")
        return str(index_path)


if __name__ == "__main__":
    # 示例用法
    visualizer = TestVisualizationGenerator()
    
    # 这里可以加载实际的测试数据并生成图表
    print("可视化生成器已初始化，可以调用相应方法生成图表")