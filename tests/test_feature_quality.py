#!/usr/bin/env python3
"""
特征质量分析模块

分析内容:
- 特征分布统计
- 特征相似性分析
- 聚类效果评估
- 特征降维可视化
- 特征稳定性测试
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import stats
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import FeatureExtractor


class FeatureQualityAnalyzer:
    """
    特征质量分析器
    """
    
    def __init__(self, test_images_dir: str):
        """
        初始化特征质量分析器
        
        Args:
            test_images_dir: 测试图片目录
        """
        self.test_images_dir = Path(test_images_dir)
        self.test_images = self._load_test_images()
        self.features = None
        self.image_names = None
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
    
    def extract_all_features(self, extractor: FeatureExtractor) -> np.ndarray:
        """
        提取所有图片的特征
        
        Args:
            extractor: 特征提取器
            
        Returns:
            np.ndarray: 特征矩阵 [N, 512]
        """
        print(f"\n🔍 提取所有图片特征 (设备: {extractor.device})")
        
        # 批量提取特征
        self.features = extractor.extract_features_batch(self.test_images, batch_size=16)
        self.image_names = [Path(img).name for img in self.test_images]
        
        print(f"✅ 提取完成，特征矩阵形状: {self.features.shape}")
        return self.features
    
    def analyze_feature_distribution(self) -> Dict[str, Any]:
        """
        分析特征分布统计
        
        Returns:
            Dict[str, Any]: 特征分布分析结果
        """
        print("\n📊 分析特征分布")
        
        if self.features is None:
            raise ValueError("请先提取特征")
        
        # 基本统计信息
        mean_features = np.mean(self.features, axis=0)
        std_features = np.std(self.features, axis=0)
        min_features = np.min(self.features, axis=0)
        max_features = np.max(self.features, axis=0)
        
        # 整体统计
        overall_stats = {
            'mean': np.mean(self.features),
            'std': np.std(self.features),
            'min': np.min(self.features),
            'max': np.max(self.features),
            'median': np.median(self.features),
            'q25': np.percentile(self.features, 25),
            'q75': np.percentile(self.features, 75)
        }
        
        # 特征维度统计
        dimension_stats = {
            'mean_per_dim': mean_features,
            'std_per_dim': std_features,
            'min_per_dim': min_features,
            'max_per_dim': max_features,
            'range_per_dim': max_features - min_features
        }
        
        # 稀疏性分析
        zero_ratio = np.mean(self.features == 0)
        near_zero_ratio = np.mean(np.abs(self.features) < 1e-6)
        
        # 正态性检验 (对前10个维度进行检验)
        normality_tests = []
        for i in range(min(10, self.features.shape[1])):
            stat, p_value = stats.normaltest(self.features[:, i])
            normality_tests.append({
                'dimension': i,
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            })
        
        results = {
            'overall_stats': overall_stats,
            'dimension_stats': dimension_stats,
            'sparsity': {
                'zero_ratio': zero_ratio,
                'near_zero_ratio': near_zero_ratio
            },
            'normality_tests': normality_tests
        }
        
        print(f"  📈 整体均值: {overall_stats['mean']:.4f} ± {overall_stats['std']:.4f}")
        print(f"  🕳️  稀疏度: {zero_ratio:.2%} (零值), {near_zero_ratio:.2%} (近零值)")
        print(f"  📊 正态分布维度: {sum(1 for t in normality_tests if t['is_normal'])}/{len(normality_tests)}")
        
        return results
    
    def analyze_feature_similarity(self) -> Dict[str, Any]:
        """
        分析特征相似性
        
        Returns:
            Dict[str, Any]: 相似性分析结果
        """
        print("\n🔍 分析特征相似性")
        
        if self.features is None:
            raise ValueError("请先提取特征")
        
        # 余弦相似度矩阵
        cosine_sim_matrix = cosine_similarity(self.features)
        
        # 欧氏距离矩阵
        euclidean_dist_matrix = euclidean_distances(self.features)
        
        # 相似度统计 (排除对角线)
        mask = ~np.eye(cosine_sim_matrix.shape[0], dtype=bool)
        cosine_similarities = cosine_sim_matrix[mask]
        euclidean_distances_flat = euclidean_dist_matrix[mask]
        
        # 找出最相似和最不相似的图片对
        max_sim_idx = np.unravel_index(np.argmax(cosine_sim_matrix * (1 - np.eye(len(cosine_sim_matrix)))), 
                                      cosine_sim_matrix.shape)
        min_sim_idx = np.unravel_index(np.argmin(cosine_sim_matrix * (1 - np.eye(len(cosine_sim_matrix)))), 
                                      cosine_sim_matrix.shape)
        
        results = {
            'cosine_similarity': {
                'matrix': cosine_sim_matrix,
                'mean': np.mean(cosine_similarities),
                'std': np.std(cosine_similarities),
                'min': np.min(cosine_similarities),
                'max': np.max(cosine_similarities),
                'median': np.median(cosine_similarities)
            },
            'euclidean_distance': {
                'matrix': euclidean_dist_matrix,
                'mean': np.mean(euclidean_distances_flat),
                'std': np.std(euclidean_distances_flat),
                'min': np.min(euclidean_distances_flat),
                'max': np.max(euclidean_distances_flat),
                'median': np.median(euclidean_distances_flat)
            },
            'most_similar_pair': {
                'images': (self.image_names[max_sim_idx[0]], self.image_names[max_sim_idx[1]]),
                'similarity': cosine_sim_matrix[max_sim_idx]
            },
            'least_similar_pair': {
                'images': (self.image_names[min_sim_idx[0]], self.image_names[min_sim_idx[1]]),
                'similarity': cosine_sim_matrix[min_sim_idx]
            }
        }
        
        print(f"  📊 平均余弦相似度: {results['cosine_similarity']['mean']:.4f} ± {results['cosine_similarity']['std']:.4f}")
        print(f"  📏 平均欧氏距离: {results['euclidean_distance']['mean']:.4f} ± {results['euclidean_distance']['std']:.4f}")
        print(f"  🔗 最相似图片对: {results['most_similar_pair']['images'][0]} & {results['most_similar_pair']['images'][1]} (相似度: {results['most_similar_pair']['similarity']:.4f})")
        
        return results
    
    def analyze_clustering_quality(self, n_clusters_range: List[int] = None) -> Dict[str, Any]:
        """
        分析聚类效果
        
        Args:
            n_clusters_range: 聚类数量范围
            
        Returns:
            Dict[str, Any]: 聚类质量分析结果
        """
        print("\n🎯 分析聚类效果")
        
        if self.features is None:
            raise ValueError("请先提取特征")
        
        if n_clusters_range is None:
            n_clusters_range = list(range(2, min(11, len(self.features) // 2)))
        
        clustering_results = []
        
        for n_clusters in n_clusters_range:
            print(f"  测试 {n_clusters} 个聚类...")
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.features)
            
            # 计算聚类质量指标
            silhouette_avg = silhouette_score(self.features, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.features, cluster_labels)
            
            # 计算簇内距离和簇间距离
            inertia = kmeans.inertia_
            
            clustering_results.append({
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'inertia': inertia,
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_
            })
            
            print(f"    轮廓系数: {silhouette_avg:.4f}")
            print(f"    Calinski-Harabasz指数: {calinski_harabasz:.2f}")
        
        # 找出最佳聚类数
        best_silhouette = max(clustering_results, key=lambda x: x['silhouette_score'])
        best_calinski = max(clustering_results, key=lambda x: x['calinski_harabasz_score'])
        
        results = {
            'clustering_results': clustering_results,
            'best_by_silhouette': {
                'n_clusters': best_silhouette['n_clusters'],
                'score': best_silhouette['silhouette_score']
            },
            'best_by_calinski': {
                'n_clusters': best_calinski['n_clusters'],
                'score': best_calinski['calinski_harabasz_score']
            }
        }
        
        print(f"  🏆 最佳聚类数 (轮廓系数): {best_silhouette['n_clusters']} (得分: {best_silhouette['silhouette_score']:.4f})")
        print(f"  🏆 最佳聚类数 (Calinski-Harabasz): {best_calinski['n_clusters']} (得分: {best_calinski['calinski_harabasz_score']:.2f})")
        
        return results
    
    def analyze_dimensionality_reduction(self) -> Dict[str, Any]:
        """
        分析降维效果
        
        Returns:
            Dict[str, Any]: 降维分析结果
        """
        print("\n📉 分析降维效果")
        
        if self.features is None:
            raise ValueError("请先提取特征")
        
        # PCA降维
        print("  执行PCA降维...")
        pca = PCA(n_components=min(50, self.features.shape[0] - 1))
        pca_features = pca.fit_transform(self.features)
        
        # 计算累积方差解释比
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # 找出解释95%方差所需的主成分数
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
        
        # t-SNE降维 (降到2D用于可视化)
        print("  执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.features) - 1))
        tsne_features = tsne.fit_transform(self.features)
        
        results = {
            'pca': {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_ratio': cumulative_variance_ratio,
                'n_components_95': n_components_95,
                'n_components_99': n_components_99,
                'features_2d': pca_features[:, :2],
                'features_reduced': pca_features
            },
            'tsne': {
                'features_2d': tsne_features
            }
        }
        
        print(f"  📊 前10个主成分解释方差: {pca.explained_variance_ratio_[:10].sum():.2%}")
        print(f"  🎯 95%方差需要 {n_components_95} 个主成分")
        print(f"  🎯 99%方差需要 {n_components_99} 个主成分")
        
        return results
    
    def test_feature_stability(self, extractor: FeatureExtractor, num_runs: int = 3) -> Dict[str, Any]:
        """
        测试特征稳定性
        
        Args:
            extractor: 特征提取器
            num_runs: 运行次数
            
        Returns:
            Dict[str, Any]: 稳定性测试结果
        """
        print(f"\n🔄 测试特征稳定性 (运行 {num_runs} 次)")
        
        # 选择前5张图片进行稳定性测试
        test_images = self.test_images[:min(5, len(self.test_images))]
        
        stability_results = []
        
        for img_path in test_images:
            img_name = Path(img_path).name
            print(f"  测试图片: {img_name}")
            
            # 多次提取同一张图片的特征
            features_runs = []
            for run in range(num_runs):
                features = extractor.extract_features(img_path)
                features_runs.append(features)
            
            # 计算特征的稳定性
            features_array = np.array(features_runs)
            mean_features = np.mean(features_array, axis=0)
            std_features = np.std(features_array, axis=0)
            
            # 计算变异系数 (CV = std/mean)
            cv = np.divide(std_features, np.abs(mean_features), 
                          out=np.zeros_like(std_features), where=mean_features!=0)
            
            stability_results.append({
                'image_name': img_name,
                'mean_features': mean_features,
                'std_features': std_features,
                'coefficient_of_variation': cv,
                'mean_cv': np.mean(cv),
                'max_cv': np.max(cv),
                'stable_dimensions': np.sum(cv < 0.01)  # CV < 1%的维度数
            })
            
            print(f"    平均变异系数: {np.mean(cv):.6f}")
            print(f"    稳定维度数: {np.sum(cv < 0.01)}/{len(cv)}")
        
        # 整体稳定性统计
        overall_cv = np.mean([result['mean_cv'] for result in stability_results])
        overall_stable_dims = np.mean([result['stable_dimensions'] for result in stability_results])
        
        results = {
            'num_runs': num_runs,
            'stability_results': stability_results,
            'overall_stability': {
                'mean_coefficient_of_variation': overall_cv,
                'mean_stable_dimensions': overall_stable_dims,
                'stability_ratio': overall_stable_dims / self.features.shape[1] if self.features is not None else 0
            }
        }
        
        print(f"  📊 整体平均变异系数: {overall_cv:.6f}")
        print(f"  🎯 平均稳定维度比例: {results['overall_stability']['stability_ratio']:.2%}")
        
        return results
    
    def run_comprehensive_analysis(self, extractor: FeatureExtractor) -> Dict[str, Any]:
        """
        运行全面的特征质量分析
        
        Args:
            extractor: 特征提取器
            
        Returns:
            Dict[str, Any]: 完整的分析结果
        """
        print("🚀 开始全面特征质量分析")
        print(f"📊 测试图片数量: {len(self.test_images)}")
        print(f"🖥️  使用设备: {extractor.device}")
        
        # 提取所有特征
        self.extract_all_features(extractor)
        
        # 运行各项分析
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(extractor.device),
            'test_images_count': len(self.test_images),
            'feature_shape': self.features.shape,
            'distribution_analysis': self.analyze_feature_distribution(),
            'similarity_analysis': self.analyze_feature_similarity(),
            'clustering_analysis': self.analyze_clustering_quality(),
            'dimensionality_analysis': self.analyze_dimensionality_reduction(),
            'stability_analysis': self.test_feature_stability(extractor)
        }
        
        # 保存结果
        self.results = all_results
        return all_results
    
    def save_results(self, output_path: str = None) -> str:
        """
        保存分析结果到JSON文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            str: 保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_quality_analysis_{timestamp}.json"
        
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
        
        print(f"📁 分析结果已保存到: {output_path}")
        return output_path


if __name__ == "__main__":
    # 运行特征质量分析
    test_images_dir = "test_data/real_images"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ 测试图片目录不存在: {test_images_dir}")
        exit(1)
    
    # 创建分析器
    analyzer = FeatureQualityAnalyzer(test_images_dir)
    
    # 创建特征提取器
    extractor = FeatureExtractor(device='auto')
    
    # 运行分析
    results = analyzer.run_comprehensive_analysis(extractor)
    
    # 保存结果
    analyzer.save_results()
    
    print("\n🎉 特征质量分析完成!")