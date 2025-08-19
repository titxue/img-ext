#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级使用示例 - 图像相似度搜索和聚类

本示例展示如何使用FeatureExtractor进行更高级的应用：
1. 图像相似度搜索
2. 图像聚类
3. 特征向量的保存和加载
4. 大规模图像处理
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import time
import json
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor


class ImageSimilaritySearch:
    """图像相似度搜索类"""
    
    def __init__(self, extractor: FeatureExtractor):
        self.extractor = extractor
        self.image_database = []
        self.feature_database = []
        self.metadata_database = []
    
    def add_image(self, image: Image.Image, metadata: Dict = None):
        """添加图像到数据库"""
        features = self.extractor.extract_features(image)
        
        self.image_database.append(image)
        self.feature_database.append(features.cpu().numpy())
        self.metadata_database.append(metadata or {})
        
        return len(self.image_database) - 1
    
    def add_images_batch(self, images: List[Image.Image], metadata_list: List[Dict] = None):
        """批量添加图像到数据库"""
        if metadata_list is None:
            metadata_list = [{}] * len(images)
        
        # 批量提取特征
        batch_features = self.extractor.extract_features_batch(images)
        
        for i, (image, features, metadata) in enumerate(zip(images, batch_features, metadata_list)):
            self.image_database.append(image)
            self.feature_database.append(features.cpu().numpy())
            self.metadata_database.append(metadata)
    
    def search_similar(self, query_image: Image.Image, top_k: int = 5) -> List[Tuple[int, float]]:
        """搜索相似图像"""
        if not self.feature_database:
            return []
        
        # 提取查询图像特征
        query_features = self.extractor.extract_features(query_image).cpu().numpy()
        
        # 计算相似度
        similarities = cosine_similarity([query_features], self.feature_database)[0]
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def save_database(self, filepath: str):
        """保存特征数据库"""
        database = {
            'features': [f.tolist() for f in self.feature_database],
            'metadata': self.metadata_database
        }
        
        with open(filepath, 'w') as f:
            json.dump(database, f)
    
    def load_database(self, filepath: str):
        """加载特征数据库"""
        with open(filepath, 'r') as f:
            database = json.load(f)
        
        self.feature_database = [np.array(f) for f in database['features']]
        self.metadata_database = database['metadata']


def create_diverse_images(num_images: int = 20) -> List[Tuple[Image.Image, str]]:
    """创建多样化的测试图像"""
    images = []
    
    # 创建不同类型的图像
    for i in range(num_images):
        # 创建224x224的图像
        image = Image.new('RGB', (224, 224))
        draw = ImageDraw.Draw(image)
        
        if i < 5:  # 红色系列
            color = (255, i * 50, i * 30)
            draw.rectangle([0, 0, 224, 224], fill=color)
            category = "red_series"
        elif i < 10:  # 蓝色系列
            color = (i * 30, i * 40, 255)
            draw.rectangle([0, 0, 224, 224], fill=color)
            category = "blue_series"
        elif i < 15:  # 几何图形
            color = (i * 15, 255 - i * 10, i * 20)
            draw.ellipse([50, 50, 174, 174], fill=color)
            category = "geometric"
        else:  # 渐变图像
            for y in range(224):
                color_val = int(255 * y / 224)
                draw.line([(0, y), (224, y)], fill=(color_val, 255 - color_val, i * 10))
            category = "gradient"
        
        # 添加文本标识
        try:
            draw.text((10, 10), f"Img {i}", fill=(255, 255, 255))
        except:
            pass  # 如果没有字体，跳过文本
        
        images.append((image, category))
    
    return images


def demonstrate_similarity_search():
    """演示图像相似度搜索"""
    print("=== 图像相似度搜索演示 ===")
    
    # 初始化
    extractor = FeatureExtractor()
    search_engine = ImageSimilaritySearch(extractor)
    
    # 创建测试图像数据库
    print("创建测试图像数据库...")
    test_images = create_diverse_images(20)
    
    # 添加图像到搜索引擎
    start_time = time.time()
    for i, (image, category) in enumerate(test_images):
        metadata = {'id': i, 'category': category, 'name': f'image_{i}'}
        search_engine.add_image(image, metadata)
    
    build_time = time.time() - start_time
    print(f"数据库构建完成，耗时: {build_time:.4f}秒")
    print(f"数据库大小: {len(search_engine.image_database)}张图像")
    
    # 执行相似度搜索
    query_image, query_category = test_images[2]  # 使用第3张图像作为查询
    print(f"\n查询图像类别: {query_category}")
    
    start_time = time.time()
    similar_results = search_engine.search_similar(query_image, top_k=5)
    search_time = time.time() - start_time
    
    print(f"搜索完成，耗时: {search_time:.4f}秒")
    print("\n相似图像结果:")
    for rank, (idx, similarity) in enumerate(similar_results, 1):
        metadata = search_engine.metadata_database[idx]
        print(f"  {rank}. 图像ID: {metadata['id']}, 类别: {metadata['category']}, 相似度: {similarity:.4f}")
    
    # 保存和加载数据库
    db_path = "/tmp/image_features.json"
    search_engine.save_database(db_path)
    print(f"\n特征数据库已保存到: {db_path}")
    
    # 测试加载
    new_search_engine = ImageSimilaritySearch(extractor)
    new_search_engine.load_database(db_path)
    print(f"数据库加载成功，包含{len(new_search_engine.feature_database)}个特征向量")
    
    print()


def demonstrate_image_clustering():
    """演示图像聚类"""
    print("=== 图像聚类演示 ===")
    
    # 初始化
    extractor = FeatureExtractor()
    
    # 创建测试图像
    print("创建测试图像...")
    test_images = create_diverse_images(16)
    images = [img for img, _ in test_images]
    categories = [cat for _, cat in test_images]
    
    # 提取特征
    print("提取图像特征...")
    start_time = time.time()
    features = extractor.extract_features_batch(images, batch_size=8)
    extraction_time = time.time() - start_time
    
    print(f"特征提取完成，耗时: {extraction_time:.4f}秒")
    print(f"特征矩阵形状: {features.shape}")
    
    # 执行聚类
    print("\n执行K-means聚类...")
    n_clusters = 4
    features_np = features.cpu().numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_np)
    
    # 分析聚类结果
    print(f"聚类完成，分为{n_clusters}个簇")
    
    cluster_analysis = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_categories = [categories[idx] for idx in cluster_indices]
        cluster_analysis[i] = {
            'size': len(cluster_indices),
            'images': cluster_indices.tolist(),
            'categories': cluster_categories
        }
    
    print("\n聚类结果分析:")
    for cluster_id, info in cluster_analysis.items():
        print(f"  簇 {cluster_id}: {info['size']}张图像")
        category_counts = {}
        for cat in info['categories']:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"    类别分布: {category_counts}")
        print(f"    图像索引: {info['images']}")
    
    # 计算聚类质量指标
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(features_np, cluster_labels)
    print(f"\n聚类质量 (轮廓系数): {silhouette_avg:.4f}")
    
    print()


def demonstrate_large_scale_processing():
    """演示大规模图像处理"""
    print("=== 大规模图像处理演示 ===")
    
    # 初始化
    extractor = FeatureExtractor()
    
    # 模拟大规模图像处理
    batch_sizes = [1, 8, 16, 32]
    num_images = 100
    
    print(f"测试处理{num_images}张图像，不同批处理大小的性能...")
    
    # 创建测试图像
    test_images = []
    for i in range(num_images):
        # 创建简单的测试图像
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(image_array))
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试批处理大小: {batch_size}")
        
        start_time = time.time()
        
        if batch_size == 1:
            # 单张处理
            all_features = []
            for img in test_images:
                features = extractor.extract_features(img)
                all_features.append(features)
            final_features = torch.stack(all_features)
        else:
            # 批量处理
            final_features = extractor.extract_features_batch(test_images, batch_size=batch_size)
        
        processing_time = time.time() - start_time
        
        results[batch_size] = {
            'time': processing_time,
            'throughput': num_images / processing_time,
            'features_shape': final_features.shape
        }
        
        print(f"  处理时间: {processing_time:.4f}秒")
        print(f"  吞吐量: {num_images/processing_time:.2f}张/秒")
        print(f"  特征形状: {final_features.shape}")
    
    # 性能总结
    print("\n性能总结:")
    best_batch_size = max(results.keys(), key=lambda x: results[x]['throughput'])
    print(f"最佳批处理大小: {best_batch_size} ({results[best_batch_size]['throughput']:.2f}张/秒)")
    
    # 显示性能提升
    single_throughput = results[1]['throughput']
    best_throughput = results[best_batch_size]['throughput']
    speedup = best_throughput / single_throughput
    print(f"相比单张处理提速: {speedup:.2f}x")
    
    print()


def demonstrate_feature_analysis():
    """演示特征向量分析"""
    print("=== 特征向量分析演示 ===")
    
    # 初始化
    extractor = FeatureExtractor()
    
    # 创建不同类型的图像
    test_cases = [
        ("纯红色", Image.new('RGB', (224, 224), (255, 0, 0))),
        ("纯蓝色", Image.new('RGB', (224, 224), (0, 0, 255))),
        ("纯绿色", Image.new('RGB', (224, 224), (0, 255, 0))),
        ("白色", Image.new('RGB', (224, 224), (255, 255, 255))),
        ("黑色", Image.new('RGB', (224, 224), (0, 0, 0))),
    ]
    
    # 添加噪声图像
    noise_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_cases.append(("随机噪声", Image.fromarray(noise_array)))
    
    print("分析不同类型图像的特征向量...")
    
    features_dict = {}
    for name, image in test_cases:
        features = extractor.extract_features(image)
        features_dict[name] = features.cpu().numpy()
        
        print(f"\n{name}:")
        print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  特征均值: {features.mean():.4f}")
        print(f"  特征标准差: {features.std():.4f}")
        print(f"  非零特征数: {(features != 0).sum().item()}/512")
    
    # 计算特征相似度矩阵
    print("\n特征相似度矩阵:")
    names = list(features_dict.keys())
    features_matrix = np.array(list(features_dict.values()))
    similarity_matrix = cosine_similarity(features_matrix)
    
    print("\t" + "\t".join([name[:8] for name in names]))
    for i, name in enumerate(names):
        row = [f"{similarity_matrix[i][j]:.3f}" for j in range(len(names))]
        print(f"{name[:8]}\t" + "\t".join(row))
    
    print()


def main():
    """主函数"""
    print("🚀 高级应用演示 - Mac M系列芯片优化版")
    print("=" * 50)
    
    try:
        # 图像相似度搜索演示
        demonstrate_similarity_search()
        
        # 图像聚类演示
        demonstrate_image_clustering()
        
        # 大规模处理演示
        demonstrate_large_scale_processing()
        
        # 特征分析演示
        demonstrate_feature_analysis()
        
        print("🎉 所有高级演示完成！")
        print("\n💡 应用场景:")
        print("1. 图像搜索引擎 - 以图搜图功能")
        print("2. 图像聚类 - 自动分类和组织")
        print("3. 内容推荐 - 基于视觉相似度的推荐")
        print("4. 重复图像检测 - 找出相似或重复的图像")
        print("5. 图像数据库索引 - 快速检索和匹配")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()