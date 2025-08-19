#!/usr/bin/env python3
"""
图片爬虫脚本 - 从网站地图下载产品图片
用于创建真实测试数据集
"""

import os
import sys
import requests
import time
from urllib.parse import urlparse
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List, Optional
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageCrawler:
    """产品图片爬虫类"""
    
    def __init__(self, output_dir: str = "tests/test_data/real_images", delay: float = 1.0):
        """
        初始化爬虫
        
        Args:
            output_dir: 图片保存目录
            delay: 请求间隔时间（秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        解析网站地图XML，提取图片URL
        
        Args:
            sitemap_url: 网站地图URL
            
        Returns:
            图片URL列表
        """
        logger.info(f"正在解析网站地图: {sitemap_url}")
        
        try:
            response = self.session.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # 解析XML
            root = ET.fromstring(response.content)
            
            # 定义命名空间
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'image': 'http://www.google.com/schemas/sitemap-image/1.1'
            }
            
            # 提取所有图片URL
            image_urls = []
            for url_elem in root.findall('.//sitemap:url', namespaces):
                for image_elem in url_elem.findall('.//image:image', namespaces):
                    image_loc = image_elem.find('image:loc', namespaces)
                    if image_loc is not None and image_loc.text:
                        image_urls.append(image_loc.text.strip())
            
            logger.info(f"找到 {len(image_urls)} 个图片URL")
            return image_urls
            
        except requests.RequestException as e:
            logger.error(f"获取网站地图失败: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"解析XML失败: {e}")
            return []
    
    def download_image(self, image_url: str) -> Optional[str]:
        """
        下载单个图片
        
        Args:
            image_url: 图片URL
            
        Returns:
            保存的文件路径，失败返回None
        """
        try:
            # 解析URL获取文件名
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            
            # 如果没有扩展名，尝试从URL参数中获取
            if not filename or '.' not in filename:
                filename = f"image_{hash(image_url) % 100000}.jpg"
            
            # 确保文件名安全
            filename = "".join(c for c in filename if c.isalnum() or c in '.-_')
            if not filename:
                filename = f"image_{hash(image_url) % 100000}.jpg"
            
            file_path = self.output_dir / filename
            
            # 如果文件已存在，跳过
            if file_path.exists():
                logger.debug(f"文件已存在，跳过: {filename}")
                return str(file_path)
            
            # 下载图片
            response = self.session.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"非图片内容类型: {content_type} for {image_url}")
                return None
            
            # 保存文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.debug(f"下载成功: {filename}")
            return str(file_path)
            
        except requests.RequestException as e:
            logger.error(f"下载图片失败 {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"保存图片失败 {image_url}: {e}")
            return None
    
    def crawl_images(self, sitemap_url: str, max_images: Optional[int] = None) -> List[str]:
        """
        爬取所有图片
        
        Args:
            sitemap_url: 网站地图URL
            max_images: 最大下载数量，None表示下载所有
            
        Returns:
            成功下载的文件路径列表
        """
        # 解析网站地图
        image_urls = self.parse_sitemap(sitemap_url)
        if not image_urls:
            logger.error("没有找到图片URL")
            return []
        
        # 限制下载数量
        if max_images and len(image_urls) > max_images:
            image_urls = image_urls[:max_images]
            logger.info(f"限制下载数量为 {max_images}")
        
        # 下载图片
        downloaded_files = []
        failed_count = 0
        
        logger.info(f"开始下载 {len(image_urls)} 个图片...")
        
        with tqdm(total=len(image_urls), desc="下载图片") as pbar:
            for i, image_url in enumerate(image_urls):
                # 请求间隔
                if i > 0:
                    time.sleep(self.delay)
                
                file_path = self.download_image(image_url)
                if file_path:
                    downloaded_files.append(file_path)
                else:
                    failed_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': len(downloaded_files),
                    '失败': failed_count
                })
        
        logger.info(f"下载完成: 成功 {len(downloaded_files)}, 失败 {failed_count}")
        return downloaded_files

def main():
    """主函数"""
    sitemap_url = "https://jettashop.com/sitemap_products_1.xml?from=7351916986606&to=8924092104942"
    
    # 创建爬虫实例
    crawler = ImageCrawler(delay=0.5)  # 0.5秒间隔，避免过于频繁的请求
    
    # 开始爬取（限制前50张图片用于测试）
    downloaded_files = crawler.crawl_images(sitemap_url, max_images=50)
    
    if downloaded_files:
        print(f"\n✅ 成功下载 {len(downloaded_files)} 个图片文件")
        print(f"📁 保存位置: {crawler.output_dir}")
        print("\n前5个文件:")
        for file_path in downloaded_files[:5]:
            print(f"  - {os.path.basename(file_path)}")
    else:
        print("❌ 没有成功下载任何图片")
        sys.exit(1)

if __name__ == "__main__":
    main()