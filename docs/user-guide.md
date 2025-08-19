# User Guide

This comprehensive guide will help you get started with the Image Feature Extractor and make the most of its capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Quick Start

Get up and running in 5 minutes:

```python
from img_ext import FeatureExtractor

# Initialize the extractor
extractor = FeatureExtractor()

# Extract features from a single image
features = extractor.extract_features('path/to/your/image.jpg')
print(f"Feature vector shape: {features.shape}")  # (512,)

# Extract features from multiple images
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
features = extractor.extract_features(images)
print(f"Batch features shape: {features.shape}")  # (3, 512)
```

## Installation Guide

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: At least 4GB RAM (8GB+ recommended)
- **Storage**: 500MB for dependencies

### Hardware Acceleration

#### Mac M-Series Chips (Recommended)
- **MPS (Metal Performance Shaders)** provides 1.4x speedup
- Automatically detected and used
- Optimal for Mac Studio, MacBook Pro M1/M2/M3

#### NVIDIA GPUs
- **CUDA** support for significant acceleration
- Requires CUDA-compatible PyTorch installation
- Recommended for high-throughput processing

#### CPU Fallback
- Works on all systems
- Slower but reliable
- Good for small-scale processing

### Installation Steps

#### Option 1: Using pip (Recommended)

```bash
# Install the package
pip install img-ext

# Verify installation
python -c "from img_ext import FeatureExtractor; print('Installation successful!')"
```

#### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/your-username/img-ext.git
cd img-ext

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Option 3: Using conda

```bash
# Create a new environment
conda create -n img-ext python=3.9
conda activate img-ext

# Install PyTorch with appropriate backend
# For Mac M-series:
conda install pytorch torchvision -c pytorch

# For CUDA (replace cu118 with your CUDA version):
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the package
pip install img-ext
```

### Verification

Run the test suite to ensure everything is working:

```bash
# Run basic tests
python -m pytest tests/ -v

# Run performance benchmarks
python scripts/benchmark.py

# Generate test report
python scripts/run_tests.py
```

## Basic Usage

### Single Image Processing

```python
from img_ext import FeatureExtractor
from PIL import Image
import numpy as np

# Initialize extractor
extractor = FeatureExtractor()

# Method 1: From file path
features = extractor.extract_features('photo.jpg')

# Method 2: From PIL Image
img = Image.open('photo.jpg')
features = extractor.extract_features(img)

# Method 3: From NumPy array
img_array = np.array(img)
features = extractor.extract_features(img_array)

print(f"Feature vector: {features.shape}")  # (512,)
print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
```

### Batch Processing

```python
# Process multiple images efficiently
images = [
    'folder/image1.jpg',
    'folder/image2.png',
    'folder/image3.bmp'
]

# Extract features with progress bar
features = extractor.extract_features(
    images, 
    batch_size=16,
    show_progress=True
)

print(f"Batch features: {features.shape}")  # (3, 512)

# Access individual feature vectors
for i, feature in enumerate(features):
    print(f"Image {i+1} features: {feature.shape}")
```

### Working with Directories

```python
import os
from pathlib import Path

# Process all images in a directory
image_dir = Path('path/to/images')
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Find all image files
image_paths = [
    str(p) for p in image_dir.iterdir() 
    if p.suffix.lower() in image_extensions
]

print(f"Found {len(image_paths)} images")

# Extract features
features = extractor.extract_features(image_paths)
print(f"Extracted features: {features.shape}")
```

## Advanced Features

### Device Management

```python
# Check available devices
import torch

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU cores: {torch.get_num_threads()}")

# Force specific device
extractor_mps = FeatureExtractor(device='mps')
extractor_cuda = FeatureExtractor(device='cuda')
extractor_cpu = FeatureExtractor(device='cpu')

print(f"Using device: {extractor_mps.device}")
```

### Memory-Efficient Processing

```python
def process_large_dataset(image_paths, chunk_size=1000, batch_size=32):
    """
    Process a large dataset in chunks to manage memory usage.
    """
    extractor = FeatureExtractor()
    all_features = []
    
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(image_paths)-1)//chunk_size + 1}")
        
        # Process chunk
        chunk_features = extractor.extract_features(
            chunk, 
            batch_size=batch_size,
            show_progress=True
        )
        
        all_features.append(chunk_features)
        
        # Optional: Save intermediate results
        np.save(f'features_chunk_{i//chunk_size}.npy', chunk_features)
    
    # Combine all features
    return np.vstack(all_features)

# Usage
large_dataset = ['image_{}.jpg'.format(i) for i in range(10000)]
features = process_large_dataset(large_dataset)
```

### Feature Analysis and Visualization

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Extract features
features = extractor.extract_features(image_paths)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Plot features
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Image Features Visualization (PCA)')
plt.show()

# t-SNE for non-linear visualization
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Image Features Visualization (t-SNE)')
plt.show()
```

## Performance Optimization

### Choosing Optimal Batch Size

```python
import time

def benchmark_batch_sizes(images, batch_sizes=[1, 8, 16, 32, 64]):
    """
    Benchmark different batch sizes to find the optimal one.
    """
    extractor = FeatureExtractor()
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        try:
            features = extractor.extract_features(
                images, 
                batch_size=batch_size,
                show_progress=False
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            images_per_second = len(images) / processing_time
            
            results[batch_size] = {
                'time': processing_time,
                'images_per_second': images_per_second,
                'memory_efficient': True
            }
            
            print(f"Batch size {batch_size:2d}: {processing_time:.2f}s ({images_per_second:.1f} img/s)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[batch_size] = {'error': 'Out of memory'}
                print(f"Batch size {batch_size:2d}: Out of memory")
            else:
                raise e
    
    return results

# Run benchmark
test_images = ['test_image.jpg'] * 100  # 100 copies for testing
results = benchmark_batch_sizes(test_images)
```

### Device-Specific Optimization

```python
def get_optimal_settings():
    """
    Get optimal settings based on available hardware.
    """
    import torch
    
    if torch.backends.mps.is_available():
        return {
            'device': 'mps',
            'batch_size': 16,
            'description': 'Mac M-series with MPS acceleration'
        }
    elif torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 8e9:  # > 8GB
            batch_size = 64
        elif gpu_memory > 4e9:  # > 4GB
            batch_size = 32
        else:
            batch_size = 16
            
        return {
            'device': 'cuda',
            'batch_size': batch_size,
            'description': f'NVIDIA GPU with {gpu_memory/1e9:.1f}GB memory'
        }
    else:
        return {
            'device': 'cpu',
            'batch_size': 8,
            'description': 'CPU processing'
        }

# Use optimal settings
settings = get_optimal_settings()
print(f"Using: {settings['description']}")

extractor = FeatureExtractor(device=settings['device'])
features = extractor.extract_features(
    images, 
    batch_size=settings['batch_size']
)
```

## Best Practices

### 1. Image Preprocessing

```python
from PIL import Image, ImageOps

def preprocess_image(image_path):
    """
    Best practices for image preprocessing.
    """
    # Open image
    img = Image.open(image_path)
    
    # Convert to RGB (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Fix orientation based on EXIF data
    img = ImageOps.exif_transpose(img)
    
    # Optional: Resize very large images to save memory
    max_size = 2048
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return img

# Use preprocessed images
preprocessed_images = [preprocess_image(path) for path in image_paths]
features = extractor.extract_features(preprocessed_images)
```

### 2. Error Handling

```python
def robust_feature_extraction(image_paths, extractor):
    """
    Robust feature extraction with error handling.
    """
    successful_features = []
    failed_images = []
    
    for i, image_path in enumerate(image_paths):
        try:
            feature = extractor.extract_features(image_path)
            successful_features.append(feature)
            
        except FileNotFoundError:
            print(f"Warning: Image not found: {image_path}")
            failed_images.append((i, image_path, "File not found"))
            
        except Exception as e:
            print(f"Warning: Failed to process {image_path}: {e}")
            failed_images.append((i, image_path, str(e)))
    
    if successful_features:
        features = np.vstack(successful_features)
        print(f"Successfully processed {len(successful_features)} images")
        print(f"Failed to process {len(failed_images)} images")
        return features, failed_images
    else:
        raise ValueError("No images were successfully processed")

# Usage
features, failures = robust_feature_extraction(image_paths, extractor)
```

### 3. Caching Results

```python
import pickle
import hashlib
from pathlib import Path

class CachedFeatureExtractor:
    """
    Feature extractor with caching capabilities.
    """
    
    def __init__(self, cache_dir='feature_cache'):
        self.extractor = FeatureExtractor()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, image_path):
        """Generate cache key based on file path and modification time."""
        path = Path(image_path)
        mtime = path.stat().st_mtime
        key_string = f"{path.absolute()}_{mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def extract_features(self, image_path):
        """Extract features with caching."""
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Extract features
        features = self.extractor.extract_features(image_path)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f)
        
        return features

# Usage
cached_extractor = CachedFeatureExtractor()
features = cached_extractor.extract_features('image.jpg')  # Computed
features = cached_extractor.extract_features('image.jpg')  # From cache
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

**Problem**: `RuntimeError: CUDA out of memory` or similar

**Solutions**:
```python
# Reduce batch size
features = extractor.extract_features(images, batch_size=8)

# Use CPU instead
extractor_cpu = FeatureExtractor(device='cpu')

# Process in smaller chunks
chunk_size = 100
for i in range(0, len(images), chunk_size):
    chunk = images[i:i + chunk_size]
    chunk_features = extractor.extract_features(chunk)
```

#### 2. Slow Performance

**Problem**: Processing is slower than expected

**Solutions**:
```python
# Check device usage
print(f"Current device: {extractor.device}")

# Ensure optimal device selection
if torch.backends.mps.is_available():
    extractor = FeatureExtractor(device='mps')
elif torch.cuda.is_available():
    extractor = FeatureExtractor(device='cuda')

# Increase batch size if memory allows
features = extractor.extract_features(images, batch_size=32)

# Preprocess images to standard size
from PIL import Image
standardized_images = []
for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    standardized_images.append(img)
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError` or import issues

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade torch torchvision
pip install --upgrade pillow numpy

# For Mac M-series, ensure MPS support
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

#### 4. Image Loading Issues

**Problem**: Images fail to load or process

**Solutions**:
```python
from PIL import Image
import os

def validate_image(image_path):
    """Validate image file before processing."""
    if not os.path.exists(image_path):
        return False, "File does not exist"
    
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {e}"

# Validate before processing
valid_images = []
for img_path in image_paths:
    is_valid, message = validate_image(img_path)
    if is_valid:
        valid_images.append(img_path)
    else:
        print(f"Skipping {img_path}: {message}")

features = extractor.extract_features(valid_images)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check system information
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")

# Test basic functionality
try:
    extractor = FeatureExtractor()
    print(f"Successfully initialized on device: {extractor.device}")
except Exception as e:
    print(f"Initialization failed: {e}")
```

## Examples

### Example 1: Image Similarity Search

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_images(query_image, image_database, top_k=5):
    """
    Find the most similar images to a query image.
    """
    extractor = FeatureExtractor()
    
    # Extract features
    query_features = extractor.extract_features(query_image)
    db_features = extractor.extract_features(image_database)
    
    # Compute similarities
    similarities = cosine_similarity(
        query_features.reshape(1, -1), 
        db_features
    )[0]
    
    # Get top-k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'image': image_database[idx],
            'similarity': similarities[idx],
            'rank': len(results) + 1
        })
    
    return results

# Usage
query = 'query_image.jpg'
database = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
similar_images = find_similar_images(query, database)

for result in similar_images:
    print(f"Rank {result['rank']}: {result['image']} (similarity: {result['similarity']:.3f})")
```

### Example 2: Image Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_images(image_paths, n_clusters=5):
    """
    Cluster images based on their features.
    """
    extractor = FeatureExtractor()
    
    # Extract features
    features = extractor.extract_features(image_paths)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Organize results
    clusters = {i: [] for i in range(n_clusters)}
    for img_path, label in zip(image_paths, cluster_labels):
        clusters[label].append(img_path)
    
    return clusters, cluster_labels

# Usage
image_paths = ['image_{}.jpg'.format(i) for i in range(50)]
clusters, labels = cluster_images(image_paths, n_clusters=5)

# Print cluster information
for cluster_id, images in clusters.items():
    print(f"Cluster {cluster_id}: {len(images)} images")
    for img in images[:3]:  # Show first 3 images
        print(f"  - {img}")
    if len(images) > 3:
        print(f"  ... and {len(images) - 3} more")
```

### Example 3: Feature-Based Image Recommendation

```python
class ImageRecommender:
    """
    Recommend images based on user preferences.
    """
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.image_features = {}
        self.user_preferences = None
    
    def add_images(self, image_paths):
        """Add images to the recommendation database."""
        features = self.extractor.extract_features(image_paths)
        
        for img_path, feature in zip(image_paths, features):
            self.image_features[img_path] = feature
    
    def learn_preferences(self, liked_images, disliked_images=None):
        """Learn user preferences from liked/disliked images."""
        liked_features = np.array([self.image_features[img] for img in liked_images])
        
        if disliked_images:
            disliked_features = np.array([self.image_features[img] for img in disliked_images])
            # Compute preference as difference between liked and disliked
            self.user_preferences = liked_features.mean(axis=0) - disliked_features.mean(axis=0)
        else:
            # Use average of liked images
            self.user_preferences = liked_features.mean(axis=0)
    
    def recommend(self, n_recommendations=10, exclude_seen=True):
        """Recommend images based on learned preferences."""
        if self.user_preferences is None:
            raise ValueError("No preferences learned. Call learn_preferences() first.")
        
        scores = {}
        for img_path, features in self.image_features.items():
            score = np.dot(features, self.user_preferences)
            scores[img_path] = score
        
        # Sort by score and return top recommendations
        sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_images[:n_recommendations]

# Usage
recommender = ImageRecommender()

# Add images to database
all_images = ['image_{}.jpg'.format(i) for i in range(100)]
recommender.add_images(all_images)

# Learn from user feedback
liked = ['image_5.jpg', 'image_23.jpg', 'image_67.jpg']
disliked = ['image_12.jpg', 'image_45.jpg']
recommender.learn_preferences(liked, disliked)

# Get recommendations
recommendations = recommender.recommend(n_recommendations=10)

print("Recommended images:")
for img_path, score in recommendations:
    print(f"{img_path}: {score:.3f}")
```

These examples demonstrate the versatility of the Image Feature Extractor for various computer vision and machine learning tasks. Adapt them to your specific use case and requirements.

## Next Steps

- Explore the [API Reference](api-reference.md) for detailed technical documentation
- Check out the [examples directory](../examples/) for more code samples
- Read the [Performance Guide](performance-guide.md) for optimization tips
- Join our [community discussions](https://github.com/your-username/img-ext/discussions) for help and ideas