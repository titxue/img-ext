# API Reference

This document provides detailed information about the Image Feature Extractor API.

## Table of Contents

- [FeatureExtractor Class](#featureextractor-class)
- [Methods](#methods)
- [Parameters](#parameters)
- [Return Values](#return-values)
- [Exceptions](#exceptions)
- [Examples](#examples)

## FeatureExtractor Class

The main class for extracting features from images using a pre-trained ResNet-18 model.

### Constructor

```python
FeatureExtractor(device=None, model_name='resnet18', pretrained=True)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` or `torch.device` | `None` | Target device for computation. If None, automatically selects the best available device (MPS > CUDA > CPU) |
| `model_name` | `str` | `'resnet18'` | Name of the model architecture. Currently supports 'resnet18' |
| `pretrained` | `bool` | `True` | Whether to use pre-trained weights from ImageNet |

#### Device Selection Logic

When `device=None`, the constructor automatically selects the optimal device:

1. **MPS (Metal Performance Shaders)** - For Mac M-series chips
2. **CUDA** - For NVIDIA GPUs
3. **CPU** - Fallback option

```python
# Automatic device selection
extractor = FeatureExtractor()  # Uses best available device

# Manual device specification
extractor = FeatureExtractor(device='mps')    # Force MPS
extractor = FeatureExtractor(device='cuda')   # Force CUDA
extractor = FeatureExtractor(device='cpu')    # Force CPU
```

### Properties

#### `device`
- **Type**: `torch.device`
- **Description**: Current device being used for computations
- **Read-only**: Yes

```python
print(f"Using device: {extractor.device}")
# Output: Using device: mps
```

#### `model`
- **Type**: `torch.nn.Module`
- **Description**: The underlying PyTorch model
- **Read-only**: Yes

## Methods

### extract_features

Extracts feature vectors from input images.

```python
extract_features(images, batch_size=32, show_progress=True)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `str`, `Path`, `PIL.Image`, `np.ndarray`, or `list` | Required | Input image(s). Can be a single image or a list of images |
| `batch_size` | `int` | `32` | Number of images to process in each batch |
| `show_progress` | `bool` | `True` | Whether to display a progress bar during batch processing |

#### Supported Input Types

1. **File Path** (string or Path object)
   ```python
   features = extractor.extract_features('path/to/image.jpg')
   features = extractor.extract_features(Path('path/to/image.jpg'))
   ```

2. **PIL Image**
   ```python
   from PIL import Image
   img = Image.open('path/to/image.jpg')
   features = extractor.extract_features(img)
   ```

3. **NumPy Array**
   ```python
   import numpy as np
   img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
   features = extractor.extract_features(img_array)
   ```

4. **List of Mixed Types**
   ```python
   images = [
       'image1.jpg',
       PIL.Image.open('image2.png'),
       np.array(...),
       Path('image3.bmp')
   ]
   features = extractor.extract_features(images)
   ```

#### Return Values

| Input Type | Return Type | Shape | Description |
|------------|-------------|-------|-------------|
| Single image | `np.ndarray` | `(512,)` | 1D feature vector |
| Multiple images | `np.ndarray` | `(N, 512)` | 2D array where N is the number of images |

#### Image Processing Pipeline

1. **Loading**: Images are loaded and converted to RGB format
2. **Resizing**: Images are resized to 224×224 pixels
3. **Normalization**: Pixel values are normalized using ImageNet statistics
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
4. **Feature Extraction**: Features are extracted from the last pooling layer
5. **Post-processing**: Features are converted to NumPy arrays and normalized

## Exceptions

### FileNotFoundError
Raised when an image file path doesn't exist.

```python
try:
    features = extractor.extract_features('nonexistent.jpg')
except FileNotFoundError as e:
    print(f"Image not found: {e}")
```

### ValueError
Raised for invalid input types or corrupted images.

```python
try:
    features = extractor.extract_features("invalid_input")
except ValueError as e:
    print(f"Invalid input: {e}")
```

### RuntimeError
Raised for device-related issues or model loading problems.

```python
try:
    extractor = FeatureExtractor(device='invalid_device')
except RuntimeError as e:
    print(f"Device error: {e}")
```

## Performance Considerations

### Batch Processing

For optimal performance when processing multiple images:

```python
# Efficient batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]
features = extractor.extract_features(images, batch_size=32)

# Less efficient individual processing
features = []
for img in images:
    feature = extractor.extract_features(img)
    features.append(feature)
features = np.array(features)
```

### Memory Management

- **Batch Size**: Adjust based on available memory
  - GPU: 32-64 images per batch
  - CPU: 8-16 images per batch
  - MPS: 16-32 images per batch

- **Large Datasets**: Process in chunks to avoid memory overflow

```python
def process_large_dataset(image_paths, extractor, chunk_size=1000):
    all_features = []
    
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i:i + chunk_size]
        features = extractor.extract_features(chunk)
        all_features.append(features)
    
    return np.vstack(all_features)
```

### Device-Specific Optimizations

#### Mac M-Series (MPS)
```python
# Optimal settings for Mac M-series chips
extractor = FeatureExtractor(device='mps')
features = extractor.extract_features(images, batch_size=16)
```

#### NVIDIA GPU (CUDA)
```python
# Optimal settings for NVIDIA GPUs
extractor = FeatureExtractor(device='cuda')
features = extractor.extract_features(images, batch_size=64)
```

#### CPU
```python
# Optimal settings for CPU processing
extractor = FeatureExtractor(device='cpu')
features = extractor.extract_features(images, batch_size=8)
```

## Advanced Usage

### Custom Preprocessing

For custom image preprocessing, you can access the internal transform:

```python
from torchvision import transforms

# Access the default transform
default_transform = extractor.transform

# Create custom transform
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Feature Analysis

```python
# Extract features
features = extractor.extract_features(images)

# Analyze feature statistics
print(f"Feature shape: {features.shape}")
print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
print(f"Feature mean: {features.mean():.3f}")
print(f"Feature std: {features.std():.3f}")

# Compute similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(features)
```

### Integration with Machine Learning

```python
# Use features for clustering
from sklearn.cluster import KMeans

features = extractor.extract_features(images)
kmeans = KMeans(n_clusters=5)
cluster_labels = kmeans.fit_predict(features)

# Use features for similarity search
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(features)
distances, indices = nn.kneighbors(query_features)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Process images in smaller chunks
   - Use CPU instead of GPU for very large images

2. **Slow Performance**
   - Ensure you're using the optimal device (MPS/CUDA)
   - Increase batch size if memory allows
   - Check image sizes (very large images slow down processing)

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check PyTorch installation for your platform
   - Verify torchvision compatibility

### Debug Mode

```python
# Enable debug information
import logging
logging.basicConfig(level=logging.DEBUG)

# Check device availability
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.8 | 3.9+ |
| PyTorch | 2.0.0 | 2.1.0+ |
| torchvision | 0.15.0 | 0.16.0+ |
| PIL/Pillow | 8.0.0 | 10.0.0+ |
| NumPy | 1.20.0 | 1.24.0+ |

For the latest compatibility information, see the [requirements.txt](../requirements.txt) file.