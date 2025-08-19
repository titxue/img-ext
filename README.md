# Image Feature Extractor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/macOS-12.3%2B-lightgrey.svg)](https://www.apple.com/macos/)
[![MPS Accelerated](https://img.shields.io/badge/MPS-Accelerated-green.svg)](https://developer.apple.com/metal/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/titxue/img-ext)

一个高性能的图像特征提取器，专门为Mac M系列芯片优化，支持MPS (Metal Performance Shaders) 加速。基于ResNet-18架构，提供512维特征向量提取，适用于图像相似性分析、聚类和检索任务。

## ✨ 主要特性

- 🚀 **Mac M系列芯片优化**: 原生支持MPS加速，性能提升高达1.4倍
- 🔧 **智能设备检测**: 自动选择最佳计算设备 (MPS > CUDA > CPU)
- 📦 **批处理支持**: 高效处理单张或批量图像，支持自定义批次大小
- 🎯 **ResNet-18特征提取**: 输出标准化的512维特征向量
- 🖼️ **多格式支持**: 支持JPEG、PNG、BMP、TIFF等常见图像格式
- 📊 **性能监控**: 内置性能基准测试和可视化报告
- 🔍 **特征质量分析**: 提供特征稳定性和聚类效果评估

## 📋 系统要求

### 最低要求
- **Python**: 3.8 或更高版本
- **操作系统**: macOS 12.3+ (推荐用于MPS支持)
- **内存**: 至少 4GB RAM
- **存储**: 500MB 可用空间

### 推荐配置
- **硬件**: Mac M1/M2/M3 系列芯片
- **Python**: 3.9+
- **内存**: 8GB+ RAM
- **PyTorch**: 2.0+ (自动安装)

## 🚀 快速安装

### 方法一：从源码安装（推荐）

```bash
# 克隆项目
git clone https://github.com/yourusername/img-ext.git
cd img-ext

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目（开发模式）
pip install -e .
```

### 方法二：直接安装依赖

```bash
# 安装核心依赖
pip install torch torchvision pillow numpy matplotlib seaborn

# 安装开发依赖（可选）
pip install pytest pytest-cov black flake8
```

### 验证安装

```bash
# 运行快速测试
python -c "from feature_extractor import FeatureExtractor; print('安装成功！')"

# 检查MPS支持
python -c "import torch; print(f'MPS可用: {torch.backends.mps.is_available()}')"
```

## 🎯 快速开始

### 基础使用

```python
from feature_extractor import FeatureExtractor
from PIL import Image
import torch

# 初始化特征提取器（自动检测最佳设备）
extractor = FeatureExtractor()
print(f"使用设备: {extractor.device}")

# 加载并处理单张图像
image = Image.open('path/to/your/image.jpg')
features = extractor.extract_features(image)
print(f"特征维度: {features.shape}")  # torch.Size([512])
print(f"特征类型: {features.dtype}")   # torch.float32
```

### 批量处理

```python
# 批量处理多张图像
images = [Image.open(f'image_{i}.jpg') for i in range(10)]
batch_features = extractor.extract_features_batch(images, batch_size=4)
print(f"批量特征维度: {batch_features.shape}")  # torch.Size([10, 512])

# 计算图像相似性
similarity = torch.cosine_similarity(features.unsqueeze(0), batch_features)
print(f"相似性分数: {similarity}")
```

### 高级用法

```python
# 指定设备和配置
extractor = FeatureExtractor(
    device='mps',  # 强制使用MPS
    model_name='resnet18'
)

# 获取设备信息
device_info = extractor.get_device_info()
print(f"设备信息: {device_info}")

# 处理不同尺寸的图像
from torchvision import transforms

# 自定义预处理
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 使用自定义预处理
features = extractor.extract_features(image, transform=custom_transform)
```

## 📚 API 文档

### FeatureExtractor 类

#### 构造函数

```python
FeatureExtractor(device='auto', model_name='resnet18')
```

**参数:**
- `device` (str, 可选): 计算设备选择
  - `'auto'`: 自动选择最佳设备 (默认)
  - `'mps'`: 强制使用MPS (Mac M系列芯片)
  - `'cuda'`: 强制使用CUDA (NVIDIA GPU)
  - `'cpu'`: 强制使用CPU
- `model_name` (str): 模型架构名称，目前支持 `'resnet18'`

#### 主要方法

##### `extract_features(image, transform=None)`

提取单张图像的特征向量。

**参数:**
- `image` (PIL.Image): 输入图像
- `transform` (callable, 可选): 自定义图像预处理函数

**返回:**
- `torch.Tensor`: 形状为 [512] 的特征向量

**示例:**
```python
features = extractor.extract_features(image)
```

##### `extract_features_batch(images, batch_size=32, transform=None)`

批量提取多张图像的特征向量。

**参数:**
- `images` (List[PIL.Image]): 图像列表
- `batch_size` (int): 批处理大小，默认32
- `transform` (callable, 可选): 自定义图像预处理函数

**返回:**
- `torch.Tensor`: 形状为 [N, 512] 的特征矩阵，N为图像数量

**示例:**
```python
batch_features = extractor.extract_features_batch(images, batch_size=16)
```

##### `get_device_info()`

获取当前使用的计算设备详细信息。

**返回:**
- `dict`: 包含设备类型、名称和可用性信息的字典

**示例:**
```python
info = extractor.get_device_info()
print(info)
# 输出: {'device': 'mps', 'available': True, 'name': 'Apple M1 Max'}
```

## 🧪 测试和基准

### 运行测试套件

```bash
# 运行所有测试
pytest

# 运行特定测试模块
pytest tests/test_feature_extractor.py
pytest tests/test_mps_support.py

# 生成覆盖率报告
pytest --cov=feature_extractor --cov-report=html
```

### 性能基准测试

```bash
# 运行完整的性能基准测试
python test_comprehensive.py

# 查看生成的测试报告
open test_results/comprehensive_test_report.html
```

### 测试结果示例

在Mac Studio M1 Max上的性能测试结果：

| 设备类型 | 单张图像处理 | 批量处理(32张) | 内存使用 | 加速比 |
|----------|-------------|----------------|----------|--------|
| **MPS**  | ~12ms       | ~180ms         | 2.1GB    | 1.4x   |
| **CPU**  | ~17ms       | ~250ms         | 1.8GB    | 1.0x   |

**特征质量指标:**
- 特征稳定性: 95%
- 聚类效果: 0.75
- 相似性分析: 0.85

## 📊 可视化和报告

项目包含完整的测试报告生成功能：

```bash
# 生成完整测试报告
python test_comprehensive.py

# 报告包含以下内容：
# - 性能基准对比图表
# - 特征质量分析
# - 设备性能对比
# - 详细的测试数据
```

生成的报告文件：
- `test_results/comprehensive_test_report.html` - 主要测试报告
- `test_results/performance_benchmark.png` - 性能对比图表
- `test_results/feature_quality_analysis.png` - 特征质量分析
- `test_results/device_comparison.png` - 设备性能对比
- `test_results/test_summary.png` - 测试总结图表

## 🔧 配置和优化

### 环境变量配置

```bash
# 设置默认设备
export IMG_EXT_DEVICE=mps

# 设置默认批次大小
export IMG_EXT_BATCH_SIZE=16

# 启用详细日志
export IMG_EXT_VERBOSE=1
```

### 性能优化建议

1. **Mac M系列芯片用户**:
   - 确保使用MPS设备以获得最佳性能
   - 推荐批次大小：16-32

2. **内存优化**:
   - 大批量处理时适当减小batch_size
   - 及时释放不需要的图像对象

3. **CPU用户**:
   - 考虑使用多线程处理
   - 适当增加批次大小以提高吞吐量

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 快速贡献步骤

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e .[dev]

# 安装pre-commit钩子
pre-commit install

# 运行代码格式化
black .
flake8 .
```

## 📝 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [torchvision](https://pytorch.org/vision/) - 计算机视觉工具包
- [Apple Metal](https://developer.apple.com/metal/) - MPS加速支持
- [ResNet](https://arxiv.org/abs/1512.03385) - 网络架构

## 📞 支持和联系

- 🐛 **问题报告**: [GitHub Issues](https://github.com/yourusername/img-ext/issues)
- 💡 **功能请求**: [GitHub Discussions](https://github.com/yourusername/img-ext/discussions)
- 📧 **邮件联系**: your.email@example.com

---

<div align="center">
  <p>如果这个项目对您有帮助，请给我们一个 ⭐️</p>
  <p>Made with ❤️ for the Mac M-series community</p>
</div>