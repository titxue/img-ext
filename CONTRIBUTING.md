# 贡献指南

感谢您对 Image Feature Extractor 项目的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 错误报告
- 💡 功能建议
- 📝 文档改进
- 🔧 代码贡献
- 🧪 测试用例
- 🌐 翻译工作

## 📋 目录

- [开始之前](#开始之前)
- [开发环境设置](#开发环境设置)
- [贡献流程](#贡献流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [提交规范](#提交规范)
- [Pull Request 指南](#pull-request-指南)
- [问题报告](#问题报告)
- [功能请求](#功能请求)

## 🚀 开始之前

### 行为准则

参与本项目即表示您同意遵守我们的行为准则：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 专注于对社区最有利的事情
- 对其他社区成员表现出同理心

### 贡献类型

我们特别欢迎以下类型的贡献：

1. **性能优化**: 特别是针对Mac M系列芯片的MPS加速优化
2. **新功能**: 新的特征提取方法或模型支持
3. **文档改进**: API文档、使用示例、教程等
4. **测试覆盖**: 单元测试、集成测试、性能测试
5. **错误修复**: 任何发现的bug或问题

## 🛠️ 开发环境设置

### 1. Fork 和克隆项目

```bash
# Fork 项目到您的GitHub账户，然后克隆
git clone https://github.com/yourusername/img-ext.git
cd img-ext

# 添加上游仓库
git remote add upstream https://github.com/originalowner/img-ext.git
```

### 2. 创建虚拟环境

```bash
# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装开发依赖

```bash
# 安装项目依赖（开发模式）
pip install -e .[dev]

# 或者手动安装所有依赖
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 pre-commit
```

### 4. 设置开发工具

```bash
# 安装pre-commit钩子
pre-commit install

# 验证安装
python -c "from feature_extractor import FeatureExtractor; print('开发环境设置成功！')"
```

### 5. 运行测试确保环境正常

```bash
# 运行所有测试
pytest

# 运行性能基准测试
python test_comprehensive.py
```

## 🔄 贡献流程

### 1. 创建功能分支

```bash
# 确保主分支是最新的
git checkout main
git pull upstream main

# 创建新的功能分支
git checkout -b feature/your-feature-name
# 或者修复bug
git checkout -b fix/bug-description
```

### 2. 进行开发

- 编写代码时请遵循[代码规范](#代码规范)
- 为新功能添加相应的测试
- 更新相关文档
- 确保所有测试通过

### 3. 提交更改

```bash
# 添加更改的文件
git add .

# 提交更改（遵循提交规范）
git commit -m "feat: add new feature description"
```

### 4. 推送并创建Pull Request

```bash
# 推送到您的fork
git push origin feature/your-feature-name

# 然后在GitHub上创建Pull Request
```

## 📝 代码规范

### Python代码风格

我们使用以下工具来保持代码质量：

- **Black**: 代码格式化
- **Flake8**: 代码检查
- **isort**: 导入排序

```bash
# 格式化代码
black .

# 检查代码风格
flake8 .

# 排序导入
isort .
```

### 代码风格要求

1. **命名规范**:
   - 类名使用 `PascalCase`
   - 函数和变量使用 `snake_case`
   - 常量使用 `UPPER_CASE`
   - 私有方法以单下划线开头 `_private_method`

2. **文档字符串**:
   ```python
   def extract_features(self, image, transform=None):
       """
       提取单张图像的特征向量。
       
       Args:
           image (PIL.Image): 输入图像
           transform (callable, optional): 自定义图像预处理函数
           
       Returns:
           torch.Tensor: 形状为 [512] 的特征向量
           
       Raises:
           ValueError: 当图像格式不支持时
       """
   ```

3. **类型提示**:
   ```python
   from typing import List, Optional, Union
   from PIL import Image
   import torch
   
   def extract_features_batch(
       self, 
       images: List[Image.Image], 
       batch_size: int = 32,
       transform: Optional[callable] = None
   ) -> torch.Tensor:
   ```

4. **错误处理**:
   ```python
   try:
       features = self.model(tensor)
   except RuntimeError as e:
       raise RuntimeError(f"特征提取失败: {e}") from e
   ```

## 🧪 测试要求

### 测试类型

1. **单元测试**: 测试单个函数或方法
2. **集成测试**: 测试组件间的交互
3. **性能测试**: 测试性能基准
4. **设备测试**: 测试不同设备（MPS、CUDA、CPU）的兼容性

### 编写测试

```python
import pytest
import torch
from PIL import Image
from feature_extractor import FeatureExtractor

class TestFeatureExtractor:
    def test_extract_features_single_image(self):
        """测试单张图像特征提取"""
        extractor = FeatureExtractor(device='cpu')
        image = Image.new('RGB', (224, 224), color='red')
        
        features = extractor.extract_features(image)
        
        assert features.shape == torch.Size([512])
        assert features.dtype == torch.float32
    
    def test_extract_features_batch(self):
        """测试批量图像特征提取"""
        extractor = FeatureExtractor(device='cpu')
        images = [Image.new('RGB', (224, 224), color='red') for _ in range(5)]
        
        features = extractor.extract_features_batch(images, batch_size=2)
        
        assert features.shape == torch.Size([5, 512])
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), 
                        reason="MPS not available")
    def test_mps_device(self):
        """测试MPS设备支持"""
        extractor = FeatureExtractor(device='mps')
        assert extractor.device.type == 'mps'
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_feature_extractor.py

# 运行特定测试方法
pytest tests/test_feature_extractor.py::TestFeatureExtractor::test_extract_features_single_image

# 生成覆盖率报告
pytest --cov=feature_extractor --cov-report=html

# 运行性能测试
pytest tests/test_performance.py -v
```

### 测试覆盖率要求

- 新功能的测试覆盖率应达到 **90%** 以上
- 核心功能的测试覆盖率应达到 **95%** 以上
- 所有公共API都必须有相应的测试

## 📋 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交消息格式

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 提交类型

- `feat`: 新功能
- `fix`: 错误修复
- `docs`: 文档更新
- `style`: 代码格式化（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动

### 提交示例

```bash
# 新功能
git commit -m "feat(extractor): add support for custom transforms"

# 错误修复
git commit -m "fix(mps): resolve memory leak in batch processing"

# 文档更新
git commit -m "docs(api): update FeatureExtractor class documentation"

# 性能优化
git commit -m "perf(mps): optimize tensor operations for M1 chips"
```

## 🔍 Pull Request 指南

### PR标题

使用与提交消息相同的格式：
```
feat(extractor): add support for custom transforms
```

### PR描述模板

```markdown
## 📝 变更描述

简要描述此PR的目的和变更内容。

## 🔧 变更类型

- [ ] 错误修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 代码重构
- [ ] 测试改进

## 🧪 测试

- [ ] 所有现有测试通过
- [ ] 添加了新的测试用例
- [ ] 手动测试通过
- [ ] 性能测试通过（如适用）

## 📋 检查清单

- [ ] 代码遵循项目规范
- [ ] 自我审查了代码变更
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 变更不会破坏现有功能

## 📸 截图（如适用）

如果有UI变更或新的可视化功能，请添加截图。

## 🔗 相关Issue

Closes #123
```

### PR审查流程

1. **自动检查**: CI/CD会自动运行测试和代码检查
2. **代码审查**: 至少需要一位维护者的审查
3. **测试验证**: 确保所有测试通过
4. **文档检查**: 确保文档是最新的
5. **合并**: 审查通过后合并到主分支

## 🐛 问题报告

### 报告Bug

使用GitHub Issues报告bug时，请包含：

1. **环境信息**:
   - 操作系统和版本
   - Python版本
   - PyTorch版本
   - 硬件信息（特别是Mac芯片型号）

2. **重现步骤**:
   ```python
   # 提供完整的代码示例
   from feature_extractor import FeatureExtractor
   
   extractor = FeatureExtractor()
   # ... 重现bug的步骤
   ```

3. **期望行为**: 描述您期望发生什么
4. **实际行为**: 描述实际发生了什么
5. **错误信息**: 完整的错误堆栈跟踪

### Bug报告模板

```markdown
## 🐛 Bug描述

简要描述bug的现象。

## 🔄 重现步骤

1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## ✅ 期望行为

描述您期望发生什么。

## 📱 环境信息

- OS: [e.g. macOS 13.0]
- Python: [e.g. 3.9.7]
- PyTorch: [e.g. 2.0.1]
- 硬件: [e.g. MacBook Pro M1 Max]

## 📋 附加信息

添加任何其他相关信息。
```

## 💡 功能请求

### 提出新功能

在提出功能请求前，请：

1. 搜索现有的issues，确保功能未被提出
2. 考虑功能是否符合项目目标
3. 提供详细的用例说明

### 功能请求模板

```markdown
## 🚀 功能描述

简要描述您希望添加的功能。

## 💭 动机

解释为什么需要这个功能，它解决了什么问题。

## 📝 详细描述

详细描述功能应该如何工作。

## 🎯 用例

提供具体的使用场景。

## 🔄 替代方案

描述您考虑过的其他解决方案。
```

## 🤝 社区

### 获得帮助

- 📖 查看[文档](docs/)
- 🐛 报告[Issues](https://github.com/yourusername/img-ext/issues)
- 💬 参与[Discussions](https://github.com/yourusername/img-ext/discussions)
- 📧 发送邮件到 your.email@example.com

### 贡献者认可

我们感谢所有贡献者的努力！贡献者将被添加到：

- README.md的贡献者部分
- CHANGELOG.md的相应版本
- GitHub的贡献者列表

## 📄 许可证

通过贡献代码，您同意您的贡献将在与项目相同的[MIT许可证](LICENSE)下授权。

---

再次感谢您的贡献！🎉

如果您有任何问题，请随时通过GitHub Issues或邮件联系我们。