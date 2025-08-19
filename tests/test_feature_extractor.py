import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """FeatureExtractor基础功能测试"""
    
    @pytest.fixture
    def extractor(self):
        """创建FeatureExtractor实例"""
        return FeatureExtractor()
    
    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        # 创建224x224的随机RGB图像
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def test_image_file(self, test_image):
        """创建临时图像文件"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_image.save(f.name, 'JPEG')
            yield f.name
        os.unlink(f.name)
    
    def test_initialization_default(self):
        """测试默认初始化"""
        extractor = FeatureExtractor()
        
        assert extractor.model_name == 'resnet18'
        assert extractor.device is not None
        assert extractor.model is not None
        assert extractor.transform is not None
    
    def test_initialization_with_device(self):
        """测试指定设备初始化"""
        extractor = FeatureExtractor(device='cpu')
        assert extractor.device.type == 'cpu'
    
    def test_device_info(self, extractor):
        """测试设备信息获取"""
        device_info = extractor.get_device_info()
        
        required_keys = ['device', 'device_type', 'mps_available', 'cuda_available']
        for key in required_keys:
            assert key in device_info
        
        assert isinstance(device_info['mps_available'], bool)
        assert isinstance(device_info['cuda_available'], bool)
    
    def test_extract_features_pil_image(self, extractor, test_image):
        """测试PIL图像特征提取"""
        features = extractor.extract_features(test_image, return_numpy=False)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == torch.Size([512])
        assert features.dtype == torch.float32
        assert not torch.allclose(features, torch.zeros_like(features))
    
    def test_extract_features_numpy_array(self, extractor, test_image):
        """测试numpy数组特征提取"""
        numpy_image = np.array(test_image)
        features = extractor.extract_features(numpy_image, return_numpy=False)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == torch.Size([512])
        assert features.dtype == torch.float32
    
    def test_extract_features_file_path(self, extractor, test_image_file):
        """测试文件路径特征提取"""
        features = extractor.extract_features(test_image_file, return_numpy=False)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == torch.Size([512])
        assert features.dtype == torch.float32
    
    def test_extract_features_batch(self, extractor):
        """测试批量特征提取"""
        # 创建测试图像批次
        batch_size = 5
        test_images = []
        for _ in range(batch_size):
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(Image.fromarray(image_array))
        
        batch_features = extractor.extract_features_batch(test_images, return_numpy=False)
        
        assert isinstance(batch_features, torch.Tensor)
        assert batch_features.shape == torch.Size([batch_size, 512])
        assert batch_features.dtype == torch.float32
        
        # 验证每个样本的特征不全为零
        for i in range(batch_size):
            assert not torch.allclose(batch_features[i], torch.zeros_like(batch_features[i]))
    
    def test_extract_features_batch_with_custom_batch_size(self, extractor):
        """测试自定义批处理大小的批量特征提取"""
        # 创建7张测试图像，使用批处理大小3
        total_images = 7
        batch_size = 3
        test_images = []
        for _ in range(total_images):
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(Image.fromarray(image_array))
        
        batch_features = extractor.extract_features_batch(test_images, batch_size=batch_size, return_numpy=False)
        
        assert batch_features.shape == torch.Size([total_images, 512])
    
    def test_image_format_conversion(self, extractor):
        """测试图像格式转换"""
        # 测试RGBA图像转换为RGB
        rgba_array = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        rgba_image = Image.fromarray(rgba_array, 'RGBA')
        
        features = extractor.extract_features(rgba_image, return_numpy=False)
        assert features.shape == torch.Size([512])
        
        # 测试灰度图像转换为RGB
        gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        gray_image = Image.fromarray(gray_array, 'L')
        
        features = extractor.extract_features(gray_image, return_numpy=False)
        assert features.shape == torch.Size([512])
    
    def test_feature_reproducibility(self, extractor, test_image):
        """测试特征提取的可重现性"""
        features1 = extractor.extract_features(test_image, return_numpy=False)
        features2 = extractor.extract_features(test_image, return_numpy=False)
        
        # 同一图像的特征应该完全一致
        assert torch.allclose(features1, features2, atol=1e-6)
    
    def test_invalid_image_format(self, extractor):
        """测试无效图像格式处理"""
        with pytest.raises(ValueError):
            extractor.extract_features(123)  # 传入数字而不是图像
    
    def test_model_eval_mode(self, extractor):
        """测试模型处于评估模式"""
        assert not extractor.model.training
    
    def test_repr(self, extractor):
        """测试字符串表示"""
        repr_str = repr(extractor)
        assert 'FeatureExtractor' in repr_str
        assert 'resnet18' in repr_str
        assert str(extractor.device) in repr_str
    
    def test_different_image_sizes(self, extractor):
        """测试不同尺寸图像的处理"""
        # 测试不同尺寸的图像都能正确处理
        sizes = [(100, 100), (300, 200), (500, 500)]
        
        for size in sizes:
            image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            features = extractor.extract_features(image, return_numpy=False)
            assert features.shape == torch.Size([512])
    
    def test_empty_batch(self, extractor):
        """测试空批次处理"""
        empty_batch = []
        
        with pytest.raises(ValueError):
            extractor.extract_features_batch(empty_batch)
    
    def test_single_image_batch(self, extractor, test_image):
        """测试单张图像的批处理"""
        single_batch = [test_image]
        batch_features = extractor.extract_features_batch(single_batch, return_numpy=False)
        
        assert batch_features.shape == torch.Size([1, 512])
        
        # 比较单张处理和批处理的结果
        single_features = extractor.extract_features(test_image, return_numpy=False)
        assert torch.allclose(batch_features[0], single_features, atol=1e-6)


if __name__ == "__main__":
    # 运行基本测试
    print("=== 开始FeatureExtractor基础功能测试 ===")
    
    test_suite = TestFeatureExtractor()
    
    try:
        # 创建测试实例
        extractor = FeatureExtractor()
        test_image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image_array)
        
        # 运行测试
        test_suite.test_initialization_default()
        print("✅ 默认初始化测试通过")
        
        test_suite.test_device_info(extractor)
        print("✅ 设备信息测试通过")
        
        test_suite.test_extract_features_pil_image(extractor, test_image)
        print("✅ PIL图像特征提取测试通过")
        
        test_suite.test_extract_features_numpy_array(extractor, test_image)
        print("✅ Numpy数组特征提取测试通过")
        
        test_suite.test_extract_features_batch(extractor)
        print("✅ 批量特征提取测试通过")
        
        test_suite.test_image_format_conversion(extractor)
        print("✅ 图像格式转换测试通过")
        
        test_suite.test_feature_reproducibility(extractor, test_image)
        print("✅ 特征可重现性测试通过")
        
        test_suite.test_different_image_sizes(extractor)
        print("✅ 不同图像尺寸测试通过")
        
        print("\n🎉 所有基础功能测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise