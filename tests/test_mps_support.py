import pytest
import torch
import numpy as np
from PIL import Image
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import FeatureExtractor


class TestMPSSupport:
    """测试Mac M系列芯片MPS支持"""
    
    def test_device_detection_auto(self):
        """测试自动设备检测"""
        extractor = FeatureExtractor(device='auto')
        device_info = extractor.get_device_info()
        
        # 验证设备信息包含必要字段
        assert 'device' in device_info
        assert 'device_type' in device_info
        assert 'mps_available' in device_info
        assert 'cuda_available' in device_info
        
        print(f"检测到的设备: {device_info['device']}")
        print(f"MPS可用: {device_info['mps_available']}")
        print(f"CUDA可用: {device_info['cuda_available']}")
    
    def test_mps_device_explicit(self):
        """测试显式指定MPS设备"""
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            pytest.skip("MPS不可用，跳过测试")
        
        extractor = FeatureExtractor(device='mps')
        assert extractor.device.type == 'mps'
        
        device_info = extractor.get_device_info()
        assert device_info['device_type'] == 'mps'
        assert device_info['mps_built'] is True
    
    def test_cpu_fallback(self):
        """测试CPU回退机制"""
        extractor = FeatureExtractor(device='cpu')
        assert extractor.device.type == 'cpu'
        
        device_info = extractor.get_device_info()
        assert device_info['device_type'] == 'cpu'
    
    def test_invalid_device_fallback(self):
        """测试无效设备的回退机制"""
        # 在没有CUDA的系统上指定CUDA应该回退到CPU
        if not torch.cuda.is_available():
            extractor = FeatureExtractor(device='cuda')
            assert extractor.device.type == 'cpu'
    
    def create_test_image(self, size=(224, 224)):
        """创建测试图像"""
        # 创建随机RGB图像
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    def test_single_image_feature_extraction(self):
        """测试单张图像特征提取"""
        extractor = FeatureExtractor()
        test_image = self.create_test_image()
        
        features = extractor.extract_features(test_image, return_numpy=False)
        
        # 验证特征维度
        assert features.shape == torch.Size([512]), f"期望特征维度[512]，实际{features.shape}"
        assert features.dtype == torch.float32
        
        # 验证特征不全为零
        assert not torch.allclose(features, torch.zeros_like(features))
        
        print(f"单张图像特征提取成功，特征维度: {features.shape}")
    
    def test_batch_feature_extraction(self):
        """测试批量图像特征提取"""
        extractor = FeatureExtractor()
        
        # 创建测试图像批次
        batch_size = 5
        test_images = [self.create_test_image() for _ in range(batch_size)]
        
        batch_features = extractor.extract_features_batch(test_images, batch_size=3, return_numpy=False)
        
        # 验证批量特征维度
        expected_shape = torch.Size([batch_size, 512])
        assert batch_features.shape == expected_shape, f"期望批量特征维度{expected_shape}，实际{batch_features.shape}"
        assert batch_features.dtype == torch.float32
        
        # 验证每个样本的特征不全为零
        for i in range(batch_size):
            assert not torch.allclose(batch_features[i], torch.zeros_like(batch_features[i]))
        
        print(f"批量图像特征提取成功，批量特征维度: {batch_features.shape}")
    
    def test_different_image_formats(self):
        """测试不同图像格式的处理"""
        extractor = FeatureExtractor()
        
        # 测试PIL图像
        pil_image = self.create_test_image()
        pil_features = extractor.extract_features(pil_image, return_numpy=False)
        assert pil_features.shape == torch.Size([512])
        
        # 测试numpy数组
        numpy_image = np.array(pil_image)
        numpy_features = extractor.extract_features(numpy_image, return_numpy=False)
        assert numpy_features.shape == torch.Size([512])
        
        print("不同图像格式处理测试通过")
    
    def test_feature_consistency(self):
        """测试特征提取的一致性"""
        extractor = FeatureExtractor()
        test_image = self.create_test_image()
        
        # 多次提取同一图像的特征
        features1 = extractor.extract_features(test_image, return_numpy=False)
        features2 = extractor.extract_features(test_image, return_numpy=False)
        
        # 验证特征一致性
        assert torch.allclose(features1, features2, atol=1e-6), "同一图像的特征提取结果应该一致"
        
        print("特征提取一致性测试通过")
    
    def test_model_loading(self):
        """测试模型加载"""
        extractor = FeatureExtractor()
        
        # 验证模型已正确加载到指定设备
        assert extractor.model is not None
        model_device = next(extractor.model.parameters()).device
        assert model_device.type == extractor.device.type
        
        # 验证模型处于评估模式
        assert not extractor.model.training
        
        print(f"模型成功加载到设备: {extractor.device}")
    
    def test_memory_efficiency(self):
        """测试内存使用效率"""
        extractor = FeatureExtractor()
        
        # 创建较大的测试批次
        large_batch = [self.create_test_image() for _ in range(10)]
        
        # 使用较小的batch_size来测试内存管理
        batch_features = extractor.extract_features_batch(large_batch, batch_size=3, return_numpy=False)
        
        assert batch_features.shape == torch.Size([10, 512])
        print("内存效率测试通过")


if __name__ == "__main__":
    # 运行基本测试
    test_suite = TestMPSSupport()
    
    print("=== 开始MPS支持测试 ===")
    
    try:
        test_suite.test_device_detection_auto()
        print("✅ 自动设备检测测试通过")
        
        test_suite.test_cpu_fallback()
        print("✅ CPU回退测试通过")
        
        test_suite.test_single_image_feature_extraction()
        print("✅ 单张图像特征提取测试通过")
        
        test_suite.test_batch_feature_extraction()
        print("✅ 批量特征提取测试通过")
        
        test_suite.test_different_image_formats()
        print("✅ 不同图像格式测试通过")
        
        test_suite.test_feature_consistency()
        print("✅ 特征一致性测试通过")
        
        test_suite.test_model_loading()
        print("✅ 模型加载测试通过")
        
        test_suite.test_memory_efficiency()
        print("✅ 内存效率测试通过")
        
        # 尝试MPS特定测试
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            test_suite.test_mps_device_explicit()
            print("✅ MPS设备测试通过")
        else:
            print("⚠️  MPS不可用，跳过MPS特定测试")
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise