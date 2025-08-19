import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Union, List, Optional
import warnings


class FeatureExtractor:
    """
    基于ResNet-18的图像特征提取器，支持Mac M系列芯片的MPS加速
    
    特性:
    - 智能设备检测 (MPS > CUDA > CPU)
    - 单张和批量图像处理
    - 512维特征向量输出
    - Mac M系列芯片优化
    """
    
    def __init__(self, device: str = 'auto', model_name: str = 'resnet18'):
        """
        初始化特征提取器
        
        Args:
            device (str): 计算设备 ('auto', 'mps', 'cuda', 'cpu')
            model_name (str): 模型名称，默认 'resnet18'
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = self._load_model()
        self.transform = self._get_transform()
        
        print(f"FeatureExtractor initialized on device: {self.device}")
        if self.device.type == 'mps':
            print("✅ Mac M系列芯片MPS加速已启用")
    
    def _get_device(self, device: str) -> torch.device:
        """
        智能设备检测，优先级: MPS > CUDA > CPU
        
        Args:
            device (str): 设备选择
            
        Returns:
            torch.device: 选择的计算设备
        """
        if device == 'auto':
            # Mac M系列芯片MPS支持检测
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device('mps')
            # CUDA支持检测
            elif torch.cuda.is_available():
                return torch.device('cuda')
            # 默认CPU
            else:
                return torch.device('cpu')
        else:
            # 验证指定设备的可用性
            if device == 'mps':
                if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                    warnings.warn("MPS不可用，回退到CPU")
                    return torch.device('cpu')
            elif device == 'cuda':
                if not torch.cuda.is_available():
                    warnings.warn("CUDA不可用，回退到CPU")
                    return torch.device('cpu')
            
            return torch.device(device)
    
    def _load_model(self) -> nn.Module:
        """
        加载预训练的ResNet-18模型并移除分类层
        
        Returns:
            nn.Module: 特征提取模型
        """
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            # 移除最后的分类层，保留特征提取部分
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
        
        model.eval()
        model.to(self.device)
        
        # 对于MPS设备，确保模型参数正确转换
        if self.device.type == 'mps':
            for param in model.parameters():
                param.data = param.data.to(self.device)
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """
        获取图像预处理变换
        
        Returns:
            transforms.Compose: 图像变换管道
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: PIL图像、numpy数组或图像路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise ValueError(f"不支持的图像格式: {type(image)}")
        
        return self.transform(image)
    
    def extract_features(self, image: Union[Image.Image, np.ndarray, str], return_numpy: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        提取单张图像的特征
        
        Args:
            image: 输入图像
            return_numpy: 是否返回numpy数组，默认True
            
        Returns:
            Union[torch.Tensor, np.ndarray]: 512维特征向量
        """
        # 预处理图像
        input_tensor = self._preprocess_image(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(input_batch)
            # 展平特征向量 [1, 512, 1, 1] -> [512]
            features = features.squeeze()
            
            # 转换为numpy数组（如果需要）
            if return_numpy:
                features = features.cpu().numpy()
        
        return features
    
    def extract_features_batch(self, images: List[Union[Image.Image, np.ndarray, str]], 
                             batch_size: int = 32, return_numpy: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        批量提取图像特征
        
        Args:
            images: 图像列表
            batch_size: 批处理大小
            return_numpy: 是否返回numpy数组，默认True
            
        Returns:
            Union[torch.Tensor, np.ndarray]: 批量特征矩阵 [N, 512]
        """
        if not images:
            raise ValueError("图像列表不能为空")
            
        all_features = []
        
        # 分批处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # 预处理批量图像
            batch_tensors = []
            for img in batch_images:
                tensor = self._preprocess_image(img)
                batch_tensors.append(tensor)
            
            # 堆叠为批量张量
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                batch_features = self.model(batch_input)
                # 展平特征向量 [B, 512, 1, 1] -> [B, 512]
                batch_features = batch_features.squeeze(dim=(2, 3))
                all_features.append(batch_features)
        
        # 合并所有批次的特征
        features = torch.cat(all_features, dim=0)
        
        # 转换为numpy数组（如果需要）
        if return_numpy:
            features = features.cpu().numpy()
            
        return features
    
    def get_device_info(self) -> dict:
        """
        获取设备信息
        
        Returns:
            dict: 设备信息字典
        """
        info = {
            'device': str(self.device),
            'device_type': self.device.type,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if self.device.type == 'mps':
            info['mps_built'] = torch.backends.mps.is_built()
        elif self.device.type == 'cuda':
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        
        return info
    
    def __repr__(self) -> str:
        return f"FeatureExtractor(model={self.model_name}, device={self.device})"