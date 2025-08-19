from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles  # Vercel部署时注释掉
# from fastapi.templating import Jinja2Templates  # Vercel部署时注释掉
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import traceback

from feature_extractor import FeatureExtractor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="图像特征提取API",
    description="基于ResNet-18的图像特征提取服务，支持Mac M系列芯片MPS加速",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 挂载静态文件 (Vercel部署时注释掉)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板 (Vercel部署时注释掉)
# templates = Jinja2Templates(directory="templates")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局特征提取器实例
feature_extractor = None

# Pydantic模型定义
class DeviceConfig(BaseModel):
    """设备配置模型"""
    device: str = Field(default="auto", description="计算设备 (auto, mps, cuda, cpu)")
    model_name: str = Field(default="resnet18", description="模型名称")

class FeatureResponse(BaseModel):
    """单张图像特征提取响应模型"""
    success: bool = Field(description="是否成功")
    features: List[float] = Field(description="512维特征向量")
    device_info: dict = Field(description="设备信息")
    processing_time: float = Field(description="处理时间(秒)")
    message: Optional[str] = Field(default=None, description="消息")

class BatchFeatureResponse(BaseModel):
    """批量图像特征提取响应模型"""
    success: bool = Field(description="是否成功")
    features: List[List[float]] = Field(description="批量特征矩阵")
    count: int = Field(description="处理的图像数量")
    device_info: dict = Field(description="设备信息")
    processing_time: float = Field(description="处理时间(秒)")
    message: Optional[str] = Field(default=None, description="消息")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(description="服务状态")
    timestamp: str = Field(description="时间戳")
    device_info: dict = Field(description="设备信息")
    version: str = Field(description="API版本")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(default=False, description="是否成功")
    error: str = Field(description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")
    timestamp: str = Field(description="时间戳")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化特征提取器"""
    global feature_extractor
    try:
        logger.info("正在初始化特征提取器...")
        feature_extractor = FeatureExtractor(device='auto')
        logger.info(f"特征提取器初始化成功，设备: {feature_extractor.device}")
    except Exception as e:
        logger.error(f"特征提取器初始化失败: {str(e)}")
        raise e

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("正在关闭特征提取服务...")

# 工具函数
def validate_image(file: UploadFile) -> None:
    """验证上传的图像文件"""
    # 检查文件类型
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file.content_type}。请上传图像文件。"
        )
    
    # 检查文件大小 (限制为10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="文件大小超过10MB限制"
        )

def process_uploaded_image(file_content: bytes) -> Image.Image:
    """处理上传的图像文件"""
    try:
        image = Image.open(io.BytesIO(file_content))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法处理图像文件: {str(e)}"
        )

# API端点
@app.get("/", response_model=dict)
async def root():
    """根端点"""
    return {
        "message": "图像特征提取API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        device_info = feature_extractor.get_device_info() if feature_extractor else {}
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            device_info=device_info,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务不可用"
        )

@app.post("/extract-features", response_model=FeatureResponse)
async def extract_single_image_features(
    file: UploadFile = File(..., description="要处理的图像文件")
):
    """提取单张图像的特征向量"""
    start_time = datetime.now()
    
    try:
        # 验证文件
        validate_image(file)
        
        # 读取文件内容
        file_content = await file.read()
        
        # 处理图像
        image = process_uploaded_image(file_content)
        
        # 提取特征
        features = feature_extractor.extract_features(image, return_numpy=True)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return FeatureResponse(
            success=True,
            features=features.tolist(),
            device_info=feature_extractor.get_device_info(),
            processing_time=processing_time,
            message=f"成功提取特征，维度: {len(features)}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"特征提取失败: {str(e)}"
        )

@app.post("/extract-features-batch", response_model=BatchFeatureResponse)
async def extract_batch_features(
    files: List[UploadFile] = File(..., description="要处理的图像文件列表"),
    batch_size: int = Query(default=32, description="批处理大小")
):
    """批量提取图像特征"""
    start_time = datetime.now()
    
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请至少上传一个图像文件"
            )
        
        if len(files) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量处理最多支持100个文件"
            )
        
        # 处理所有图像
        images = []
        for file in files:
            validate_image(file)
            file_content = await file.read()
            image = process_uploaded_image(file_content)
            images.append(image)
        
        # 批量提取特征
        features_matrix = feature_extractor.extract_features_batch(
            images, 
            batch_size=batch_size, 
            return_numpy=True
        )
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchFeatureResponse(
            success=True,
            features=features_matrix.tolist(),
            count=len(images),
            device_info=feature_extractor.get_device_info(),
            processing_time=processing_time,
            message=f"成功处理{len(images)}张图像，特征维度: {features_matrix.shape[1]}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量特征提取失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量特征提取失败: {str(e)}"
        )

@app.get("/device-info", response_model=dict)
async def get_device_info():
    """获取设备信息和系统信息"""
    try:
        if not feature_extractor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="特征提取器未初始化"
            )
        
        # 获取系统信息
        import platform
        import psutil
        import torch
        import sys
        
        # 获取友好的操作系统名称
        def get_friendly_os_name():
            system = platform.system()
            release = platform.release()
            
            if system == "Darwin":
                # macOS版本映射
                version_map = {
                    "24": "macOS Sequoia",
                    "23": "macOS Ventura", 
                    "22": "macOS Monterey",
                    "21": "macOS Big Sur",
                    "20": "macOS Catalina",
                    "19": "macOS Mojave",
                    "18": "macOS High Sierra"
                }
                major_version = release.split('.')[0]
                friendly_name = version_map.get(major_version, "macOS")
                
                # 尝试获取更详细的版本信息
                try:
                    mac_version = platform.mac_ver()[0]
                    if mac_version:
                        return f"{friendly_name} ({mac_version})"
                except:
                    pass
                    
                return f"{friendly_name} ({release})"
            elif system == "Windows":
                return f"Windows {platform.release()}"
            elif system == "Linux":
                try:
                    # 尝试获取Linux发行版信息
                    import distro
                    return f"{distro.name()} {distro.version()}"
                except ImportError:
                    return f"Linux {release}"
            else:
                return f"{system} {release}"
        
        system_info = {
            "system": get_friendly_os_name(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
        }
        
        return {
            "success": True,
            "device_info": feature_extractor.get_device_info(),
            "system_info": system_info,
            "extractor_info": str(feature_extractor)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取设备信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备信息失败: {str(e)}"
        )

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """测试页面"""
    return templates.TemplateResponse("test.html", {"request": request})

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="内部服务器错误",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )