// 全局变量
let singleFile = null;
let batchFiles = [];
const API_BASE_URL = window.location.origin;

// DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkAPIStatus();
    // 定期检查API状态
    setInterval(checkAPIStatus, 30000);
});

// 初始化事件监听器
function initializeEventListeners() {
    // 单张图像上传
    const singleUploadArea = document.getElementById('singleUploadArea');
    const singleFileInput = document.getElementById('singleFileInput');
    
    singleUploadArea.addEventListener('click', () => singleFileInput.click());
    singleUploadArea.addEventListener('dragover', handleDragOver);
    singleUploadArea.addEventListener('dragleave', handleDragLeave);
    singleUploadArea.addEventListener('drop', (e) => handleSingleDrop(e));
    singleFileInput.addEventListener('change', (e) => handleSingleFileSelect(e));
    
    // 批量图像上传
    const batchUploadArea = document.getElementById('batchUploadArea');
    const batchFileInput = document.getElementById('batchFileInput');
    
    batchUploadArea.addEventListener('click', () => batchFileInput.click());
    batchUploadArea.addEventListener('dragover', handleDragOver);
    batchUploadArea.addEventListener('dragleave', handleDragLeave);
    batchUploadArea.addEventListener('drop', (e) => handleBatchDrop(e));
    batchFileInput.addEventListener('change', (e) => handleBatchFileSelect(e));
}

// 标签页切换
function showTab(tabName) {
    // 隐藏所有标签页内容
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // 移除所有按钮的活动状态
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 显示选中的标签页
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.classList.add('active');
}

// 检查API状态
async function checkAPIStatus() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'API服务正常';
        } else {
            throw new Error('API响应异常');
        }
    } catch (error) {
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'API服务离线';
    }
}

// 拖拽处理函数
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

// 单张图像处理
function handleSingleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && isImageFile(files[0])) {
        handleSingleFile(files[0]);
    } else {
        showError('请选择有效的图像文件');
    }
}

function handleSingleFileSelect(e) {
    const file = e.target.files[0];
    if (file && isImageFile(file)) {
        handleSingleFile(file);
    } else {
        showError('请选择有效的图像文件');
    }
}

function handleSingleFile(file) {
    singleFile = file;
    
    // 显示预览
    const preview = document.getElementById('singlePreview');
    const previewImg = document.getElementById('singlePreviewImg');
    const uploadArea = document.getElementById('singleUploadArea');
    const extractBtn = document.getElementById('singleExtractBtn');
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        preview.style.display = 'block';
        uploadArea.style.display = 'none';
        extractBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeSingleImage() {
    singleFile = null;
    document.getElementById('singlePreview').style.display = 'none';
    document.getElementById('singleUploadArea').style.display = 'block';
    document.getElementById('singleExtractBtn').disabled = true;
    document.getElementById('singleResult').style.display = 'none';
    document.getElementById('singleFileInput').value = '';
}

// 批量图像处理
function handleBatchDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files).filter(isImageFile);
    if (files.length > 0) {
        addBatchFiles(files);
    } else {
        showError('请选择有效的图像文件');
    }
}

function handleBatchFileSelect(e) {
    const files = Array.from(e.target.files).filter(isImageFile);
    if (files.length > 0) {
        addBatchFiles(files);
    } else {
        showError('请选择有效的图像文件');
    }
}

function addBatchFiles(files) {
    // 限制最多10张图片
    const remainingSlots = 10 - batchFiles.length;
    const filesToAdd = files.slice(0, remainingSlots);
    
    if (files.length > remainingSlots) {
        showError(`最多只能上传10张图片，已添加前${remainingSlots}张`);
    }
    
    batchFiles.push(...filesToAdd);
    updateBatchPreview();
}

function updateBatchPreview() {
    const preview = document.getElementById('batchPreview');
    const grid = document.getElementById('batchPreviewGrid');
    const uploadArea = document.getElementById('batchUploadArea');
    const extractBtn = document.getElementById('batchExtractBtn');
    
    if (batchFiles.length > 0) {
        preview.style.display = 'block';
        uploadArea.style.display = 'none';
        extractBtn.disabled = false;
        
        grid.innerHTML = '';
        batchFiles.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'preview-item';
            
            const img = document.createElement('img');
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-btn';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.onclick = () => removeBatchImage(index);
            
            item.appendChild(img);
            item.appendChild(removeBtn);
            grid.appendChild(item);
        });
    } else {
        preview.style.display = 'none';
        uploadArea.style.display = 'block';
        extractBtn.disabled = true;
    }
}

function removeBatchImage(index) {
    batchFiles.splice(index, 1);
    updateBatchPreview();
    
    if (batchFiles.length === 0) {
        document.getElementById('batchResult').style.display = 'none';
    }
}

function clearBatchImages() {
    batchFiles = [];
    updateBatchPreview();
    document.getElementById('batchResult').style.display = 'none';
    document.getElementById('batchFileInput').value = '';
}

// 特征提取功能
async function extractSingleFeatures() {
    if (!singleFile) {
        showError('请先选择图像文件');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', singleFile);
        
        const response = await fetch(`${API_BASE_URL}/extract-features`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        displaySingleResult(result);
        showSuccess('特征提取完成');
        
    } catch (error) {
        console.error('特征提取失败:', error);
        showError(`特征提取失败: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function extractBatchFeatures() {
    if (batchFiles.length === 0) {
        showError('请先选择图像文件');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        batchFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch(`${API_BASE_URL}/extract-features-batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        displayBatchResult(result);
        showSuccess(`成功处理${result.count}张图片`);
        
    } catch (error) {
        console.error('批量特征提取失败:', error);
        showError(`批量特征提取失败: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// 获取系统信息
async function getSystemInfo() {
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/device-info`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        displaySystemInfo(result);
        showSuccess('系统信息获取成功');
        
    } catch (error) {
        console.error('获取系统信息失败:', error);
        showError(`获取系统信息失败: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// 结果显示函数
function displaySingleResult(result) {
    const resultSection = document.getElementById('singleResult');
    const resultContent = document.getElementById('singleResultContent');
    
    // 从嵌套的数据结构中提取设备信息
    const deviceInfo = result.device_info || {};
    
    const html = `
        <div class="feature-info">
            <div class="info-card">
                <h4><i class="fas fa-image"></i> 图像信息</h4>
                <p>文件名: ${singleFile.name}</p>
                <p>文件大小: ${formatFileSize(singleFile.size)}</p>
                <p>处理时间: ${result.processing_time.toFixed(3)}秒</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-microchip"></i> 设备信息</h4>
                <p>设备类型: ${deviceInfo.device || 'N/A'}</p>
                <p>特征维度: ${result.feature_dim || 'N/A'}</p>
                <p>模型: ResNet-18</p>
            </div>
        </div>
        <div class="feature-vector">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong>特征向量 (前50维):</strong>
                <button class="copy-btn" onclick="copyFeatureVector(${JSON.stringify(result.features).replace(/"/g, '&quot;')})" title="复制完整特征向量">
                    <i class="fas fa-copy"></i> 复制完整向量
                </button>
            </div>
            [${result.features.slice(0, 50).map(f => f.toFixed(4)).join(', ')}${result.features.length > 50 ? ', ...' : ''}]
        </div>
    `;
    
    resultContent.innerHTML = html;
    resultSection.style.display = 'block';
}

function displayBatchResult(result) {
    const resultSection = document.getElementById('batchResult');
    const resultContent = document.getElementById('batchResultContent');
    
    // 验证数据结构
    if (!result.features || !Array.isArray(result.features)) {
        console.error('Invalid batch result structure:', result);
        showError('批量特征提取结果格式错误');
        return;
    }
    
    const totalTime = result.processing_time || 0;
    const avgTime = result.count > 0 ? totalTime / result.count : 0;
    
    // 从嵌套的数据结构中提取设备信息
    const deviceInfo = result.device_info || {};
    
    let html = `
        <div class="feature-info">
            <div class="info-card">
                <h4><i class="fas fa-images"></i> 批量处理信息</h4>
                <p>处理图片数: ${result.count}</p>
                <p>总处理时间: ${totalTime.toFixed(3)}秒</p>
                <p>平均处理时间: ${avgTime.toFixed(3)}秒/张</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-microchip"></i> 设备信息</h4>
                <p>设备类型: ${deviceInfo.device || 'N/A'}</p>
                <p>特征维度: ${result.features[0]?.length || 'N/A'}</p>
                <p>模型: ResNet-18</p>
            </div>
        </div>
    `;
    
    result.features.forEach((featureVector, index) => {
        html += `
            <div class="info-card" style="margin-top: 15px;">
                <h4><i class="fas fa-file-image"></i> 图片 ${index + 1}: ${batchFiles[index]?.name || 'Unknown'}</h4>
                <p>特征维度: ${featureVector.length}</p>
                <div class="feature-vector" style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <strong>特征向量 (前20维):</strong>
                        <button class="copy-btn" onclick="copyFeatureVector(${JSON.stringify(featureVector).replace(/"/g, '&quot;')})" title="复制完整特征向量">
                            <i class="fas fa-copy"></i> 复制完整向量
                        </button>
                    </div>
                    [${featureVector.slice(0, 20).map(f => f.toFixed(4)).join(', ')}${featureVector.length > 20 ? ', ...' : ''}]
                </div>
            </div>
        `;
    });
    
    resultContent.innerHTML = html;
    resultSection.style.display = 'block';
}

function displaySystemInfo(info) {
    const resultSection = document.getElementById('infoResult');
    const resultContent = document.getElementById('infoResultContent');
    
    // 从嵌套的数据结构中提取信息
    const deviceInfo = info.device_info || {};
    const systemInfo = info.system_info || {};
    
    const html = `
        <div class="feature-info">
            <div class="info-card">
                <h4><i class="fas fa-server"></i> 系统信息</h4>
                <p>操作系统: ${systemInfo.system || 'N/A'}</p>
                <p>Python版本: ${systemInfo.python_version || 'N/A'}</p>
                <p>PyTorch版本: ${systemInfo.torch_version || 'N/A'}</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-microchip"></i> 设备信息</h4>
                <p>当前设备: ${deviceInfo.device || 'N/A'}</p>
                <p>设备类型: ${deviceInfo.device_type || 'N/A'}</p>
                <p>MPS可用: ${deviceInfo.mps_available ? '是' : '否'}</p>
                <p>CUDA可用: ${deviceInfo.cuda_available ? '是' : '否'}</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-memory"></i> 硬件信息</h4>
                <p>CPU核心数: ${systemInfo.cpu_count || 'N/A'}</p>
                <p>内存总量: ${systemInfo.memory_total || 'N/A'}</p>
                <p>可用内存: ${systemInfo.memory_available || 'N/A'}</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-info-circle"></i> 提取器信息</h4>
                <p>${info.extractor_info || 'N/A'}</p>
            </div>
        </div>
    `;
    
    resultContent.innerHTML = html;
    resultSection.style.display = 'block';
}

// 工具函数
function isImageFile(file) {
    return file.type.startsWith('image/');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// UI反馈函数
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showError(message) {
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorToast.style.display = 'flex';
    
    // 5秒后自动隐藏
    setTimeout(hideError, 5000);
}

function hideError() {
    document.getElementById('errorToast').style.display = 'none';
}

function showSuccess(message) {
    const successToast = document.getElementById('successToast');
    const successMessage = document.getElementById('successMessage');
    
    successMessage.textContent = message;
    successToast.style.display = 'flex';
    
    // 3秒后自动隐藏
    setTimeout(hideSuccess, 3000);
}

function hideSuccess() {
    document.getElementById('successToast').style.display = 'none';
}

// 复制特征向量到剪贴板
async function copyFeatureVector(features) {
    try {
        // 将特征向量格式化为字符串
        const featureString = '[' + features.map(f => f.toFixed(6)).join(', ') + ']';
        
        // 使用现代剪贴板API
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(featureString);
        } else {
            // 降级方案：使用传统方法
            const textArea = document.createElement('textarea');
            textArea.value = featureString;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand('copy');
            textArea.remove();
        }
        
        showSuccess(`已复制完整特征向量 (${features.length}维) 到剪贴板`);
        
    } catch (error) {
        console.error('复制失败:', error);
        showError('复制失败，请手动选择并复制');
    }
}