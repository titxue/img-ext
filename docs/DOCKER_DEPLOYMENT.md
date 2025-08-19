# Docker 部署指南

本文档介绍如何使用 Docker 部署图像特征提取 API 服务。

## 快速开始

### 方式一：使用 Docker Compose（推荐）

1. **启动服务**
```bash
# 启动基础服务
docker-compose up -d

# 启动包含 Nginx 反向代理的完整服务
docker-compose --profile with-nginx up -d
```

2. **访问服务**
- API 服务：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 使用 Nginx 时：http://localhost:80

3. **停止服务**
```bash
docker-compose down
```

### 方式二：使用 Docker 命令

1. **构建镜像**
```bash
docker build -t img-feature-extractor .
```

2. **运行容器**
```bash
docker run -d \
  --name feature-extractor \
  -p 8000:8000 \
  img-feature-extractor
```

3. **停止容器**
```bash
docker stop feature-extractor
docker rm feature-extractor
```

## 配置选项

### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `WORKERS` | 1 | Uvicorn 工作进程数量 |
| `PYTHONPATH` | /app | Python 模块搜索路径 |
| `LOG_LEVEL` | info | 日志级别 |

### 端口映射

- `8000`: FastAPI 服务端口
- `80`: Nginx 反向代理端口（使用 with-nginx profile 时）

### 数据卷

```yaml
volumes:
  - ./uploads:/app/uploads    # 上传文件目录
  - ./temp:/app/temp          # 临时文件目录
```

## 生产环境部署

### 1. 多实例部署

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  feature-extractor-api:
    build: .
    deploy:
      replicas: 3
    environment:
      - WORKERS=2
    networks:
      - feature-extractor-network
```

### 2. 资源限制

```yaml
services:
  feature-extractor-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 3. 健康检查配置

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 监控和日志

### 查看日志

```bash
# 查看服务日志
docker-compose logs -f feature-extractor-api

# 查看 Nginx 日志
docker-compose logs -f nginx
```

### 监控容器状态

```bash
# 查看容器状态
docker-compose ps

# 查看资源使用情况
docker stats
```

## 故障排除

### 常见问题

1. **容器启动失败**
```bash
# 检查日志
docker-compose logs feature-extractor-api

# 检查容器状态
docker-compose ps
```

2. **内存不足**
```bash
# 增加内存限制
docker-compose up -d --scale feature-extractor-api=1
```

3. **端口冲突**
```bash
# 修改端口映射
ports:
  - "8080:8000"  # 使用不同的主机端口
```

### 调试模式

```bash
# 以交互模式运行容器
docker run -it --rm \
  -p 8000:8000 \
  img-feature-extractor \
  /bin/bash
```

## 性能优化

### 1. 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### 2. 缓存优化

```bash
# 使用 BuildKit 进行并行构建
DOCKER_BUILDKIT=1 docker build -t img-feature-extractor .
```

### 3. 镜像大小优化

- 使用 `python:3.11-slim` 基础镜像
- 清理 apt 缓存
- 使用 `.dockerignore` 排除不必要文件
- 合并 RUN 指令减少层数

## 安全考虑

### 1. 非 root 用户

```dockerfile
# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

### 2. 只读文件系统

```yaml
services:
  feature-extractor-api:
    read_only: true
    tmpfs:
      - /tmp
      - /app/temp
```

### 3. 网络隔离

```yaml
networks:
  feature-extractor-network:
    driver: bridge
    internal: true  # 内部网络，不能访问外网
```

## 备份和恢复

### 数据备份

```bash
# 备份上传文件
docker run --rm -v img-ext_uploads:/data -v $(pwd):/backup \
  alpine tar czf /backup/uploads-backup.tar.gz -C /data .
```

### 数据恢复

```bash
# 恢复上传文件
docker run --rm -v img-ext_uploads:/data -v $(pwd):/backup \
  alpine tar xzf /backup/uploads-backup.tar.gz -C /data
```

## 更新和维护

### 滚动更新

```bash
# 构建新镜像
docker build -t img-feature-extractor:v2 .

# 更新服务
docker-compose up -d --no-deps feature-extractor-api
```

### 清理资源

```bash
# 清理未使用的镜像
docker image prune -f

# 清理未使用的容器
docker container prune -f

# 清理未使用的网络
docker network prune -f
```