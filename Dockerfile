# Dockerfile
# 基础镜像：Python 3.11
FROM python:3.11-slim AS base

# 基本设置：不生成 pyc，stdout 不缓存(方便看日志)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. 安装系统级依赖
# chromadb, llama_cpp_python, tree-sitter 等库需要 C++ 编译器
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 2. 优先复制依赖文件（利用 Docker 缓存层）
COPY requirements.txt ./requirements.txt

# 3. 安装 Python 依赖
# 注意：llama-cpp-python 在这里只作为普通库安装（CPU模式），
# 真正的推理计算交给另一个 llama.cpp-server 容器，所以这里不需要配置 GPU
RUN pip install --no-cache-dir -r requirements.txt

# 4. 再复制项目源码
COPY . .

# 暴露 FastAPI 端口
EXPOSE 8000

# 默认启动命令：指定 host 0.0.0.0 才能被外部访问
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
