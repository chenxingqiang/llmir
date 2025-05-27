FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir \
    vllm \
    pandas \
    matplotlib \
    seaborn \
    requests \
    tqdm \
    huggingface_hub

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 复制基准测试脚本
COPY llama31_benchmark.py /app/
COPY run_llama31_benchmark.sh /app/

# 设置权限
RUN chmod +x /app/run_llama31_benchmark.sh

# 入口点
ENTRYPOINT ["/bin/bash"]
