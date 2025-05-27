#!/bin/bash

# 确保HUGGINGFACE_TOKEN环境变量存在
if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "警告: HUGGINGFACE_TOKEN 环境变量未设置。"
  echo "某些模型可能需要认证。如果您有令牌，请设置: export HUGGINGFACE_TOKEN=your_token"
else
  echo "HUGGINGFACE_TOKEN已设置，将传递给Docker容器"
fi

# 创建结果目录
mkdir -p results/llama31

# 构建Docker镜像
echo "正在构建Docker镜像..."
docker build -t llama31-benchmark .

# 运行基准测试
echo "正在运行基准测试..."
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
  llama31-benchmark \
  -c "./run_llama31_benchmark.sh --batch-sizes 1 --seq-lens 128 --repetitions 2 --output_dir ./results/llama31"

# 检查结果
echo "基准测试完成，结果保存在 $(pwd)/results/llama31 目录中" 