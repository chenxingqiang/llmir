FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    vllm \
    pandas \
    matplotlib \
    seaborn \
    requests \
    tqdm \
    huggingface_hub

ENV PYTHONUNBUFFERED=1

COPY benchmark/LLM/llama31_benchmark.py /app/
COPY benchmark/LLM/run_llama31_benchmark.sh /app/

RUN chmod +x /app/run_llama31_benchmark.sh

ENTRYPOINT ["/bin/bash"] 