# Decoder 主流架构与标准 Workload

LLMIR 默认模型注册表（`llmir-list-models`）仅保留 **千问 Qwen、Google Gemma 3、DeepSeek** 最新 open-weight preset。**不含 Llama / Mistral / Phi / Falcon**（仍可通过 HuggingFace `from_pretrained` 自动导入）。

> **Gemini 说明：** 闭源 Gemini API 模型不在此表；开源侧用 **Gemma 3** 作为 Google decoder 对标。

## 1. 精选架构（`ModelRegistry`）

| 系列 | Registry 名 | HF 示例 | Attention | FFN | 规模 |
|------|-------------|---------|-----------|-----|------|
| **Qwen3** | `qwen3-8b` | `Qwen/Qwen3-8B` | GQA | SwiGLU | 36×4096 |
| **Qwen3** | `qwen3-14b` | `Qwen/Qwen3-14B` | GQA | SwiGLU | 40×5120 |
| **Qwen2.5** | `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | GQA | SwiGLU | 28×3584 |
| **Qwen2.5** | `qwen2.5-72b` | `Qwen/Qwen2.5-72B-Instruct` | GQA | SwiGLU | 80×8192 |
| **Gemma 3** | `gemma-3-4b` | `google/gemma-3-4b-it` | GQA + SWA | GeGLU | 34×2560 |
| **Gemma 3** | `gemma-3-12b` | `google/gemma-3-12b-it` | GQA + SWA | GeGLU | 48×3840 |
| **Gemma 3** | `gemma-3-27b` | `google/gemma-3-27b-it` | GQA + SWA | GeGLU | 62×5376 |
| **DeepSeek** | `deepseek-v3` | `deepseek-ai/DeepSeek-V3` | MLA + MoE | SwiGLU | 61×7168 |
| **DeepSeek** | `deepseek-r1` | `deepseek-ai/DeepSeek-R1` | MLA + MoE | SwiGLU | 同 V3 |

**CI smoke：** `gpt2` 仅用于离线集成测试，不在精选注册表。

### 单层算子栈（compiler 关心）

```
input → RMSNorm → Self-Attention (+ KV cache) → residual
     → RMSNorm → SwiGLU / GeGLU FFN         → residual
```

## 2. 标准 Workload 形状

| 桶 | 共享前缀 \(L_s\) | 后缀 \(L_u\) | 请求 \(N\) | 场景 |
|----|------------------|-------------|-----------|------|
| **S1** | 128 | 8–32 | 32 | 短指令多租户 |
| **S2** | 2048 | 8–128 | 32 | RAG / shared system |
| **S3** | 8192 | 64+ | 16 | 长文档 prefill |

## 3. 命令

```bash
llmir-list-models

# E2 — 千问
llmir-benchmark --shared-prefix-bench --model qwen3-8b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32

# E2 — Gemma 3
llmir-benchmark --shared-prefix-bench --model gemma-3-12b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32

# E2 — DeepSeek（需 GPU + llmir[full]）
llmir-benchmark --shared-prefix-bench --model deepseek-v3 \
  --shared-prefix-tokens 2048 --shared-prefix-requests 8
```

## 4. 论文写法

- 主对标：**Qwen3 / Gemma 3 / DeepSeek-V3** decoder 算子栈
- 不用 ShareGPT 数据集命名；用 S1/S2/S3 长度桶
- `gpt2` 仅脚注为 legacy integration smoke
