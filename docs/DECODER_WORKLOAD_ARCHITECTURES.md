# Decoder 主流架构与标准 Workload

LLMIR 论文与 benchmark **不以 ShareGPT 数据集命名**，而以 **主流 decoder-only 架构 + 标准 prefill/decode 形状** 组织实验。ShareGPT 仅是聊天 JSON 来源，**不能代表** Llama/Qwen 类模型的算子组合（GQA Attention、SwiGLU FFN、RMSNorm）。

## 1. 主流架构（论文应对标）

| 架构族 | 代表 preset | Attention | FFN | Norm | 典型层数×隐藏维 | 论文角色 |
|--------|-------------|-----------|-----|------|-----------------|----------|
| **Llama 3** | `llama3-8b`, `llama3.1-8b` | GQA | SwiGLU | RMSNorm | 32×4096 | **主对标** |
| **Qwen2** | `qwen2-7b` | GQA | SwiGLU | RMSNorm | 28×3584 | **主对标** |
| **DeepSeek** | `deepseek-7b`, `deepseek-v2.5-7b` | MHA / GQA | SwiGLU | RMSNorm | 30×4096 | **主对标** |
| **DeepSeek-V2** | `deepseek-v2-lite-16b` | MLA + MoE | SwiGLU + routed experts | RMSNorm | 26×2048 | 附录表 / MoE 未来 |
| **Mistral** | `mistral-7b` | GQA + SWA | SwiGLU | RMSNorm | 32×4096 | 长上下文对照 |
| **Mixtral** | `mixtral-8x7b` | MoE + GQA | SwiGLU | RMSNorm | sparse FFN | 未来 MoE Pass |
| GPT-2 | `gpt2` | MHA | GELU-MLP | LayerNorm | 12×768 | **仅 CI smoke**，非主文架构代表 |

注册表：`llmir-list-models` / `src/llmir/models/__init__.py` → `ModelRegistry`。

### 单层算子栈（compiler 关心）

每个 decoder block 在 IR 中应能表达：

```
input → RMSNorm → Self-Attention (+ KV cache) → residual
     → RMSNorm → SwiGLU FFN              → residual
```

- **Prefill**：Attention 计算主导，长度 \(L\) 时复杂度 \(\mathcal{O}(L^2)\)（或 SWA 窗口内）
- **Decode**：KV cache 访存主导，每步 \(\mathcal{O}(L)\) 内存带宽

E1 针对 **KV block / append / paged_attention**；E2 针对 **共享前缀的 prefill 复用**；E3 针对 **KV 驻留**。三者均应在 **GQA decoder** 语义下叙述，而非聊天语料品牌。

## 2. 标准 Workload 形状（替代 ShareGPT 命名）

采用 serving 文献常见 **长度分桶**，不绑定某一开源聊天集：

| 名称 | 共享前缀 \(L_s\) | 每请求后缀 \(L_u\) | 请求数 \(N\) | decode 步 | 场景 | 主导算子阶段 |
|------|------------------|-------------------|-------------|-----------|------|--------------|
| **S1** short | 128 | 8–32 | 32 | 4–16 | 多租户短指令 | decode |
| **S2** RAG | 512–2048 | 32–128 | 32 | 16–128 | 共享 system/RAG ctx | **prefill** |
| **S3** long-doc | 4096–8192 | 64–256 | 8–16 | 32–256 | 长上下文 agent | **attention prefill** |
| **D1** single | 64–512 | — | 1 | 8–64 | 延迟基线 | mixed |

E2 harness 实现的是 **S2 类 shared-prefix decoder workload**（多请求复用同一前缀 KV），与 vLLM/SGLang 论文中的 multi-tenant / prefix caching 设定一致，**与 ShareGPT 无必然关系**。

## 3. 命令与 legacy CLI

```bash
# 推荐：E2 shared-prefix decoder（legacy flag 仍可用）
llmir-benchmark --shared-prefix-bench --model llama3-8b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32

llmir-benchmark --shared-prefix-bench --model deepseek-7b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32

# CI smoke（非主文架构代表）
llmir-benchmark --shared-prefix-bench --model gpt2 \
  --shared-prefix-tokens 128 --shared-prefix-requests 32

python scripts/sharegpt_prefix_bench.py --simulation-only  # 脚本名 legacy
```

| Legacy | 推荐名称 |
|--------|----------|
| `--sharegpt-prefix-bench` | `--shared-prefix-bench` |
| `--sharegpt-system-tokens` | `--shared-prefix-tokens` |
| `sharegpt_2048_sim.json` | `shared_prefix_decoder_2048_sim.json` |

## 4. 论文写法

- **用**：「Llama/Qwen-class decoder」「shared-prefix prefill」「length bucket 2048/128」
- **不用**：「ShareGPT-shaped」作为主实验定义（最多在 related work 提聊天 trace 来源）
- **gpt2 主文**：仅 integration proxy，须脚注 **non-GQA legacy stack**

## 5. E4 组合验证输入

E4 trace 应参数化为 \((L_s, N, L_u, \text{arch})\)，其中 `arch` 取自 `ModelRegistry`（层数、GQA head 数），**不要**绑定 ShareGPT 语料统计。
