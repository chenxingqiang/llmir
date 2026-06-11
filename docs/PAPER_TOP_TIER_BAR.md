# 论文顶级标准对标（Top-Tier Bar）

LLMIR 论文长期对标 **OSDI / ASPLOS / PLDI / MLSys** 档 **LLM 编译与 IR** 方向（非 vLLM 替代品）。本文档是 `AGENTS.md` § LOOPs 的展开版。

**现状定位：** ICCD 修订稿 = Tier-B 诚实版（E1–E3）。Tier-A **编译向**目标 = 闭合 **A 类可验证证据（E1–E6）**，不把 **7B A100 实测对标** 作为理论或必要门槛。

---

## 1. 证据二分法（核心修正）

### 1.1 为何原「E4 端到端对标」不能作为理论验证

「LLMIR vs vLLM/HF，7B+，A100，同 harness 吞吐/延迟」依赖：

- 特定 GPU SKU、驱动、CUDA、cuDNN、Flash kernel 版本
- vLLM 连续批处理、PagedAttention CUDA 实现、调度策略
- 模型权重精度、量化、TP/PP 配置
- 网络、磁盘、HF 实现细节

上述因素 **无法从 IR/Pass 正确性推导**，不存在「理论建模即可闭合」的证明路径。因此它属于 **B 类 · 实测对标**，只能进实验室或 **可选 E8**，**不能** 作为 compiler 创新是否成立的必要条件。

### 1.2 A 类 vs B 类

| 类型 | 问什么 | 典型方法 | LLMIR 实验 |
|------|--------|----------|------------|
| **A · 可验证** | Pass 对吗？语义变了吗？组合 proxy 可算吗？ | lit、pytest、reference、KV 仿真、trace 公式 | E1–E6 |
| **B · 实测对标** | 生产集群谁更快？ | 同机 benchmark、多轮统计 | **E8**（可选） |

**顶会 compiler 稿的正当叙事：** 把 runtime 难以静态做的 **block / prefix / KV 布局** 提升到 compile-time；用 **A 类链条** 证明「设计成立 + 收益可分析 + 实现可复现」。B 类是 **生态位补充**，不是理论内核。

---

## 2. 领域顶级论文在问什么（编译向改写）

1. **Why compiler?** compile-time 能 **确定性地** 改变哪些量（block 数、重复 prefill、host 拷贝）？
2. **Why LLM-specific IR?** 这些量为何在通用 tensor IR 里 **不可表达或不可分析**？
3. **Is it correct?** Pass 是否保持语义？（E1 + lit）
4. **Is the benefit compositional?** E1+E2+E3 能否在 trace 上 **相加/上界**？（**E4**）
5. **Can I reproduce without your GPU farm?** E1–E5 是否 CPU/CI 可跑？

第 4 条用 **E4 组合验证** 回答，而非「我们比 vLLM 快 X%」。

---

## 3. 实验档位（E1–E8）

| ID | 名称 | 类型 | Tier-B | Tier-A 编译向 |
|----|------|------|--------|---------------|
| E1 | Compile-Time Pass Verification | A | 主文 | 主文 |
| E2 | Prefix-Aware Serving Evaluation | A | 主文 | 主文 |
| E3 | GPU-Resident KV Integration | A | 主文 | 主文 |
| **E4** | **Compositional / Trace-Driven Verification** | **A** | 未来 | **主文必需** |
| E5 | Ablation at Verifiable Layers | A | 附录 | 主文 |
| E6 | Multi-Backend Correctness Parity | A (+性能分写) | 未来 | 主文（正确性） |
| E7 | Quality Preservation (PPL/MMLU) | B | 未来 | 可选 |
| **E8** | Empirical E2E vs vLLM (7B+, A100) | **B** | 不做 | **可选加分** |

### E4 组合验证（可建模）应产出什么

1. **输入：** Decoder workload 参数 \((L_s, N, L_u, \text{arch})\)（S1/S2/S3 桶）或 `shared_prefix_decoder_*_sim.json`；`arch` 来自 `ModelRegistry`（Llama-3/Qwen2）
2. **模型：**
   - E1 → 每序列 block 数 \(B(L)\) 随 `block_size` 变化
   - E2 → 重复 prefill token 数 \(\sum_i \mathbb{1}[\text{prefix hit}]\cdot L_s\)
   - E3 → 每 decode 步 host↔device 拷贝次数上界（numpy vs torch_cuda）
3. **输出：** 可复现脚本 + JSON；论文一张「分析 vs 实测 proxy」对照表（允许 gpt2/仿真档）
4. **禁止：** 将 E4 结论写成「已证明端到端优于 vLLM」

### E8 实测对标（若做）应如何写

- 单独小节或附录，标题含 **Empirical comparison, same harness**
- 脚注：硬件 SKU、驱动、vLLM 版本、batch 策略
- **不与 E4 组合公式混为一条定理**

---

## 4. 六维验收（编译向）

| 维度 | Tier-A 期待 | 验证手段 | 是否依赖 E8 |
|------|-------------|----------|-------------|
| 问题重要性 | compile-time 可静态优化的量清晰 | 引言 + E4 trace 示例 | 否 |
| 技术新颖性 | LLM IR 语义 + Pass | Dialect + E1 | 否 |
| 系统完整性 | IR → runtime 链路 | E1–E3 + M5 hot path | 否 |
| 实验严谨性 | A 类可复现 + 消融 | E1–E6 | 否 |
| 对标诚实性 | A/B 分类清晰 | 正文措辞 | 否 |
| 可复现性 | CPU CI 复现主 claim | E1–E5 artifact | 否 |

---

## 5. 里程碑

```
M1  E1 单层 IR + CI                    [done]
M2  E4 trace/compositional 脚本 + JSON  [done]
M3  E5 消融开关                    [done]
M4  E6 多后端 parity          [done]
M5  hot-path lowered op（完整性）  [done]
M6  artifact 包（E1–E5，CPU 可跑）
M7  E8 GPU 实测（可选，不阻塞 M6）
```

---

## 6. 写作层级

| 元素 | ICCD 现稿 | Tier-A 编译向 |
|------|-----------|---------------|
| 摘要 | verify, demonstrate, proxy | verify + **compositional analysis** |
| 主文实验 | E1–E3 | E1–E6（A 类） |
| 端到端比 vLLM | 不做 / 引用第三方 | 仅 **E8** 可选附录 |
| 结论 | 链路可验证 | **可静态分析的优化空间** + 适用边界 |

---

## 7. 维护

- 新增实验先标 **A/B**，再决定能否进摘要。
- 审稿意见要求「比 vLLM 快」→ 能转 **E4 组合分析** 则转；否则 **E8 实测** 或 **收窄 scope**，不写假定理。
