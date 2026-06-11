# 论文顶级标准对标（Top-Tier Bar）

LLMIR 论文长期对标 **OSDI / SOSP / ASPLOS / MLSys** 档 LLM 推理编译与系统方向。本文档是 `AGENTS.md` § LOOPs 的展开版，用于 Loop 2 Step 1 短板定位与里程碑排期。

**现状定位：** ICCD 修订稿 = **Tier-B 诚实版**（E1–E3 可复现 + 主文不夸大）。投 Tier-A 前必须闭合 E4–E7。

---

## 1. 领域顶级论文在问什么

顶会审稿人通常用五句话压测全文：

1. **Why compiler?** 为何 runtime（vLLM/SGLang）不够，compile-time 能带来什么可度量收益？
2. **Why LLM-specific IR?** 为何不是 Torch-MLIR / XLA / TVM 加几条 pass？
3. **Does it run end-to-end?** 优化是否在 **真实推理 hot path** 上生效，而非 microbench only？
4. **Is the evaluation fair?** 基线、模型、硬件、workload 是否与 SOTA 对齐？
5. **Can I reproduce it?** Artifact / 开源 / 固定环境能否复现主表？

LLMIR 当前最强叙事：**(3) 的部分答案** — E1–E3 证明 compiler→serving 链路可验证；**(4)(5) 在 gpt2/CPU 档部分满足**；**(1)(2) 有设计，需 E4+ 数据支撑**。

---

## 2. 六维验收详表

### 2.1 问题重要性（Problem）

| Tier-A 标准 | 当前 | 下一步 |
|-------------|------|--------|
| 生产 LLM 推理成本/延迟有量化背景 | 引言定性 | 引用 vLLM/行业报告或自有 trace |
| 明确 runtime-only 天花板（跨请求、跨层） | §1–2 已有 | 补 1 个「仅 runtime 做不到」的可编译例子（prefix + block co-design） |

### 2.2 技术新颖性（Novelty）

| Tier-A 标准 | 当前 | 下一步 |
|-------------|------|--------|
| 新 IR 类型/不变量（PagedKVCache 等） | Dialect + 类型定义 | 补语义不变量段落或 lit 证明 |
| 新 Pass / 算法（block size 等） | Algorithm 1 + E1 | E5 消融量化 Pass 收益 |
| 与最接近工作清晰分界 | Table related | 增 Torch-MLIR / MLC 对比表（能力维，非性能） |

### 2.3 系统完整性（System）

| Tier-A 标准 | 当前 | 下一步 |
|-------------|------|--------|
| Model import | Toy / partial | Llama-3-8B 或 Qwen2.5-7B import 路径 |
| Pass pipeline | 单层 + lit | 多层 / 全图 pass pipeline |
| Lowering + codegen | 部分 C++/CUDA | **M2**：`llm.paged_attention` 执行在 hot path |
| Serving integration | `llmir_paged` + vLLM pass-through | E4 同 harness 对标 |

### 2.4 实验严谨性（Evaluation）

| Tier-A 标准 | 当前 | 下一步 |
|-------------|------|--------|
| 模型规模 | gpt2（integration） | ≥7B ×2 家族 |
| 硬件 | CPU 为主；GPU 局部 | A100/L40 固定 SKU + CI nightly |
| Workload | ShareGPT 形状 sim + proxy | 真实 trace 或标准 bench（ShareGPT, HumanEval 长度分布） |
| 基线 | HF, 引用 Qwen/vLLM | 同机 vLLM, TRT-LLM, SGLang（能装则装） |
| 指标 | tok/s, prefix tokens, TTFT proxy | p50/p99 TTFT, throughput@batch, KV memory |
| 消融 | 附录 illustrative | E5 实测开关 |
| 统计 | 单次或少量 | ≥3 run, 方差/置信区间 |

### 2.5 对标诚实性（Positioning）

| 规则 | 说明 |
|------|------|
| 主文 measured 必须有 JSON + 命令 | 见 `PAPER_REVISION_TRACEABILITY.md` |
| 引用第三方数字须标明 **cited, not LLMIR** | `external_baselines.json` |
| 投影/设计目标仅附录 + 脚注 | `app:projected` |
| 禁止无 harness 的「优于 FlashAttention」 | `app:future_ops` |

### 2.6 可复现性（Artifact）

| Tier-A 标准 | 当前 | 下一步 |
|-------------|------|--------|
| 公开仓库 + 标签版本 | 是 | 投稿打 `paper-iccd-2025` tag |
| 一键复现主实验 | E1–E3 脚本 | `scripts/reproduce_paper.sh` |
| 硬件说明 | 部分 | `docs/HARDWARE.md` 固定 SKU |
| CI badge | Python offline CI | GPU workflow + artifact upload |

---

## 3. 实验档位定义（E1–E7）

| ID | 名称 | Tier-B (ICCD) | Tier-A (OSDI/MLSys) |
|----|------|---------------|---------------------|
| E1 | Compile-Time Pass Verification | **主文** | 主文（基础门槛） |
| E2 | Prefix-Aware Serving Evaluation | **主文** | 主文 + 真实 trace |
| E3 | GPU-Resident KV Integration | **主文**（panel c illustrative） | 主文全实测 |
| E4 | End-to-End Serving Parity | 附录/未来 | **主文必需** |
| E5 | Ablation Study | 附录 illustrative | **主文必需** |
| E6 | Multi-Hardware Matrix | 未来 | **主文必需** |
| E7 | Quality Preservation | 未来 | 强烈建议 |

---

## 4. 里程碑与依赖

```
M1  E1 单层 IR + CI               [done]
M2  Hot-path lowered execution      [blocker for E4]
M3  GPU harness 7B+                 [blocker for E4]
M4  Ablation flags → JSON           [E5]
M5  Multi-SKU benchmark matrix      [E6]
M6  Artifact bundle + doc           [MLSys AE]
```

**建议排期逻辑：** M2 ∥ M3 可部分并行；E4 依赖 M2+M3；E5 依赖 E4 框架；E6 依赖 E4 稳定；M6 与投稿同步。

---

## 5. 写作层级：同一仓库，两档文稿

| 元素 | ICCD 修订稿（现在） | Tier-A 目标稿 |
|------|---------------------|---------------|
| 摘要动词 | verify, demonstrate, proxy | improve, outperform (同 harness), reduce |
| §5 主文 | E1–E3 + 诚实表 | E1–E7 |
| 附录 | projected + future_ops | 仅补充材料，无主 claim |
| 结论 | 链路已验证；scale-out 未来 | 量化端到端收益 + 适用边界 |

**原则：** 不把 Tier-A 语言提前写进现稿；用本表驱动 Loop 1，达标后再升稿。

---

## 6. 维护

- 每闭合一个里程碑：更新本文件状态列 + `CAPABILITY_MATRIX.md` + `PAPER_REVISION_TRACEABILITY.md`。
- Loop 2 返修时：审稿意见映射到 **六维** 或 **E4–E7** 缺口，避免只改措辞。
