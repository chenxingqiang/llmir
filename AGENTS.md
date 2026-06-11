# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

**LLMIR** is an MLIR-based compiler and Python runtime for optimizing LLM inference. The primary development surface is the **`llmir` Python package** in `src/llmir/`. The repo also vendors a full MLIR tree and a custom LLM dialect under `include/mlir/Dialect/LLM/` and `lib/Dialect/LLM/`.

## Cursor Cloud specific instructions

### Default development path (Python)

Most agent work should use the Python package path — this matches CI (`.github/workflows/python-package.yml`):

```bash
export PATH="$HOME/.local/bin:$PATH"
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/llmir
black --check src/llmir
mypy src/llmir --ignore-missing-imports
```

**PATH note:** `pip install --user` places CLI entry points (`pytest`, `llmir-benchmark`, `llmir-list-models`, etc.) in `~/.local/bin`. Ensure that directory is on `PATH` before running commands.

### Services

| Service | Required? | Notes |
|---------|-----------|-------|
| Python 3.8+ | Yes | System Python 3.12 works |
| `llmir` editable install | Yes | `pip install -e ".[dev]"` |
| pytest / ruff / black / mypy | Yes | Installed via `[dev]` extra |
| MLIR/LLVM native build | No (default) | Only for C++/lit work; see below |
| GPU / CUDA | No | Optional for vLLM/GPU benchmarks |
| HuggingFace network | No | Optional; tests marked `@pytest.mark.network` |
| Docker | No | Optional for `docker-compose.yml` GPU benchmarks |

There is no long-running dev server. The "application" is the Python library and its CLI tools.

### Hello-world verification

After install, confirm the environment with:

```bash
python3 -c "from llmir import PagedKVCache, KVCacheConfig; c = KVCacheConfig(num_layers=8, num_heads=8, head_dim=64); print(PagedKVCache(c))"
llmir-list-models
llmir-benchmark --model qwen3-8b --batch-sizes 1,4
```

### Optional extras

- **`pip install -e ".[full]"`** — adds PyTorch + Transformers for `llmir_paged` backend and HuggingFace integration tests.
- **Native MLIR build** — requires LLVM 18, CMake ≥3.20, Ninja. From repo root: `mkdir build && cd build && cmake -G Ninja .. && ninja`. LLM dialect tests live under `test/Dialect/LLM/`. This is a large build and is not required for Python-only changes.
- **GPU benchmarks** — `docker-compose up llama31-benchmark` or `benchmark/LLM/` scripts need NVIDIA GPU + vLLM + often `HUGGINGFACE_TOKEN`.

### Lint / test caveats

- **110 pytest tests** pass offline; 3 tests in `test_paged_decoder.py` skip without `[full]` (torch/transformers).
- **`black --check`** may report pre-existing formatting drift in `src/llmir/profiling/__init__.py`.
- **`mypy`** may report errors in `src/llmir/serving/engine.py` depending on mypy version; CI uses the same command.

### Key directories

- `src/llmir/` — Python package (runtime, serving, models, CLI)
- `tests/` — pytest suite
- `include/mlir/Dialect/LLM/`, `lib/Dialect/LLM/` — MLIR dialect and C++ runtime
- `benchmark/`, `scripts/` — performance and integration scripts

---

## LOOPs — 工程迭代与论文对齐

面向 **LLMIR** 开源运维与学术论文撰写，沿用**工程叙事、闭环迭代、问题驱动**文风。Agent 执行任务时，先判断当前处于哪条 Loop、哪一步，再动手改代码或改论文。

**权威状态锚点（改论文前必读）：**

| 文档 | 用途 |
|------|------|
| [`docs/DECODER_WORKLOAD_ARCHITECTURES.md`](docs/DECODER_WORKLOAD_ARCHITECTURES.md) | Qwen + **开源 Gemma** + DeepSeek；S1/S2/S3 workload |
| [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) | 功能真实实现程度（C++ / Python ref / Demo / Planned） |
| [`docs/PAPER_REVISION_TRACEABILITY.md`](docs/PAPER_REVISION_TRACEABILITY.md) | 论文主张 ↔ 仓库证据（E1–E3、JSON、测试） |
| [`IEEE-conference/REVISION_NOTES.md`](IEEE-conference/REVISION_NOTES.md) | 审稿意见映射与「已验证 / 投影 / 未来工作」边界 |
| [`IEEE-conference/LLMIR-paper-ICCD2025-revised.tex`](IEEE-conference/LLMIR-paper-ICCD2025-revised.tex) | 当前修订稿主文件 |

### 一、整体说明

LLMIR 是面向 **LLM 推理** 的专用 MLIR 方言与编译/runtime 栈，核心能力包括：LLM 专用 IR 类型与算子、KV/PagedAttention 编译 Pass、Lowering 到 runtime 调用、薄 runtime + `llmir_paged` serving 集成。

本仓库设计 **两套可联动迭代循环**：

| Loop | 定位 | 周期 | 终止条件 |
|------|------|------|----------|
| **Loop 1 · 工程迭代** | 代码、性能、功能、多硬件适配 | 周/月，**无限循环** | 无（开源长期演进） |
| **Loop 2 · 论文迭代** | 撰写、实验、评审、返修、定稿 | 日/周（初稿）或 月/季（投稿） | 录用或转投重构 |

**论文对齐原则（贯穿 Loop 2）：** 主文只写 **CI 可复现 + JSON 可查** 的结论；算子级 speedup、多卡吞吐、全模型对标等，除非 harness 落地，否则只能进附录并标注 *projected / future work*。

**文章目标：对标领域顶级标准。** 长期对标 **OSDI / SOSP / ASPLOS / MLSys** 档 LLM 编译与推理系统论文；当前 ICCD 修订稿是**诚实的中期版本**（E1–E3 证据链），不是终态。Loop 2 每次改稿先用下表自检：**主张是否达到该 venue 的默认审稿门槛**；达不到则收窄表述或回流 Loop 1 补实验。

#### 顶级论文六维验收（审稿人默认门槛）

| 维度 | Tier-A 期待（OSDI/ASPLOS/MLSys） | 当前仓库（ICCD 修订稿） | 差距处理 |
|------|----------------------------------|-------------------------|----------|
| **问题重要性** | Runtime-only 优化瓶颈清晰；compile-time 跨算子/跨请求机会可量化 | 引言 + Table I 已论证 | 保持；补 1–2 个生产场景数字（cite 或 measured） |
| **技术新颖性** | LLM 专用 IR/语义可静态分析 runtime 难做之事（PagedKV、prefix、block policy） | Dialect + Algorithm 1 + E1 | 强化「与 Torch-MLIR/TVM 差异」；避免泛化 MLIR 包装叙事 |
| **系统完整性** | Import → Pass → Lowering → **hot path 执行** 闭环 | Import/Pass 偏 toy；serving 走 HF + `llmir_paged` | **M2**：lowered kernel 上 hot path |
| **实验严谨性** | 多模型、多硬件、强基线、消融、显著性/方差 | E1–E3 + gpt2 CPU；附录为 projected | **M3–M5**（见下） |
| **对标诚实性** | 主张类型与证据类型匹配；不把实测对标说成可证明定理 | Qwen 为引用；gpt2 为 integration proxy | 主文禁写「全面优于」；**E4 用组合验证**，非 7B 实测 |
| **可复现性** | Artifact：一键脚本 + 固定硬件 + CI badge | E1–E3 pytest + JSON + 图脚本 | **M6**：artifact 包（CPU 可复现优先） |

#### 对标参照系（Related Work 必覆盖）

| 类别 | 代表工作 | 论文中如何差异化 |
|------|----------|------------------|
| Runtime serving | vLLM, SGLang, TensorRT-LLM | LLMIR 做 **compile-time IR + 薄 runtime**，非调度器替代 |
| DL 编译器 | XLA, TVM, Torch-MLIR | 强调 **PagedAttention / KV / prefix** 语义，非通用 tensor 图 |
| LLM 部署编译 | MLC-LLM, TensorRT-LLM engine | 强调 **IR 级 block/prefix 分析与 Pass**，非仅 kernel 库 |
| 注意力优化 | FlashAttention 系列 | 算子级对比放 future work；主文只做 **IR→serving 链路** |

#### 证据二分法（写论文前必分清）

LLMIR 是 **编译器 / IR** 贡献，不是 serving 框架替换。证据分两类，**不可混写**：

| 类型 | 含义 | 典型实验 | 能否离线/理论闭合 |
|------|------|----------|-------------------|
| **A · 可验证** | Pass 正确性、语义保持、组合代理指标、trace 驱动上界 | E1–E3、E4、E5 | **能**（CI、仿真、reference、lit） |
| **B · 实测对标** | 同机击败 vLLM/TRT、7B+ A100 吞吐 | **E8**（原「E4 端到端」） | **不能**理论建模，仅实验室实测 |

> **结论：** 「LLMIR vs vLLM 7B A100 谁更快」属于 **B 类**，无法用定理保证，也不应作为 compiler 创新的**必要**验收条件。顶会 compiler 稿应把 **A 类链条做满**，B 类作可选实证补充并明确 scope。

#### 实验升级路线图（E1–E8）

| 阶段 | ID | 目标 | 验证性质 | Tier-A（编译向） |
|------|-----|------|----------|------------------|
| 已具备 | **E1** | Pass 正确性 + block rewrite | A · 可证明 | **主文必需** |
| 已具备 | **E2** | Prefix / TTFT 代理 | A · 可仿真 | **主文必需** |
| 已具备 | **E3** | GPU KV 集成正确性 | A · 可单测 | **主文必需** |
| 已实现 | **E4** | **组合验证**：E1 block 策略 + E2 prefix 命中率 + E3 拷贝次数 → trace 上 prefill/KV **可计算上界** | A · 建模 | `scripts/e4_compositional_verify.py` |
| 已实现 | **E5** | **可验证消融**：关 Pass / prefix / block-opt，各层 proxy 变化可预期 | A · 开关+JSON | `scripts/e5_ablation_verify.py` |
| 已实现 | **E6** | **多后端正确性**：CPU/GPU/native 输出一致；性能分面板、不混口径 | A 正确性 + B 性能 | `scripts/e6_backend_parity_verify.py` |
| 可选 | **E7** | 质量无损 PPL/MMLU | B · 实测 | 加分 |
| 可选 | **E8** | 同 harness 端到端：7B+ vs vLLM/HF on A100 | **B · 仅实测** | **非必需**；有则脚注硬件 SKU |

**E4 组合验证（可建模）示例表述：**

- 给定 trace：系统 prompt 长度 \(L_s\)、请求数 \(N\)、suffix \(L_u\)
- E1 给出 block 数上界 \(B(L)\)；E2 给出重复 prefill 节省 \(\Delta T_\text{prefill}\)；E3 给出每步 host 拷贝上界 \(C_\text{step}\)
- 主文结论：**在同样 HF 前向前提下，优化来自可静态分析的 KV/block/prefix 项**，而非声称已证明端到端优于 vLLM

**工程里程碑（Loop 1 牵引 Loop 2）：**

```
M1  E1 单层 IR 闭环              ← 已完成
M2  E4 trace 驱动组合验证脚本     ← Tier-A（编译向）核心 [done]
M3  E5 消融开关 + JSON schema          [done]
M4  E6 多后端正确性 parity tests     [done]
M5  Lowered op hot path（增强 E1）← 强化完整性，非 E8 前置
M6  Artifact 包（CPU 可复现 E1–E5）
M7  E8 GPU 实测对标（可选实验室）
```

#### 各 venue 口径（避免错配）

| Venue 类型 | 代表 | 实验深度 | 当前稿策略 |
|------------|------|----------|------------|
| **Tier-A 编译/系统** | OSDI, ASPLOS, PLDI | **E1–E6（A 类）** + 清晰 scope | 目标稿；**不绑架 E8** |
| **Tier-A ML 系统** | MLSys | A 类 + 常要 B 类实测 | E8 加分；无 GPU 仍可投 compiler 子贡献 |
| **Tier-B** | ICCD, ATC | E1–E3 + 诚实边界 | **当前修订稿** |
| **Workshop** | 局部创新 | E1–E3 | 技术报告 |

#### Loop 2 顶级标准自检（改摘要/结论前必答）

1. 每条贡献是 **A 类还是 B 类**？B 类未实测则不得写进摘要。
2. 能否用 **E4 组合论证** 解释「为何 compile-time 有价值」，而不靠「打败 vLLM」？
3. E1–E5 是否在 **无 A100** 环境可复现？
4. 数字来自 **E 几档**？投影表不得进摘要。
5. 审稿人问「为何不直接改 vLLM？」→ 答 **IR 静态分析项（block/prefix/KV）**，不是调度器竞争。

详细差距追踪与里程碑状态：[`docs/PAPER_TOP_TIER_BAR.md`](docs/PAPER_TOP_TIER_BAR.md)。

**论文实验命名（正文用 E1–E3，CLI 保留 legacy）：**

| ID | 名称 | 验证命令 |
|----|------|----------|
| **E1** | Compile-Time Pass Verification | `pytest tests/test_mvp_a_e2e.py -m "not network"` |
| **E2** | Prefix-Aware Serving Evaluation | `llmir-benchmark --shared-prefix-bench` |
| **E3** | GPU-Resident KV Integration | `pytest tests/test_mvp_c_e2e.py -m "not network"` |

### 二、Loop 1：工程持续迭代闭环（无限循环）

#### 核心定位

以 **实测数据 + Issues/用户反馈** 驱动 `src/llmir/`、MLIR dialect、runtime 与 benchmark 脚本演进。每轮只收敛 1–3 个可验证目标。

#### 单轮流程

```
Step 1 基线 & 瓶颈诊断
    ↓
Step 2 方案 & 任务拆解
    ↓
Step 3 开发 & 单测
    ↓
Step 4 回归 & 多环境验证  ──失败──→ 回到 Step 2
    ↓
Step 5 发布 & 沉淀  ──→ 回到 Step 1
```

**Step 1 · 基线评测 & 瓶颈诊断**

```bash
pytest tests/ -m "not network" -q
pytest tests/test_mvp_a_e2e.py tests/test_mvp_c_e2e.py tests/test_sharegpt_prefix_bench.py -m "not network" -q
python3 scripts/paper_benchmark_collect.py --model gpt2
```

| 层级 | 测什么 | 典型入口 |
|------|--------|----------|
| 编译 Pass | block size 改写、reference 数值一致 | E1、`llmir-compile --mvp-a-e2e` |
| Serving 代理 | shared-prefix decoder prefill 复用 | E2、`docs/DECODER_WORKLOAD_ARCHITECTURES.md` |
| Runtime 集成 | GPU KV 无 NumPy 往返 | E3、`llmir-benchmark --mvp-c-bench` |
| 算子微基准 | attention toy kernel | `benchmark/attention/`（**非** hot path） |

优先级：**P0 正确性** → **P1 主文证据链** → **P2 性能** → **P3 生态**。

**Step 2–5：** 方案拆解 → 最小 diff 开发 → 三层回归（pytest / E1–E3 / 图再生）→ 合并并更新 `CAPABILITY_MATRIX` 与 benchmarks JSON。改动影响论文主张时，同步触发 Loop 2 Step 1。

**Step 5 收工（必做）：** 执行 **§六 Agent 迭代收工协议** — 合并 PR 到 `main` → `git pull origin main` → 从最新 `main` 创建下一轮 `cursor/<topic>-575e` 分支。

### 三、Loop 2：学术论文迭代闭环

核心职责：**把文章与仓库真实能力对齐**。

#### 场景 A：初稿打磨（短循环）

```
Step 1 框架 & 短板定位 → Step 2 实验 & 数据 → Step 3 行文重构 → Step 4 内部评审
```

**Step 1 三连表（必做）：**

1. 贡献点 ↔ 论文 §X 是否都有 harness？
2. `CAPABILITY_MATRIX` ↔ 是否把 Planned 写成 Implemented？
3. `REVISION_NOTES` ↔ 未闭合项是否仍出现在摘要/结论？

**主文允许的证据：**

| 类型 | 论文位置 | 仓库要求 |
|------|----------|----------|
| 编译验证 | §5 E1、`lst:e1_mlir` | `gpt2_e1_snippet.mlir`、E1 pytest |
| Serving 代理 | §5 E2、`fig:prefix_ttft` | `shared_prefix_decoder_2048_sim.json`、`paper_results.json` |
| GPU KV | §5 E3、`fig:e1_e3_eval` | E3 pytest；panel c 标 illustrative |
| decode 基线 | `tab:measured_harness` | gpt2 实测 + Qwen **引用** |
| 算子/多模型 | Appendix | 脚注 *not measured in CI* |

**Step 2 命令：**

```bash
python3 scripts/paper_benchmark_collect.py --model gpt2
python3 IEEE-conference/figures/generate_all_nature_figures.py
python3 IEEE-conference/figures/generate_projected_figures.py  # 仅附录
```

行文主线：**问题 → LLM 专用 IR / Pass → 可验证 serving 路径 → 诚实边界**。

#### 场景 B：投稿 & 返修（长循环）

排版投稿 → 分拣意见（创新/实验/原理/格式）→ 论文+工程联动整改 → rebuttal → 二次评审。

每条审稿意见映射：`revised.tex` 改动 + JSON/pytest + `REVISION_NOTES` 状态。无法补实验则**收窄 claim**，不做纯文字辩护。

### 四、双循环联动

| 方向 | 触发 | 动作 |
|------|------|------|
| 工程 → 论文 | E1/E2/E3 PR 合并 | 更新 §5、traceability、regen figures |
| 论文 → 工程 | 审稿要新实验 | Loop 1 开 harness 分支，先绿测试再写主文 |
| 迭代收工 | 里程碑 Mx 完成 | **§六**：merge PR → pull `main` → 新分支 |

### 五、论文对齐验证清单（收工前必跑）

```bash
pytest tests/test_mvp_a_e2e.py tests/test_mvp_c_e2e.py tests/test_sharegpt_prefix_bench.py tests/test_e4_compositional.py tests/test_e5_ablation.py tests/test_e6_backend_parity.py -m "not network" -q
python3 scripts/e4_compositional_verify.py --from-sim IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json
python3 scripts/e5_ablation_verify.py --from-sim IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json
python3 scripts/e6_backend_parity_verify.py --model toy
python3 -c "
import json; from pathlib import Path
p=json.loads(Path('IEEE-conference/benchmarks/paper_results.json').read_text())
assert 'e1' in p; assert Path(p['e1']['mlir_snippet_path']).is_file()
print('paper_results OK')
"
python3 IEEE-conference/figures/generate_all_nature_figures.py
```

**Definition of Done：** traceability 可核对；主文无未标注的 operator/多卡 measured 口吻；图由脚本生成；E1–E6 命名统一。

### 六、Agent 迭代收工协议（自动合并 + 新分支）

每完成 **一个里程碑（Mx）或一轮 Loop Step 5** 后，Agent **必须**合并进 `main`，再从 **最新 `main`** 开下一轮分支。不要在未合并的旧分支上连续堆多个里程碑。

> 说明：原 `Agent.md` 已并入本文件；下文 Git 协议对 Cloud Agent 与本地 Agent 均适用。

#### 触发条件（须全部满足）

1. 本迭代 Definition of Done 已达成（测试绿、文档 / traceability 已同步）
2. 变更已 commit 并 push 到当前 `cursor/<topic>-575e` 分支
3. PR 已创建或更新，描述含本迭代摘要与测试计划

#### 收工步骤（按序执行，不跳步）

```bash
# 0. 收工前最后一轮验证（见 §五）
pytest tests/ -m "not network" -q

# 1. 合并当前 PR 到 main（优先 squash；与仓库默认策略一致即可）
gh pr merge <PR_NUMBER> --squash --delete-branch
# 若无 gh：在 GitHub UI 合并后，本地执行 git pull

# 2. 同步最新 main
git fetch origin main
git checkout main
git pull origin main

# 3. 从最新 main 开下一轮分支（命名：cursor/<next-topic>-575e）
git checkout -b cursor/<next-milestone-topic>-575e

# 4. 推送远程并开始下一迭代
git push -u origin cursor/<next-milestone-topic>-575e
```

#### 分支命名示例

| 刚完成 | 下一分支名 |
|--------|------------|
| M2 E4 compositional | `cursor/e5-ablation-575e` |
| M3 E5 ablation | `cursor/e6-backend-parity-575e` |
| M4 E6 parity | `cursor/m5-hot-path-575e` |

#### 禁止

- 在 **未合并** 的 stale 分支上继续堆 M3、M4、M5（应 merge → pull main → 新分支）
- 从 **落后 main 数周** 的旧分支直接 rebase 后继续开发（应先合并当前工作，再开新分支）
- 合并前跳过 §五 验证清单

#### 例外

用户显式要求「继续在同一 PR / 分支上堆叠」时，可跳过合并，但须在回复中说明原因与风险。

### 七、落地规范

1. 主张 ≤ 证据（`CAPABILITY_MATRIX` 为上限）
2. 代码 / `docs/` / `IEEE-conference/` 三件套同步
3. 聚焦 LLM 专用 IR + serving 集成，不与通用 MLIR 拼大而全
4. Agent 分支：`cursor/<topic>-575e`；**每轮迭代收工后从最新 `main` 新建**（见 §六）
5. 里程碑完成 → 合并 PR → `git pull origin main` → 新分支，作为默认工作流

### 八、快速索引

| 任务 | Loop | 文档 |
|------|------|------|
| 修 CI / 新 Pass | Loop 1 | `tests/`, `lib/Dialect/LLM/` |
| 补 prefix 实验 | Loop 1→2 | `docs/E2_PREFIX_SERVING_EVAL.md` |
| 审稿返修 | Loop 2 | `IEEE-conference/REVISION_NOTES.md` |
| 新图进主文 | Loop 2 | `IEEE-conference/figures/FIGURES_NATURE.md` |
| 能否写进摘要 | Loop 2 | `docs/PAPER_REVISION_TRACEABILITY.md` |
| 对标顶会差距 | Loop 2 Step 1 | `docs/PAPER_TOP_TIER_BAR.md` |
