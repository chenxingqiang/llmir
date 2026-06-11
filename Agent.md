# Agent.md — LLMIR 双循环迭代与论文对齐手册

面向 **LLMIR（Large Language Model Intermediate Representation）** 开源运维与学术论文撰写，沿用**工程叙事、闭环迭代、问题驱动**文风。Agent 执行任务时，先判断当前处于哪条 Loop、哪一步，再动手改代码或改论文。

**权威状态锚点（改论文前必读）：**

| 文档 | 用途 |
|------|------|
| [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) | 功能真实实现程度（C++ / Python ref / Demo / Planned） |
| [`docs/PAPER_REVISION_TRACEABILITY.md`](docs/PAPER_REVISION_TRACEABILITY.md) | 论文主张 ↔ 仓库证据（E1–E3、JSON、测试） |
| [`IEEE-conference/REVISION_NOTES.md`](IEEE-conference/REVISION_NOTES.md) | 审稿意见映射与「已验证 / 投影 / 未来工作」边界 |
| [`IEEE-conference/LLMIR-paper-ICCD2025-revised.tex`](IEEE-conference/LLMIR-paper-ICCD2025-revised.tex) | 当前修订稿主文件 |

---

# LOOPs

## 一、整体说明

LLMIR 是面向 **LLM 推理** 的专用 MLIR 方言与编译/runtime 栈，核心能力包括：LLM 专用 IR 类型与算子、KV/PagedAttention 编译 Pass、Lowering 到 runtime 调用、薄 runtime + `llmir_paged` serving 集成。

本仓库设计 **两套可联动迭代循环**：

| Loop | 定位 | 周期 | 终止条件 |
|------|------|------|----------|
| **Loop 1 · 工程迭代** | 代码、性能、功能、多硬件适配 | 周/月，**无限循环** | 无（开源长期演进） |
| **Loop 2 · 论文迭代** | 撰写、实验、评审、返修、定稿 | 日/周（初稿）或 月/季（投稿） | 录用或转投重构 |

**论文对齐原则（贯穿 Loop 2）：** 主文只写 **CI 可复现 + JSON 可查** 的结论；算子级 speedup、多卡吞吐、全模型对标等，除非 harness 落地，否则只能进附录并标注 *projected / future work*。

**论文实验命名（正文用 E1–E3，CLI 保留 legacy）：**

| ID | 名称 | 验证命令 |
|----|------|----------|
| **E1** | Compile-Time Pass Verification | `pytest tests/test_mvp_a_e2e.py -m "not network"` |
| **E2** | Prefix-Aware Serving Evaluation | `llmir-benchmark --sharegpt-prefix-bench` |
| **E3** | GPU-Resident KV Integration | `pytest tests/test_mvp_c_e2e.py -m "not network"` |

---

## 二、Loop 1：LLMIR 工程持续迭代闭环（无限循环）

### 核心定位

以 **实测数据 + Issues/用户反馈** 驱动 `src/llmir/`、MLIR dialect、runtime 与 benchmark 脚本演进。不追求单轮「大而全」，每轮只收敛 1–3 个可验证目标。

### 单轮流程

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

#### Step 1 · 基线评测 & 瓶颈诊断（感知层）

**现象前置：** 先跑通当前稳定版本的「最小可信基准」，再谈优化。

```bash
# Python 离线回归（CI 同款）
pytest tests/ -m "not network" -q

# E1–E3 论文 harness
pytest tests/test_mvp_a_e2e.py tests/test_mvp_c_e2e.py tests/test_sharegpt_prefix_bench.py -m "not network" -q

# 采集主文 JSON
python3 scripts/paper_benchmark_collect.py --model gpt2
```

**对照维度（按仓库现实分层，勿混口径）：**

| 层级 | 测什么 | 典型入口 |
|------|--------|----------|
| 编译 Pass | block size 改写、reference 数值一致 | E1、`llmir-compile --mvp-a-e2e` |
| Serving 代理 | prefix 复用、TTFT/prefill token 代理 | E2、`sharegpt_prefix_bench.py` |
| Runtime 集成 | GPU KV 无 NumPy 往返、`llmir_paged` decode | E3、`llmir-benchmark --mvp-c-bench` |
| 算子微基准 | attention toy kernel | `benchmark/attention/`（**非** hot path，不可写进主文） |

**问题归类 → 优先级：**

- **P0 正确性**：编译崩溃、数值不一致、测试红
- **P1 主文证据链**：E1–E3 任一断裂 → 优先修 harness，再改论文
- **P2 性能**：在正确性之上做 GPU/多硬件收益
- **P3 生态**：文档、vLLM connector、native build 路径

**输出：** `issues` 列表 + 本轮 Top-3 任务 + 是否影响论文表述（是/否）。

#### Step 2 · 方案设计 & 任务拆解（规划层）

针对诊断结果选 **一类主战场**，避免并行堆功能：

1. **编译/IR**：新 Pass、Lowering、lit 用例、`BlockSizeAnalysis` 类优化
2. **Runtime/Serving**：`PagedKVDecoder`、`TorchGpuPagedKVCache`、prefix store
3. **Benchmark/Harness**：新 JSON schema、可复现 CLI、图表脚本
4. **硬件后端**：CUDA native、`LLMIR_KV_BACKEND=native`（可选构建）

**输出：** 分支名、`CAPABILITY_MATRIX` 拟更新行、验收命令（必须可 `pytest` 或产出 JSON）。

#### Step 3 · 代码开发 & 逻辑实现（落地层）

- Python 改动：`src/llmir/`，匹配现有风格，最小 diff
- C++/MLIR：`lib/Dialect/LLM/`，同步 `test/Dialect/LLM/*.mlir`
- 新能力默认 **先 reference Python 路径**，再挂 native

本地门槛：`pytest` 相关用例绿 + `ruff`/`black` 不过度恶化。

#### Step 4 · 全量回归 & 验证（验证层）

三层回归：

1. **功能**：`pytest tests/ -m "not network"`
2. **论文 harness**：E1–E3 命令 + `paper_benchmark_collect.py`
3. **图表**：`python3 IEEE-conference/figures/generate_all_nature_figures.py`

退化（性能回退、新测试失败、JSON 字段断裂）→ **退回 Step 2**。  
若改动影响论文主张 → **同步触发 Loop 2 Step 1（短板定位）**。

#### Step 5 · 版本发布 & 经验沉淀（归档层）

1. 合并 PR、更新 `CHANGELOG` / README / `scripts/README.md`
2. 更新 [`docs/CAPABILITY_MATRIX.md`](docs/CAPABILITY_MATRIX.md) 状态列
3. 若产出新证据：写入 `IEEE-conference/benchmarks/*.json`，更新 traceability 表
4. 可复用模式记入 docs（如 E1/E2/E3 文档模板）

**→ 回到 Step 1**，开启下一轮。

### Loop 1 特点

- **数据驱动**：无 JSON / 无测试 = 不可对外声称
- **分层诚实**：integration 代理 ≠ operator kernel 优势
- **长效运维**：周迭代保 CI 绿，月迭代扩硬件/模型覆盖面

---

## 三、Loop 2：LLMIR 学术论文迭代闭环

论文循环的核心职责：**把文章与仓库真实能力对齐**，而不是用投影数字撑主文。

### 场景 A：初稿打磨循环（短循环 · 高频）

适用于框架搭建、实验补全、内部评审。

```
Step 1 框架 & 短板定位
    ↓
Step 2 实验 & 数据更新
    ↓
Step 3 行文 & 逻辑重构
    ↓
Step 4 内部评审 & 修正  ──实验不足──→ 回到 Step 2
    ↓                      ──结构问题──→ 回到 Step 1
初稿定稿
```

#### Step 1 · 框架梳理 & 短板定位

**对照三连表（Agent 必做）：**

1. **贡献点** ↔ 论文 §X 是否_each_ 有 harness？
2. **CAPABILITY_MATRIX** ↔ 文中是否把 Planned 写成 Implemented？
3. **REVISION_NOTES 审稿清单** ↔ 未闭合项是否仍出现在摘要/结论？

**主文允许的证据类型：**

| 类型 | 论文位置 | 仓库要求 |
|------|----------|----------|
| 编译验证 | §5 E1、Listing `lst:e1_mlir` | `gpt2_e1_snippet.mlir`、E1 pytest |
| Serving 代理 | §5 E2、Fig. `fig:prefix_ttft` | `sharegpt_2048_sim.json`、`paper_results.json` |
| GPU KV 集成 | §5 E3、Fig. `fig:e1_e3_eval` panel c | E3 pytest；panel c 须标 illustrative |
| 诚实 decode 基线 | Table `tab:measured_harness` | 仅 gpt2 实测 + Qwen **引用** |
| 算子/多模型/多卡 | Appendix `app:future_ops` / `app:projected` | 必须脚注 *not measured in CI* |

**输出：** 章节短板清单（缺实验 / 夸大 / 术语不一 / 图过期）。

#### Step 2 · 实验补充 & 数据更新

**原则：** 先补 **E1–E3 证据链**，再考虑扩展实验；新数据必须进 `IEEE-conference/benchmarks/` 或可追溯脚本。

```bash
# 重采集主文 JSON
python3 scripts/paper_benchmark_collect.py --model gpt2

# 重生成实测图（禁止手改 PDF 数组）
python3 IEEE-conference/figures/generate_all_nature_figures.py

# 投影/插图仅附录
python3 IEEE-conference/figures/generate_projected_figures.py
```

实验分层写法：

- **实测（主文）**：E1–E3 + gpt2 integration proxy
- **引用（主文）**：第三方官方 benchmark（`external_baselines.json`）
- **设计目标（附录）**：Table II 热力图、多卡表——标明 illustrative
- **未来工作（附录）**：`benchmark/attention/` 微基准——标明非 LLMIR lowered kernel

**输出：** 更新 JSON、图、`\ref{fig:e1_e3_eval}` 等交叉引用；同步 `PAPER_REVISION_TRACEABILITY.md`。

#### Step 3 · 行文重构 & 逻辑优化

主线固定为：

> **问题（runtime-only 优化局限）→ LLM 专用 IR / Pass → 可验证 serving 路径 → 诚实边界（何处未测）**

术语统一：

- 正文：**E1 / E2 / E3**（首次给出英文全称）
- 仓库 CLI：脚注 legacy `mvp-*` 映射
- 禁止：主文写「全面优于 vLLM/FlashAttention」而无同 harness 对照

#### Step 4 · 内部评审 & 问题修正

模拟审稿人四连问：

1. 这条结论对应的 **命令 + 文件** 是什么？
2. 关掉 LLMIR 后，优势是否仍存在？（消融）
3. 数字是 **LLMIR 跑出来的** 还是 **设计目标/引用**？
4. 图表能否一键再生？

- 行文/逻辑问题 → Step 3 当场改
- 实验缺陷 → **退回 Step 2**
- 贡献点漂移 → **退回 Step 1**

---

### 场景 B：投稿 & 返修循环（长循环 · 会议/期刊）

```
Step 1 排版 & 投稿
    ↓
Step 2 拆解审稿意见
    ↓
Step 3 定向整改（论文 ↔ 工程联动）
    ↓
Step 4 返修稿 & rebuttal
    ↓
Step 5 二次评审  ──大修/小修──→ 回到 Step 2
                  ──拒稿──────→ 重构框架或转投
录用 → 开源 artifact 定版
```

#### Step 2 · 意见四类分拣

| 类型 | 论文侧 | 工程侧（Loop 1 输入） |
|------|--------|------------------------|
| 创新性质疑 | 强化 IR 语义 vs 通用 MLIR/TVM | 补 dialect 用例、lit |
| 实验不足 | 缩主张或补 harness | 新 benchmark 脚本、CI job |
| 原理漏洞 | 补算法/类型不变量 | Pass 实现与测试 |
| 格式行文 | 直接改 tex | 一般不需要代码 |

#### Step 3 · 定向整改 & 工程联动（关键）

**每条审稿意见必须映射为「论文修改 + 证据产物」：**

```
审稿意见 #N
  → revised.tex 改动位置
  → 新/旧 JSON 或 pytest
  → REVISION_NOTES 一行状态（Verified / Projected / Planned）
```

禁止只做文字辩护而不补 artifact。若短期内无法补实验 → **收窄论文 claim**，而非保留夸大表述。

#### Step 4 · Rebuttal 写法

- 一条意见 ↔ 一段回复 ↔ 一个证据（命令输出、图、commit hash）
- 复杂问题：补充材料放附录 + 仓库路径

---

## 四、双循环联动机制

```
                    ┌─────────────────┐
    论文创新点 ────→│  Loop 1 工程     │
    （新 Pass/IR）   │  Step 2–3 实现  │
                    └────────┬────────┘
                             │ JSON / pytest / 图
                             ▼
                    ┌─────────────────┐
                    │  Loop 2 论文     │
                    │  Step 2 更新 §5  │
                    └────────┬────────┘
                             │ 未测清的能力
                             ▼
                    ┌─────────────────┐
    收窄主张 ←──────│  REVISION_NOTES  │
    或补 harness    │  + CAPABILITY    │
                    └─────────────────┘
```

| 方向 | 触发条件 | 动作 |
|------|----------|------|
| **工程 → 论文** | 新合并 E1/E2/E3 相关 PR | 更新 tex §5、traceability、regen figures |
| **论文 → 工程** | 审稿要求新实验 | 在 Loop 1 开 harness 分支，**先绿测试再写进主文** |
| **口径锁定** | 任一循环 | 同一模型（如 gpt2）、同一 JSON schema、同一图生成脚本 |

**节奏建议：**

- Loop 1：周迭代（CI + E1–E3 不红）
- Loop 2：随投稿节点；每次改 tex 前跑一遍验证清单（见下）

---

## 五、论文对齐验证清单（Agent 收工前必跑）

```bash
# 1. 证据链测试
pytest tests/test_mvp_a_e2e.py tests/test_mvp_c_e2e.py tests/test_sharegpt_prefix_bench.py -m "not network" -q

# 2. 主文 JSON 与 E1 字段
python3 -c "
import json; from pathlib import Path
p=json.loads(Path('IEEE-conference/benchmarks/paper_results.json').read_text())
assert 'e1' in p; assert Path(p['e1']['mlir_snippet_path']).is_file()
print('paper_results OK')
"

# 3. 实测图再生
python3 IEEE-conference/figures/generate_all_nature_figures.py

# 4. 无 stale 引用（不应出现已废弃名）
# mvp_evaluation / fig:mvp_eval / gpt2_mvp_a_snippet → 应已改为 e1_e3_* / gpt2_e1_*
```

**论文改动的 Definition of Done：**

- [ ] 摘要/结论每条 **Verified** 主张可在 traceability 表找到
- [ ] 主文无 operator-level / 多卡吞吐 **未标注** 的 measured 口吻
- [ ] 图表由脚本生成，非手填数组（`paper-only/` 仅附录且已标注）
- [ ] `CAPABILITY_MATRIX` 与论文 §3–§4 能力描述一致
- [ ] E1–E3 命名统一；CLI legacy 仅脚注

---

## 六、落地执行规范

1. **叙事统一**：现象/问题 → 根因 → 方案 → 可复现验证；避免教科书式空泛定义
2. **主张 ≤ 证据**：`CAPABILITY_MATRIX` 是上限；论文不能超过矩阵
3. **三件套同步**：代码 / `docs/` / `IEEE-conference/` 同一 PR 或连续 PR，避免论文领先仓库
4. **聚焦差异化**：LLM 专用 IR（PagedKV、KV Pass、compile-time block sizing）+ serving 集成路径；不与通用 MLIR 拼大而全
5. **硬件分层**：CPU 集成代理、GPU KV、native CUDA 分写；不混为「全面 GPU 优势」
6. **Agent 分支命名**：`cursor/<topic>-575e`；改论文与 harness 同一分支，便于 traceability

---

## 七、快速索引

| 任务 | 进入哪条 Loop | 首选文档 |
|------|---------------|----------|
| 修 CI / 新 Pass | Loop 1 Step 3 | `tests/`, `lib/Dialect/LLM/` |
| 补 prefix 实验 | Loop 1→2 | `docs/E2_PREFIX_SERVING_EVAL.md` |
| 审稿返修缩表 | Loop 2 Step 3 | `IEEE-conference/REVISION_NOTES.md` |
| 新图进主文 | Loop 2 Step 2 | `FIGURES_NATURE.md`, `generate_all_nature_figures.py` |
| 判断能否写进摘要 | Loop 2 Step 1 | `PAPER_REVISION_TRACEABILITY.md` |
