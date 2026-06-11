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
llmir-benchmark --model llama3-8b --batch-sizes 1,4
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

**论文实验命名（正文用 E1–E3，CLI 保留 legacy）：**

| ID | 名称 | 验证命令 |
|----|------|----------|
| **E1** | Compile-Time Pass Verification | `pytest tests/test_mvp_a_e2e.py -m "not network"` |
| **E2** | Prefix-Aware Serving Evaluation | `llmir-benchmark --sharegpt-prefix-bench` |
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
| Serving 代理 | prefix 复用、TTFT/prefill token 代理 | E2、`sharegpt_prefix_bench.py` |
| Runtime 集成 | GPU KV 无 NumPy 往返 | E3、`llmir-benchmark --mvp-c-bench` |
| 算子微基准 | attention toy kernel | `benchmark/attention/`（**非** hot path） |

优先级：**P0 正确性** → **P1 主文证据链** → **P2 性能** → **P3 生态**。

**Step 2–5：** 方案拆解 → 最小 diff 开发 → 三层回归（pytest / E1–E3 / 图再生）→ 合并并更新 `CAPABILITY_MATRIX` 与 benchmarks JSON。改动影响论文主张时，同步触发 Loop 2 Step 1。

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
| Serving 代理 | §5 E2、`fig:prefix_ttft` | `sharegpt_2048_sim.json`、`paper_results.json` |
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

### 五、论文对齐验证清单（收工前必跑）

```bash
pytest tests/test_mvp_a_e2e.py tests/test_mvp_c_e2e.py tests/test_sharegpt_prefix_bench.py -m "not network" -q
python3 -c "
import json; from pathlib import Path
p=json.loads(Path('IEEE-conference/benchmarks/paper_results.json').read_text())
assert 'e1' in p; assert Path(p['e1']['mlir_snippet_path']).is_file()
print('paper_results OK')
"
python3 IEEE-conference/figures/generate_all_nature_figures.py
```

**Definition of Done：** traceability 可核对；主文无未标注的 operator/多卡 measured 口吻；图由脚本生成；E1–E3 命名统一。

### 六、落地规范

1. 主张 ≤ 证据（`CAPABILITY_MATRIX` 为上限）
2. 代码 / `docs/` / `IEEE-conference/` 三件套同步
3. 聚焦 LLM 专用 IR + serving 集成，不与通用 MLIR 拼大而全
4. Agent 分支：`cursor/<topic>-575e`

### 七、快速索引

| 任务 | Loop | 文档 |
|------|------|------|
| 修 CI / 新 Pass | Loop 1 | `tests/`, `lib/Dialect/LLM/` |
| 补 prefix 实验 | Loop 1→2 | `docs/E2_PREFIX_SERVING_EVAL.md` |
| 审稿返修 | Loop 2 | `IEEE-conference/REVISION_NOTES.md` |
| 新图进主文 | Loop 2 | `IEEE-conference/figures/FIGURES_NATURE.md` |
| 能否写进摘要 | Loop 2 | `docs/PAPER_REVISION_TRACEABILITY.md` |
