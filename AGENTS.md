# AGENTS.md

Guidance for AI coding agents working in this repository.

For **iteration loops (engineering + paper alignment)**, see [`Agent.md`](Agent.md) § LOOPs.

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
