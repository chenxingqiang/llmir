# MLIR / Native Runtime Build Guide

In-tree `mlir-opt` (with LLM dialect passes) and `libMLIRLLMRuntime` require a
**full LLVM + MLIR development tree**. The pip wheel does not ship these binaries.

## Quick check

```bash
bash scripts/check_native_build_prereqs.sh
bash scripts/check_native_build_prereqs.sh --strict   # exit 1 if llvm missing
```

## Option A — LLVM monorepo (recommended for lit closure)

1. Clone [llvm-project](https://github.com/llvm/llvm-project) at a release matching this tree.
2. Place or symlink this repository as `llvm-project/mlir` **or** build LLVM first and set `LLVM_DIR`.
3. Configure with `LLVM_ENABLE_PROJECTS=mlir` and LLMIR CMake flags (`LLMIR_ENABLE_CUDA` optional).

Typical outline:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mlir-opt
```

Then point LLMIR scripts at the built `mlir-opt`:

```bash
export LLVM_DIR="$PWD/build/lib/cmake/llvm"
export PATH="$PWD/build/bin:$PATH"
cd /path/to/llmir
bash scripts/build_mlir_opt.sh
bash scripts/mlir_lit_smoke.sh
```

## Option B — System LLVM dev packages

When `llvm-config` and `LLVMConfig.cmake` are on the system:

```bash
export LLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm   # distro-specific
bash scripts/build_native_runtime.sh
bash scripts/build_mlir_opt.sh
```

Stock distro `mlir-opt` **does not** register `-llm-optimize-kv-cache`; you must use the **in-tree** binary.

## Targets

| Script | Target | Use |
|--------|--------|-----|
| `build_mlir_opt.sh` | `mlir-opt` | MLIR lit suite (M8) |
| `build_native_runtime.sh` | `libMLIRLLMRuntime.so` | `LLMIR_LIB_PATH`, native KV |

Both reuse `BUILD_DIR` (default `build-native/`).

## Lab smoke

```bash
bash scripts/lab_smoke_all.sh
cat IEEE-conference/benchmarks/lab_status_summary.json
```

## CI

- `native-runtime.yml` — best-effort native build on `ubuntu-latest`
- `mlir-lit-lab.yml` — pass `mlir_opt_executable` from a lab runner with a built opt

See also: `docs/MLIR_LIT_RUNBOOK.md`, `docs/P5_CUDA_NATIVE.md`.
