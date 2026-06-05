# E2E examples (P2+)

Numerical check compares the reference interpreter to explicit PyTorch einsum
(decode-step Q length 1, KV length S — not `scaled_dot_product_attention` with
mismatched Q/K lengths).

Run without MLIR build (Python reference only):

```bash
PYTHONPATH=src python examples/e2e/kv_micro_pipeline.py
```

With MLIR optimizer (after building `llmir-opt`):

```bash
export LLMIR_OPT_EXECUTABLE=/path/to/llmir-opt
PYTHONPATH=src python examples/e2e/kv_micro_pipeline.py
```

CLI equivalent:

```bash
llmir-compile --emit-kv-pipeline --run-reference --compare-torch --run-opt -o /tmp/kv.mlir
```
