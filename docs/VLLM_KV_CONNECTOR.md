# vLLM KV connector (P4)

LLMIR provides an experimental **vLLM V1 KV connector** that stores prompt-prefix K/V on disk and reloads them into vLLM's paged buffer (similar to vLLM's `ExampleConnector`, but keyed by LLMIR prefix hashing).

## Register

```python
from llmir.integration.vllm_connector import (
    register_llmir_vllm_connector,
    build_kv_transfer_extra_config,
    CONNECTOR_NAME,
)

register_llmir_vllm_connector()
```

Or point vLLM at the module directly:

```python
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="LLMIRConnector",
    kv_connector_module_path="llmir.integration.vllm_connector",
    kv_role="kv_both",
    kv_connector_extra_config=build_kv_transfer_extra_config(
        storage_path="/tmp/llmir_vllm_kv",
        min_prefix_length=4,
    ),
)
```

## Smoke test

```bash
python scripts/vllm_kv_connector_smoke.py
python scripts/vllm_kv_connector_smoke.py --register  # when vLLM is installed
```

## GPU inference compare

```bash
llmir-benchmark --compare hf,vllm,llmir-paged \
  --model facebook/opt-125m \
  --compare-device auto \
  --compare-dtype auto \
  -o compare.json

python scripts/gpu_inference_compare.py --device cuda --backends hf,llmir-paged
```

`auto` selects CUDA when `torch.cuda.is_available()`.

## Status

- **MVP**: disk-backed prefix store + scheduler/worker hooks aligned with vLLM V1 connector API.
- **Not yet**: NIXL/RDMA transport, native `libMLIRLLMRuntime` block copy into vLLM pools.
- Requires a recent vLLM build with `KVConnectorBase_V1` (see vLLM disaggregated prefill docs).
