"""Device resolution for inference compare."""

from llmir.benchmark.device import cuda_available, resolve_inference_device
from llmir.benchmark.inference_compare import InferenceCompareConfig


def test_resolve_cpu():
    cfg = resolve_inference_device("cpu")
    assert cfg.device == "cpu"
    assert cfg.torch_dtype == "float32"


def test_inference_compare_config_dtype_override():
    compare = InferenceCompareConfig(device="cpu", dtype="bfloat16")
    assert compare.resolved.device == "cpu"
    assert compare.resolved.torch_dtype == "bfloat16"


def test_cuda_available_is_bool():
    assert isinstance(cuda_available(), bool)
