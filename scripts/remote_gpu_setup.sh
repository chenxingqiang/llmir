#!/usr/bin/env bash
# Bootstrap LLMIR on a GPU host (NVIDIA container / SSH VM).
# Reuses system-site-packages PyTorch when present to avoid multi-GB pip downloads.
set -euo pipefail

ENV_ROOT="${LLMIR_ENV_ROOT:-/root/llmir-env}"
REPO_DIR="${ENV_ROOT}/llmir"
BRANCH="${LLMIR_BRANCH:-cursor/p5-cuda-native-wheel-575e}"
VENV="${ENV_ROOT}/.venv"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${ENV_ROOT}"
cd "${ENV_ROOT}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/chenxingqiang/llmir.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin "${BRANCH}" main 2>/dev/null || git fetch origin
git checkout "${BRANCH}" 2>/dev/null || git checkout -b "${BRANCH}" "origin/${BRANCH}"

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PIP_INDEX_URL="${PYPI_MIRROR}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

pip install -U pip wheel setuptools -q
pip install -e ".[benchmark]" --no-deps -q 2>/dev/null || pip install -e . --no-deps -q
pip install "transformers>=4.40,<5" accelerate safetensors huggingface_hub numpy -q \
  -i "${PYPI_MIRROR}"

cat > "${ENV_ROOT}/activate-llmir.sh" <<'ACTIVATE'
#!/usr/bin/env bash
export LLMIR_ENV_ROOT="${LLMIR_ENV_ROOT:-/root/llmir-env}"
source "${LLMIR_ENV_ROOT}/.venv/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
cd "${LLMIR_ENV_ROOT}/llmir"
ACTIVATE
chmod +x "${ENV_ROOT}/activate-llmir.sh"

python3 - <<'VERIFY'
import torch
import llmir
from llmir.runtime.cuda_probe import summarize_cuda_stack

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.randn(4, 4, device="cuda")
    print("matmul", (x @ x).shape)
print("llmir", getattr(llmir, "__version__", "?"))
print("cuda_stack", summarize_cuda_stack())
print("ENV_OK")
VERIFY

echo "Setup complete. Run: source ${ENV_ROOT}/activate-llmir.sh"
