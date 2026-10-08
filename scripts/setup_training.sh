#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv-training"
  PYTHON="$ROOT/.venv-training/bin/python"
fi
"$PYTHON" - <<'CHECK'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.machine() != 'x86_64':
    raise SystemExit('This reproducible kernel wheel requires Python 3.12 on Linux x86_64.')
CHECK
"$PYTHON" -m pip install --no-cache-dir --timeout 600 'https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
"$PYTHON" -m pip install --no-cache-dir --timeout 600 -r "$ROOT/requirements-training.txt"
"$PYTHON" -m pip check
"$PYTHON" - <<'CHECK'
import torch
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.is_bf16_supported(), 'BF16 is unsupported'
x = torch.randn(1, 16, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(16, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y = causal_conv1d_fn(x, w, activation='silu')
y.float().sum().backward()
torch.cuda.synchronize()
print('Training environment and CUDA convolution ready:', torch.__version__, torch.version.cuda)
CHECK
"$PYTHON" "$ROOT/scripts/download_training_assets.py"
PYTHON_BIN="$PYTHON" "$ROOT/train.sh" --prepare-only
