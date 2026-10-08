#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task=all
config=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="${2:?--task requires documents, sft, or all}"; shift 2 ;;
    --config) config="${2:?--config requires a file}"; shift 2 ;;
    -h|--help)
      echo "Usage: ./scripts/setup_training.sh [--task documents|sft|all] [--config PATH]"
      echo "Creates .venv-training (Python 3.12), installs pinned dependencies, downloads assets, and checks data."
      echo "Set PYTHON_BIN to reuse an existing Python 3.12 environment; HF_TOKEN grants private dataset access."
      exit 0 ;;
    *) echo "Unknown setup argument: $1" >&2; exit 2 ;;
  esac
done
case "$task" in documents|sft|all) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
if [[ -n "$config" && "$task" == all ]]; then
  echo "--config requires --task documents or --task sft." >&2
  exit 2
fi
if [[ -n "$config" ]]; then
  config="$(realpath -- "$config")"
fi
python_bin="${PYTHON_BIN:-$root/.venv-training/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  if [[ -n ${PYTHON_BIN:-} ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
  fi
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 --seed "$root/.venv-training"
  elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$root/.venv-training"
  else
    echo "Install Python 3.12 (with venv) or uv, then rerun setup." >&2
    exit 1
  fi
fi
"$python_bin" - <<'CHECK'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('The pinned training kernels require Python 3.12 on Linux x86_64.')
CHECK
"$python_bin" -m pip install --no-cache-dir --timeout 600 'https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
"$python_bin" -m pip install --no-cache-dir --timeout 600 -r "$root/requirements-training.txt"
"$python_bin" -m pip check
"$python_bin" - <<'CHECK'
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
cd -- "$root"
if [[ "$task" == documents || "$task" == all ]]; then
  selected_config="${config:-$root/configs/lora.yaml}"
  "$python_bin" "$root/scripts/download_training_assets.py" --config "$selected_config"
  PYTHON_BIN="$python_bin" "$root/scripts/train.sh" "$selected_config" --prepare-only
fi
if [[ "$task" == sft || "$task" == all ]]; then
  selected_config="${config:-$root/configs/sft.yaml}"
  "$python_bin" "$root/scripts/download_training_assets.py" --config "$selected_config"
  PYTHON_BIN="$python_bin" "$root/scripts/train_sft.sh" "$selected_config" --prepare-only
fi
