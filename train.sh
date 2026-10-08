#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/venv/main/bin/python}"
config="$project_dir/configs/lora.yaml"
if [[ ${1:-} != --* && $# -gt 0 ]]; then
  config="$1"
  shift
fi
config="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$config")"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
num_processes="$("$python_bin" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["num_processes"])' "$config")"
cd -- "$project_dir"
if [[ ${1:-} == --prepare-only ]]; then
  shift
  exec "$python_bin" -m dependency_alignment.training.data --config "$config" "$@"
fi
exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$num_processes" --module dependency_alignment.training.train \
  --config "$config" "$@"
