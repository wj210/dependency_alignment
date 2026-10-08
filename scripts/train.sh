#!/usr/bin/env bash
set -euo pipefail
dependency_alignment="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$dependency_alignment/.venv-training/bin/python}"
if [[ -z ${PYTHON_BIN:-} && ! -x "$python_bin" && -x /venv/main/bin/python ]]; then
  python_bin=/venv/main/bin/python
fi
task="${TRAINING_TASK:-documents}"
case "$task" in
  documents) config="$dependency_alignment/configs/lora.yaml" ;;
  sft) config="$dependency_alignment/configs/sft.yaml" ;;
  *) echo "Unknown TRAINING_TASK: $task" >&2; exit 2 ;;
esac
if [[ ${1:-} != --* && $# -gt 0 ]]; then
  config="$1"
  shift
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Training Python not found: $python_bin. Run ./scripts/setup_training.sh, or set PYTHON_BIN." >&2
  exit 1
fi
config="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$config")"
export PYTHONPATH="$dependency_alignment/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd -- "$dependency_alignment"
for arg in "$@"; do
  if [[ "$arg" == --prepare-only || "$arg" == --help || "$arg" == -h ]]; then
    exec "$python_bin" -m dependency_alignment.training.train --config "$config" "$@"
  fi
done
num_processes="$("$python_bin" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["num_processes"])' "$config")"
exec "$python_bin" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$num_processes" --module dependency_alignment.training.train \
  --config "$config" "$@"
