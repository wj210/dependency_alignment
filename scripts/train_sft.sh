#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAINING_TASK=sft
exec "$root/scripts/train.sh" "$@"
