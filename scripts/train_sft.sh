#!/usr/bin/env bash
set -euo pipefail
# Overrides are forwarded to the shared trainer, e.g.:
# ./scripts/train_sft.sh configs/sft_chat.yaml --epochs 1
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAINING_TASK=sft
exec "$root/scripts/train.sh" "$@"
