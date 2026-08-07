#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${1:-mps}"
exec ../.venv-aloha/bin/lerobot-eval \
  --policy.type=act --policy.pretrained_path=outputs/act_aloha_transfer_80k/checkpoints/last/pretrained_model \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=500 --eval.batch_size=50 --policy.device="$DEVICE"
