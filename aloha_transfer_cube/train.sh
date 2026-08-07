#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${1:-mps}"
exec ../.venv-aloha/bin/lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.type=act --policy.push_to_hub=false --policy.device="$DEVICE" \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=80000 --batch_size=8 --save_freq=10000 --log_freq=1000 \
  --output_dir=outputs/act_aloha_transfer_80k
