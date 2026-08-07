#!/usr/bin/env bash
# Closed-loop simulation eval for a trained ACT checkpoint.
#
# Usage:
#   scripts/eval_act.sh [checkpoint] [n_episodes] [--ensemble]
#
# Examples:
#   scripts/eval_act.sh                                  # quick eval, 5 episodes, latest checkpoint
#   scripts/eval_act.sh outputs/train_act_50ep_100k/checkpoints/025000/pretrained_model 10 --ensemble
#
# Modes:
#   default    quick chunked eval (temporal ensembling off): one policy call
#              every 100 steps, ~100x faster. Use for mid-training checks.
#   --ensemble official-style eval (temporal ensembling on, coeff 0.01):
#              policy called every step, slower but matches training config.
#
# NOTE: eval competes with training for the MPS GPU, so it will slow a running
# training job. Expect ~0.1-0.3 s per env step in quick mode while training.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-outputs/train_act_50ep_100k/checkpoints/last/pretrained_model}"
EPISODES="${2:-5}"
MODE_FLAGS="--no-ensemble"
if [[ "${3:-}" == "--ensemble" ]]; then
  MODE_FLAGS=""
fi

echo "eval: checkpoint=$CHECKPOINT episodes=$EPISODES mode=${MODE_FLAGS:-ensemble}"
exec env PYTHONPATH=src .venv/bin/python -m so_arm100_sim.scripts.eval_act \
  --checkpoint "$CHECKPOINT" \
  --num-episodes "$EPISODES" \
  $MODE_FLAGS
