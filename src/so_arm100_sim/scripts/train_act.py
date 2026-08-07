"""Train ACT on the local dataset via the official lerobot-train entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACT (wraps lerobot-train)")
    parser.add_argument("--config", default="configs/train_act.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu or mps")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={args.config}"]
    if args.steps is not None:
        cmd.append(f"--steps={args.steps}")
    if args.batch_size is not None:
        cmd.append(f"--batch_size={args.batch_size}")
    if args.device is not None:
        cmd.append(f"--policy.device={args.device}")
    if args.output_dir is not None:
        cmd.append(f"--output_dir={args.output_dir}")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
