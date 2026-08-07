"""lerobot-eval wrapper that appends a static tail to every rendered video.

LeRobot's eval videos end exactly when the episode terminates (success), so the
"cube held stable" state right after the handoff is only visible for a single
frame. This wrapper monkeypatches ``write_video`` to duplicate the final frame
for ``tail_frames`` extra frames (default 75 at 50 fps = 1.5 s), making the
ending easier to watch.

Usage (identical to ``lerobot-eval``):
    python eval_with_tail.py --policy.type=act \
        --policy.pretrained_path=<ckpt>/pretrained_model \
        --env.type=aloha --env.task=AlohaTransferCube-v0 \
        --eval.n_episodes=20 --eval.batch_size=8 --eval.use_async_envs=false \
        --policy.device=cuda

Set ``EVAL_TAIL_FRAMES`` to change the tail length (0 disables the tail).
"""

from __future__ import annotations

import os

import numpy as np

import lerobot.scripts.lerobot_eval as lerobot_eval

TAIL_FRAMES = int(os.environ.get("EVAL_TAIL_FRAMES", "75"))

_original_write_video = lerobot_eval.write_video


def _write_video_with_tail(video_path, stacked_frames, fps, *args, **kwargs):
    frames = np.asarray(stacked_frames)
    if TAIL_FRAMES > 0 and frames.ndim == 4 and len(frames) > 0:
        tail = np.repeat(frames[-1:], TAIL_FRAMES, axis=0)
        frames = np.concatenate([frames, tail], axis=0)
    return _original_write_video(video_path, frames, fps, *args, **kwargs)


lerobot_eval.write_video = _write_video_with_tail

from lerobot.scripts.lerobot_eval import main  # noqa: E402


if __name__ == "__main__":
    main()
