"""Run the scripted pick & place baseline once and (optionally) record a video."""

from __future__ import annotations

import argparse

import numpy as np

from ..baseline import SmoothPickPlaceController
from ..env import EnvConfig, SoArm100PickEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted pick & place baseline")
    parser.add_argument("--cameras", default="top")
    parser.add_argument("--video", default="outputs/baseline.mp4")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-camera",
        default="view",
        help="Camera used for the video (default: 'view', a third-person view)",
    )
    args = parser.parse_args()

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    env = SoArm100PickEnv(EnvConfig(cameras=cameras, seed=args.seed))
    controller = SmoothPickPlaceController(env)

    frames = []
    orig_step = env.step
    if not args.no_video:
        try:
            video_cam_id = env.model.camera(args.video_camera).id
        except Exception:
            video_cam_id = env.camera_ids[cameras[0]]

        def step_with_frames(self, action: np.ndarray):
            frames.append(env.render(video_cam_id))
            return orig_step(action)
        import types
        env.step = types.MethodType(step_with_frames, env)

    stats = controller.run()
    print(f"success={stats['success']}  contact={stats['contact']}  steps={env.step_count}")

    if not args.no_video and frames:
        import imageio.v2 as imageio
        from pathlib import Path
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=env.cfg.fps, codec="libx264", quality=7)
        print(f"video saved to {out}")
    env.close()


if __name__ == "__main__":
    main()
