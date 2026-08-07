"""Collect scripted demonstrations and write them as a LeRobot dataset.

Every control step stores:

    observation.state          : 5 arm joint positions (rad) + gripper [0,1]
    observation.images.<name>  : uint8 RGB camera frames (mp4-encoded)
    action                     : absolute joint targets (rad) + gripper target

Episodes in which the scripted controller fails are discarded, so the dataset
only contains successful demonstrations.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

from .baseline import SmoothPickPlaceController
from .env import EnvConfig, SoArm100PickEnv, TASK_STR


def build_features(cameras: tuple[str, ...], state_dim: int = 6, action_dim: int = 6) -> dict:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,)},
        "action": {"dtype": "float32", "shape": (action_dim,)},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def collect_dataset(
    repo_id: str = "local/so_arm100_pick",
    root: str | Path = "data",
    num_episodes: int = 20,
    *,
    cameras: tuple[str, ...] = ("top",),
    cube_jitter: float = 0.0,
    seed: int = 0,
    video_codec: str = "h264_videotoolbox",
    keyint: int = 30,
    video_files_size_in_mb: int = 2,
) -> Path:
    """Collect ``num_episodes`` successful demonstrations into ``root/repo_id``.

    ``video_codec`` must be one of the names LeRobot's encoder accepts, e.g.
    ``"h264_videotoolbox"`` (fast Apple hardware H.264 on macOS),
    ``"h264"`` (libx264 software) or ``"libsvtav1"``. All of them decode back
    as plain H.264/AV1, so training never cares which encoder was used.
    """
    cfg = EnvConfig(cameras=cameras, cube_jitter=cube_jitter, seed=seed)
    env = SoArm100PickEnv(cfg)
    # Smooth parabolic trajectories (human-like) produce faster, more natural
    # demonstrations for ACT than the old waypoint state machine.
    controller = SmoothPickPlaceController(env)

    dataset_dir = Path(root) / repo_id
    dataset = LeRobotDataset.create(
        repo_id,
        cfg.fps,
        features=build_features(cameras),
        root=str(dataset_dir),
        robot_type="so101_so_arm100",
        use_videos=True,
        # Frequent keyframes (every 1s) + one small video file per episode make
        # random frame access during training much faster than a single big file
        # with sparse keyframes (the default batching was 200MB per file).
        rgb_encoder=RGBEncoderConfig(vcodec=video_codec, crf=23, preset="medium", g=keyint),
        video_files_size_in_mb=video_files_size_in_mb,
    )

    orig_step = env.step
    frames: list[dict[str, Any]] = []
    prev_obs: dict[str, Any] | None = None

    def step_with_record(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        nonlocal prev_obs
        if prev_obs is None:
            prev_obs = env.get_observation()
        frame = {
            "observation.state": prev_obs["observation.state"],
            "action": np.asarray(action, dtype=np.float32).reshape(-1),
            "task": TASK_STR,
        }
        for cam in cameras:
            frame[f"observation.images.{cam}"] = prev_obs[f"observation.images.{cam}"]
        frames.append(frame)
        dataset.add_frame(frame)
        obs, reward, terminated, truncated, info = orig_step(action)
        prev_obs = obs
        return obs, reward, terminated, truncated, info

    env.step = types.MethodType(step_with_record, env)

    saved = 0
    rng = np.random.default_rng(seed)
    attempts = 0
    max_attempts = num_episodes * 4 + 10
    try:
        while saved < num_episodes and attempts < max_attempts:
            attempts += 1
            frames.clear()
            prev_obs = None  # refetched at the first step after controller reset
            jx = jy = cfg.cube_jitter
            jitter = rng.uniform(-jx, jy, size=2)
            xy = np.array(cfg.cube_default_xy, dtype=float) + jitter
            cube_pos = np.array([xy[0], xy[1], cfg.table_top_z + env.cube_half])
            stats = controller.run(cube_pos=cube_pos)
            if stats["success"] and frames:
                dataset.save_episode()
                saved += 1
                print(f"saved episode {saved}/{num_episodes} "
                      f"({len(frames)} frames, {stats['steps']} steps)")
            else:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                print(f"discarded failed attempt ({stats})")
    finally:
        dataset.finalize()
        env.close()

    print(f"dataset saved to {dataset_dir}")
    return dataset_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect SO-ARM100 pick & place demos")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--repo-id", default="local/so_arm100_pick")
    parser.add_argument("--root", default="data")
    parser.add_argument("--cameras", default="top", help="comma-separated camera names")
    parser.add_argument("--cube-jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--video-codec",
        default="h264_videotoolbox",
        help="e.g. h264_videotoolbox (macOS hardware), h264 (libx264) or libsvtav1",
    )
    args = parser.parse_args()
    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    collect_dataset(
        repo_id=args.repo_id,
        root=args.root,
        num_episodes=args.num_episodes,
        cameras=cameras,
        cube_jitter=args.cube_jitter,
        seed=args.seed,
        video_codec=args.video_codec,
    )


if __name__ == "__main__":
    main()
