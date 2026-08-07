"""Evaluate a trained ACT checkpoint in the MuJoCo pick & place environment.

The policy is loaded with LeRobot's standard checkpoint layout
(``checkpoints/last/pretrained_model``) together with its pre/post-processors,
then rolled out in the simulation. A short MP4 of the rollout is written to
``outputs/eval_<name>/rollout.mp4`` and the success rate is reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

from .env import EnvConfig, SoArm100PickEnv, TASK_STR


def obs_to_batch(
    obs: dict, cameras: tuple[str, ...], device: str, resize: int = 224
) -> dict[str, torch.Tensor]:
    """Convert env observations to the batch format expected by the policy."""
    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(device)
    }
    for cam in cameras:
        img = torch.from_numpy(obs[f"observation.images.{cam}"]).float()
        img = einops.rearrange(img, "h w c -> 1 c h w") / 255.0
        if resize:
            img = F.interpolate(img, size=(resize, resize), mode="bilinear", align_corners=False)
        batch[f"observation.images.{cam}"] = img.to(device)
    return batch


def run_eval(
    checkpoint_dir: str | Path,
    *,
    cameras: tuple[str, ...] = ("top",),
    n_episodes: int = 5,
    max_steps: int = 1200,
    record_video: bool = True,
    seed: int = 0,
    cube_jitter: float = 0.01,
    resize: int = 224,
    ensemble: bool = True,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    policy = ACTPolicy.from_pretrained(checkpoint_dir)
    if not ensemble:
        # Quick eval mode: drop temporal ensembling and run chunked inference
        # (one policy call every 100 steps instead of every step). ~100x fewer
        # policy calls; fine for mid-training checks, use ensemble=True for the
        # final official-style numbers.
        policy.config.temporal_ensemble_coeff = None
        policy.config.n_action_steps = 100
    policy.eval()
    device = str(policy.config.device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    env = SoArm100PickEnv(EnvConfig(cameras=cameras, seed=seed, cube_jitter=cube_jitter))
    video_frames: list[np.ndarray] = []
    successes = 0
    results = []

    for ep in range(n_episodes):
        obs = env.reset()
        policy.reset()
        ep_frames = []
        done = False
        for step in range(max_steps):
            if record_video:
                ep_frames.append(obs["observation.images.top"])
            batch = obs_to_batch(obs, cameras, device, resize=resize)
            batch = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action_np = action.to("cpu").numpy().reshape(-1)
            obs, reward, terminated, truncated, info = env.step(action_np)
            if terminated or truncated:
                done = True
                break
        successes += int(info["success"])
        results.append({"episode": ep, "success": bool(info["success"]), "steps": env.step_count})
        if ep == 0 and record_video and ep_frames:
            video_frames = ep_frames
        print(f"episode {ep}: success={info['success']} steps={env.step_count}")

    if record_video and video_frames:
        out_dir = Path("outputs") / f"eval_{Path(checkpoint_dir).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_mp4(video_frames, out_dir / "rollout.mp4", fps=env.cfg.fps)
        print(f"rollout video saved to {out_dir / 'rollout.mp4'}")

    env.close()
    summary = {"success_rate": successes / n_episodes, "results": results}
    print(f"success rate: {successes}/{n_episodes}")
    return summary


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ACT policy in simulation")
    parser.add_argument("--checkpoint", default="outputs/train_act_50ep/checkpoints/last/pretrained_model")
    parser.add_argument("--cameras", default="top")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cube-jitter",
        type=float,
        default=0.01,
        help="Uniform cube position jitter in meters (default 0.01 = +/-1cm)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Resize camera images to NxN before the policy (must match training; 0 disables)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable temporal ensembling (fast chunked eval; ~100x fewer policy calls)",
    )
    args = parser.parse_args()
    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    run_eval(
        args.checkpoint,
        cameras=cameras,
        n_episodes=args.num_episodes,
        max_steps=args.max_steps,
        record_video=not args.no_video,
        seed=args.seed,
        cube_jitter=args.cube_jitter,
        resize=args.resize,
        ensemble=not args.no_ensemble,
    )


if __name__ == "__main__":
    main()
