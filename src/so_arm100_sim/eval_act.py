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
    video_camera: str = "view",
    tail_steps: int = 45,
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
    # Record the video from a dedicated third-person camera (clearer than the
    # overhead observation camera). Falls back to the first observation camera
    # if the scene has no such camera.
    try:
        env.model.camera(video_camera)
        video_cam_id = env.model.camera(video_camera).id
    except Exception:
        video_cam_id = env.camera_ids[cameras[0]]
    out_dir = Path("outputs") / f"eval_{Path(checkpoint_dir).stem}"
    if record_video:
        out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    results = []

    for ep in range(n_episodes):
        obs = env.reset()
        policy.reset()
        ep_writer = None
        if record_video:
            ep_path = out_dir / f"episode_{ep:03d}.mp4"
            import imageio.v2 as imageio
            ep_writer = imageio.get_writer(ep_path, fps=env.cfg.fps, codec="libx264", quality=7)
        done = False
        for step in range(max_steps):
            if record_video:
                ep_writer.append_data(env.render(video_cam_id))
            batch = obs_to_batch(obs, cameras, device, resize=resize)
            batch = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action_np = action.to("cpu").numpy().reshape(-1)
            obs, reward, terminated, truncated, info = env.step(action_np)
            if terminated or truncated:
                if record_video:
                    # The loop records pre-action frames, so the release that
                    # triggers success would never appear in the video. Append
                    # the post-step state (cube placed) plus a short static
                    # tail so the ending is visible.
                    final_frame = env.render(video_cam_id)
                    ep_writer.append_data(final_frame)
                    for _ in range(tail_steps - 1):
                        ep_writer.append_data(final_frame)
                done = True
                break
        successes += int(info["success"])
        results.append({"episode": ep, "success": bool(info["success"]), "steps": env.step_count})
        if record_video:
            ep_writer.close()
            # Keep a canonical rollout.mp4 for the first episode (backwards
            # compatible with existing docs/scripts).
            if ep == 0:
                import shutil
                shutil.copyfile(ep_path, out_dir / "rollout.mp4")
            print(f"video saved: {ep_path}")
        print(f"episode {ep}: success={info['success']} steps={env.step_count}")

    env.close()
    summary = {"success_rate": successes / n_episodes, "results": results}
    print(f"success rate: {successes}/{n_episodes}")
    return summary


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
    parser.add_argument(
        "--video-camera",
        default="view",
        help="Camera used for the rollout video (default: 'view', a third-person view)",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=45,
        help="Static tail frames appended after episode end (1.5s at 30fps)",
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
        video_camera=args.video_camera,
        tail_steps=args.tail_steps,
    )


if __name__ == "__main__":
    main()
