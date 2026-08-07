"""MuJoCo pick-and-place environment for the SO-ARM100 (SO-101) robot.

The environment exposes the same observation/action vocabulary as LeRobot's
SO-100/101 robot configs:

    state  : [q1..q5 (rad), gripper (0=closed, 1=open)]
    action : [dq1..dq5 targets (rad), gripper target (0..1)]

and camera observations ``observation.images.<name>`` as uint8 RGB arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_PATH = PROJECT_ROOT / "assets" / "so_arm100" / "so_arm100_pick.xml"

# Joint indices (qpos order from the official MJCF).
ARM_JOINT_IDS = [0, 1, 2, 3, 4]  # base yaw, shoulder, elbow, wrist pitch, wrist roll
GRIPPER_JOINT_ID = 5

# Gripper slide limits (linear parallel-jaw gripper; 0=closed, 1=open).
GRIPPER_CLOSED_ANGLE = -0.045
GRIPPER_OPEN_ANGLE = 0.045

TASK_STR = "Pick up the red cube and place it on the green target"


def gripper_norm_to_angle(g: float | np.ndarray) -> np.ndarray:
    """Map gripper value in [0, 1] (0=closed, 1=open) to jaw joint angle."""
    g = np.asarray(g, dtype=float)
    return GRIPPER_CLOSED_ANGLE + g * (GRIPPER_OPEN_ANGLE - GRIPPER_CLOSED_ANGLE)


def gripper_angle_to_norm(angle: float | np.ndarray) -> np.ndarray:
    """Map jaw joint angle to gripper value in [0, 1]."""
    angle = np.asarray(angle, dtype=float)
    return (angle - GRIPPER_CLOSED_ANGLE) / (GRIPPER_OPEN_ANGLE - GRIPPER_CLOSED_ANGLE)


@dataclass
class EnvConfig:
    scene_path: Path = SCENE_PATH
    fps: int = 30
    image_height: int = 480
    image_width: int = 640
    cameras: tuple[str, ...] = ("top",)
    # Demo episodes take ~900 steps; keep the eval cap above that so a trained
    # policy is not truncated before it can finish the pick & place.
    max_episode_steps: int = 1200
    # Workspace geometry.
    table_top_z: float = 0.0
    cube_default_xy: tuple[float, float] = (0.02, -0.34)
    cube_size: float = 0.03
    target_xy: tuple[float, float] = (0.06, -0.36)
    target_radius: float = 0.05
    # Randomization.
    cube_jitter: float = 0.0
    seed: int | None = None
    # Home posture (zero config of the mounted arm is collision-free).
    home_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class SoArm100PickEnv:
    """Single-arm pick & place environment built on the official SO-101 MJCF."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.scene_path))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(self.cfg.seed)

        self.site_gripper = self.model.site("gripper").id
        self.body_cube = self.model.body("cube").id
        self.body_gripper = self.model.body("gripper").id
        self.body_moving_jaw = self.model.body("moving_jaw_so101_v1").id
        self.geom_cube = self.model.geom("cube_geom").id
        self.table_top_z = self.cfg.table_top_z
        self.cube_half = self.cfg.cube_size / 2

        self.decimation = max(1, int(round(1.0 / (self.cfg.fps * self.model.opt.timestep))))
        self.camera_ids = {name: self.model.camera(name).id for name in self.cfg.cameras}

        self._renderer = None
        if self.camera_ids:
            self._renderer = mujoco.Renderer(
                self.model, self.cfg.image_height, self.cfg.image_width
            )

        self._gripper_bodies = {self.body_gripper, self.body_moving_jaw}
        self.cube_initial_z = 0.0
        self.step_count = 0
        self._lift_hold_counter = 0
        self._was_lifted = False
        # Offset from the "gripper" site to the midpoint of the two fingers,
        # expressed in the site frame (computed once at the home configuration).
        self.data.qpos[GRIPPER_JOINT_ID] = GRIPPER_OPEN_ANGLE  # approach with the gripper open
        mujoco.mj_forward(self.model, self.data)
        centers = []
        for bid in (self.body_gripper, self.body_moving_jaw):
            for gi in range(self.model.ngeom):
                if self.model.geom_bodyid[gi] == bid and self.model.geom(gi).type == 6:
                    centers.append(self.data.geom_xpos[gi].copy())
        if centers:
            mid = np.mean(centers, axis=0)
            site_rot = self.data.site_xmat[self.site_gripper].reshape(3, 3)
            self.finger_offset = site_rot.T @ (mid - self.data.site_xpos[self.site_gripper])
        else:
            self.finger_offset = np.zeros(3)
        self.data.qpos[GRIPPER_JOINT_ID] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def reset(self, *, cube_pos: np.ndarray | None = None, seed: int | None = None) -> dict[str, Any]:
        """Reset the simulation and return an observation dict."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.data.qpos[:6] = np.array(self.cfg.home_qpos, dtype=float)
        np.copyto(self.data.qvel, np.zeros(self.model.nv))
        self.data.ctrl[:] = np.array(self.cfg.home_qpos, dtype=float)

        if cube_pos is None:
            jx, jy = self.cfg.cube_jitter, self.cfg.cube_jitter
            jitter = self._rng.uniform(-jx, jy, size=2)
            xy = np.array(self.cfg.cube_default_xy, dtype=float) + jitter
            cube_pos = np.array([xy[0], xy[1], self.table_top_z + self.cube_half])
        self.cube_pos = np.asarray(cube_pos, dtype=float)

        # Place the cube (reset its pose/velocity).
        self.data.qpos[6:9] = self.cube_pos
        self.data.qpos[9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        self.data.qvel[6:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        # Let the cube settle on the table.
        for _ in range(int(round(0.5 / (self.decimation * self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

        self.cube_initial_z = self.data.xpos[self.body_cube][2]
        self.step_count = 0
        self._lift_hold_counter = 0
        self._was_lifted = False
        self._update_wrist_camera()
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one control action (arm joint targets + gripper target)."""
        action = np.asarray(action, dtype=float).reshape(-1)
        ctrl = np.array(self.cfg.home_qpos, dtype=float)
        ctrl[ARM_JOINT_IDS] = action[:5]
        ctrl[GRIPPER_JOINT_ID] = gripper_norm_to_angle(np.clip(action[5], 0.0, 1.0))
        np.copyto(self.data.ctrl, ctrl)

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._update_wrist_camera()

        obs = self.get_observation()
        success, lifted = self._eval_success()
        terminated = success
        truncated = self.step_count >= self.cfg.max_episode_steps
        reward = 1.0 if success else 0.0
        info = {
            "success": success,
            "lifted": lifted,
            "cube_pos": self.data.xpos[self.body_cube].copy(),
            "step_count": self.step_count,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def get_state(self) -> np.ndarray:
        """Proprioceptive state: 5 arm joints + normalized gripper."""
        arm = self.data.qpos[ARM_JOINT_IDS].copy().astype(np.float32)
        gripper = np.float32(gripper_angle_to_norm(self.data.qpos[GRIPPER_JOINT_ID]))
        return np.concatenate([arm, [gripper]]).astype(np.float32)

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"observation.state": self.get_state()}
        for name, cam_id in self.camera_ids.items():
            img = self.render(cam_id)
            obs[f"observation.images.{name}"] = img
        return obs

    def render(self, camera_id: int) -> np.ndarray:
        """Render the given camera to a uint8 RGB (H, W, 3) array."""
        assert self._renderer is not None, "No renderer configured"
        self._renderer.update_scene(self.data, camera=camera_id)
        return self._renderer.render().copy()

    def _update_wrist_camera(self) -> None:
        """Re-position the 'wrist' camera on the gripper body (MuJoCo has no
        body-attached cameras for <include>d models)."""
        cam_id = self.camera_ids.get("wrist")
        if cam_id is None:
            return
        rg = self.data.xmat[self.body_gripper].reshape(3, 3)
        local_pos = np.array([0.0, 0.045, -0.05])
        cam_pos = self.data.xpos[self.body_gripper] + rg @ local_pos

        target = self.data.xpos[self.body_cube]
        z = cam_pos - target
        if np.linalg.norm(z) < 1e-6:
            z = -rg[:, 2]
        z = z / np.linalg.norm(z)
        x = np.array([1.0, 0.0, 0.0])
        x = x - (x @ z) * z
        if np.linalg.norm(x) < 1e-4:
            x = rg[:, 0]
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        self.data.cam_xpos[cam_id] = cam_pos
        self.data.cam_xmat[cam_id] = np.stack([x, y, z]).ravel()

    # ------------------------------------------------------------------ #
    # Task state
    # ------------------------------------------------------------------ #
    def ee_pose(self, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation of the gripper site (+ optional local offset)."""
        pos = self.data.site_xpos[self.site_gripper].copy()
        rot = self.data.site_xmat[self.site_gripper].reshape(3, 3)
        if offset is not None:
            pos = pos + rot @ np.asarray(offset, dtype=float)
        return pos, rot

    def cube_contacts(self) -> bool:
        """True when the cube is in contact with either gripper jaw."""
        for c in self.data.contact:
            geom1, geom2 = c.geom1, c.geom2
            if geom1 == self.geom_cube:
                other = geom2
            elif geom2 == self.geom_cube:
                other = geom1
            else:
                continue
            if self.model.geom_bodyid[other] in self._gripper_bodies:
                return True
        return False

    def is_holding_cube(self) -> bool:
        """True when the cube is lifted off the table (proxy for a solid grasp)."""
        z = self.data.xpos[self.body_cube][2]
        return z > self.table_top_z + 2.0 * self.cube_half + 0.005

    def _eval_success(self) -> tuple[bool, bool]:
        cube_pos = self.data.xpos[self.body_cube]
        lifted = cube_pos[2] > self.table_top_z + 2.0 * self.cube_half + 0.01
        if lifted:
            self._was_lifted = True
            self._lift_hold_counter += 1
        else:
            self._lift_hold_counter = 0

        # Pick & place success: cube released near the target on the table
        # after having been lifted.
        at_target = np.hypot(cube_pos[0] - self.cfg.target_xy[0], cube_pos[1] - self.cfg.target_xy[1]) < self.cfg.target_radius
        low = cube_pos[2] < self.table_top_z + 3.0 * self.cube_half
        success = self._was_lifted and at_target and low
        return success, lifted

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
