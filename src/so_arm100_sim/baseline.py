"""Scripted pick-and-place controller used to (a) demo the task in simulation
and (b) generate demonstrations for ACT training.

The controller is a small state machine operating in task space on the
finger-midpoint frame (see ``env.finger_offset``):

    1. APPROACH  : rise well above the cube, then translate to above it
    2. DESCEND   : lower straight down to the grasp pose
    3. CLOSE     : close the gripper and verify cube contact
    4. LIFT      : raise the cube to a safe height
    5. TRANSPORT : move horizontally to above the target zone
    6. PLACE     : lower to the table, release, retreat

Waypoints are interpolated linearly in task space with a bounded speed, and
each step re-solves IK from the current configuration (without mutating the
physics state), which makes the motion smooth and collision-conscious.
"""

from __future__ import annotations

import numpy as np

from .env import SoArm100PickEnv
from .ik import compute_ik


GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0
GRASP_Z_OFFSET = 0.005  # fingers centered on the cube's middle
LIFT_HEIGHT = 0.14
TRAVEL_SPEED = 0.10  # m/s of finger midpoint


class PickPlaceController:
    def __init__(self, env: SoArm100PickEnv, *, verbose: bool = False):
        self.env = env
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def solve_ik(self, target: np.ndarray, **kwargs) -> tuple[np.ndarray, bool]:
        """Solve IK without mutating the physics state."""
        q0 = self.env.data.qpos[:5].copy()
        q, ok = compute_ik(
            self.env.model,
            self.env.data,
            self.env.site_gripper,
            target,
            ee_offset=self.env.finger_offset,
            joint_bias={4: (np.pi / 2, 0.3)},
            max_iter=400,
            tol_pos=1e-3,
            **kwargs,
        )
        self.env.data.qpos[:5] = q0
        return q, ok

    def finger_midpoint(self) -> np.ndarray:
        rot = self.env.data.site_xmat[self.env.site_gripper].reshape(3, 3)
        return self.env.data.site_xpos[self.env.site_gripper] + rot @ self.env.finger_offset

    def move_to(
        self,
        target: np.ndarray,
        *,
        gripper: float = GRIPPER_OPEN,
        max_steps: int = 400,
        tol: float = 0.012,
        speed: float = TRAVEL_SPEED,
    ) -> bool:
        """Move the finger midpoint to ``target`` and hold briefly."""
        target = np.asarray(target, dtype=float)
        for _ in range(max_steps):
            current = self.finger_midpoint()
            delta = target - current
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                # Hold still for a moment so the arm settles.
                self.env.step(np.concatenate([self.env.data.qpos[:5], [gripper]]))
                self.env.step(np.concatenate([self.env.data.qpos[:5], [gripper]]))
                return True
            step = delta * min(1.0, speed / self.env.cfg.fps / max(dist, 1e-6))
            waypoint = current + step
            q, ok = self.solve_ik(waypoint)
            if not ok:
                return False
            self.env.step(np.concatenate([q[:5], [gripper]]))
        return False

    def settle_at(
        self,
        target: np.ndarray,
        *,
        gripper: float = GRIPPER_OPEN,
        iters: int = 8,
        steps_per_iter: int = 12,
        tol: float = 0.004,
    ) -> bool:
        """Feedback-correct the pose: repeatedly re-solve IK from the current
        (physical) configuration and hold, compensating the actuator lag."""
        target = np.asarray(target, dtype=float)
        for _ in range(iters):
            q, ok = self.solve_ik(target)
            if not ok:
                return False
            for _ in range(steps_per_iter):
                self.env.step(np.concatenate([q[:5], [gripper]]))
            err = float(np.linalg.norm(self.finger_midpoint() - target))
            if err < tol:
                return True
        return float(np.linalg.norm(self.finger_midpoint() - target)) < 0.008

    def close_gripper(self, steps: int = 90) -> None:
        for _ in range(steps):
            self.env.step(np.concatenate([self.env.data.qpos[:5], [GRIPPER_CLOSED]]))

    def open_gripper(self, steps: int = 60) -> None:
        for _ in range(steps):
            self.env.step(np.concatenate([self.env.data.qpos[:5], [GRIPPER_OPEN]]))

    # ------------------------------------------------------------------ #
    def run(self, cube_pos: np.ndarray | None = None) -> dict:
        """Execute one pick & place episode. Returns stats."""
        self.env.reset(cube_pos=cube_pos)
        env = self.env
        cube = env.data.xpos[env.body_cube].copy()
        target_xy = np.array(env.cfg.target_xy, dtype=float)
        grasp = cube + np.array([0.0, 0.0, GRASP_Z_OFFSET])

        stats = {
            "success": False,
            "contact": False,
            "dropped": False,
            "cube_displaced": False,
            "steps": 0,
        }

        # 1. Approach from above (rise, then translate, then descend BEHIND the
        #    cube so the finger tips never shove it, then sweep forward so the
        #    cube slides between the open fingers).
        moves_ok = []
        high = cube + np.array([0.0, 0.0, 0.20])
        moves_ok.append(self.move_to(high, max_steps=400))
        behind = cube + np.array([0.0, 0.025, 0.0])
        moves_ok.append(self.move_to(behind + np.array([0.0, 0.0, 0.10]), max_steps=300))
        moves_ok.append(self.move_to(behind + np.array([0.0, 0.0, 0.04]), max_steps=300, speed=0.05))
        moves_ok.append(self.move_to(behind + np.array([0.0, 0.0, 0.005]), max_steps=400, speed=0.03, tol=0.015))
        moves_ok.append(self.move_to(grasp, max_steps=400, speed=0.04, tol=0.006))
        moves_ok.append(self.settle_at(grasp, iters=3, steps_per_iter=8))
        if not all(moves_ok):
            return stats

        # Did we shove the cube during approach?
        cube_now = env.data.xpos[env.body_cube].copy()
        if np.linalg.norm(cube_now[:2] - cube[:2]) > 0.008:
            stats["cube_displaced"] = True
            return stats

        # 2. Close and verify contact.
        self.close_gripper()
        stats["contact"] = env.cube_contacts()
        if not stats["contact"]:
            return stats

        # 3. Lift.
        lift = cube + np.array([0.0, 0.0, LIFT_HEIGHT])
        ok = self.move_to(lift, gripper=GRIPPER_CLOSED, max_steps=400)
        if not ok:
            return stats
        cube_z = env.data.xpos[env.body_cube][2]
        if cube_z < cube[2] + 0.05:
            stats["dropped"] = True
            return stats

        # 4. Transport above the target.
        above_target = np.array([target_xy[0], target_xy[1], lift[2]])
        ok = self.move_to(above_target, gripper=GRIPPER_CLOSED, max_steps=500)
        if not ok:
            return stats

        # 5. Lower onto the target.
        place = np.array([target_xy[0], target_xy[1], cube[2] + GRASP_Z_OFFSET + 0.02])
        ok = self.move_to(place, gripper=GRIPPER_CLOSED, max_steps=300)
        if ok:
            self.open_gripper()
            self.move_to(np.array([target_xy[0], target_xy[1], 0.20]), max_steps=200)

        obs, reward, terminated, truncated, info = env.step(
            np.concatenate([env.data.qpos[:5], [GRIPPER_OPEN]])
        )
        stats["success"] = bool(info["success"])
        stats["steps"] = env.step_count
        return stats

    def run_many(self, n_episodes: int, *, seed: int = 0, max_failures: int = 10) -> list[dict]:
        """Run several episodes with per-episode cube jitter."""
        rng = np.random.default_rng(seed)
        results = []
        failures = 0
        for i in range(n_episodes):
            jx, jy = self.env.cfg.cube_jitter, self.env.cfg.cube_jitter
            jitter = rng.uniform(-jx, jy, size=2)
            xy = np.array(self.env.cfg.cube_default_xy, dtype=float) + jitter
            cube_pos = np.array([xy[0], xy[1], self.env.table_top_z + self.env.cube_half])
            stats = self.run(cube_pos=cube_pos)
            stats["episode"] = i
            results.append(stats)
            if not stats["success"]:
                failures += 1
                if failures >= max_failures:
                    break
        return results
