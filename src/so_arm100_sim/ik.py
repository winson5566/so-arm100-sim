"""Damped-least-squares (DLS) inverse kinematics for the SO-ARM100.

The arm has 5 revolute joints (base yaw, shoulder, elbow, wrist pitch,
wrist roll). We solve for the joint angles that place a virtual end-effector
point (a fixed offset from the "gripper" site) at ``target_pos`` while
aligning the site's z-axis with ``target_z`` (e.g. pointing down for a
top-down grasp).
"""

from __future__ import annotations

import numpy as np
import mujoco


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def compute_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    *,
    target_z: np.ndarray | None = None,
    ee_offset: np.ndarray | None = None,
    joint_ids: list[int] | None = None,
    max_iter: int = 300,
    tol_pos: float = 2e-3,
    tol_orient: float = 0.05,
    damping: float = 0.08,
    weight_orient: float = 1.0,
    joint_bias: dict[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Solve IK in-place (mutates ``data.qpos``) and return (q, success)."""
    if joint_ids is None:
        joint_ids = [0, 1, 2, 3, 4]
    if ee_offset is None:
        ee_offset = np.zeros(3)
    ee_offset = np.asarray(ee_offset, dtype=float)
    joint_bias = joint_bias or {}

    q = data.qpos[joint_ids].copy()
    jnt_ranges = model.jnt_range[joint_ids]
    jnt_limited = model.jnt_limited[joint_ids]

    target_z = None if target_z is None else np.asarray(target_z, dtype=float) / np.linalg.norm(target_z)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        site_pos = data.site_xpos[site_id]
        site_rot = data.site_xmat[site_id].reshape(3, 3)
        ee_pos = site_pos + site_rot @ ee_offset

        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)

        # Jacobian of the virtual EE point.
        jac_ee = jac_pos - _skew(site_rot @ ee_offset) @ jac_rot

        err_pos = target_pos - ee_pos
        err = err_pos.copy()
        jac = jac_ee.copy()

        if target_z is not None:
            # Align the site z-axis with target_z (yaw stays free). The z-error
            # is the cross product r_c x r_d; its Jacobian is
            # M @ J_rot with M = r_c r_d^T - (r_c . r_d) I.
            r_c = site_rot[:, 2]
            err_orient = np.cross(r_c, target_z)
            m_mat = np.outer(r_c, target_z) - np.dot(r_c, target_z) * np.eye(3)
            jac_orient = m_mat @ jac_rot
            err = np.concatenate([err_pos, weight_orient * err_orient])
            jac = np.vstack([jac_ee, weight_orient * jac_orient])

        for jid, (target, weight) in joint_bias.items():
            err = np.concatenate([err, [weight * (target - data.qpos[jid])]])
            row = np.zeros(model.nv)
            row[jid] = 1.0
            jac = np.vstack([jac, weight * row])

        jac_q = jac[:, joint_ids]
        jtj = jac_q @ jac_q.T + damping * np.eye(len(err))
        dq = jac_q.T @ np.linalg.solve(jtj, err)
        q = q + dq

        for i, idx in enumerate(joint_ids):
            if jnt_limited[i]:
                lo, hi = jnt_ranges[i]
                q[i] = np.clip(q[i], lo, hi)

        data.qpos[joint_ids] = q

        # Re-evaluate the error at the updated configuration.
        site_rot = data.site_xmat[site_id].reshape(3, 3)
        ee_pos = data.site_xpos[site_id] + site_rot @ ee_offset
        pos_err = np.linalg.norm(target_pos - ee_pos)
        orient_ok = True
        if target_z is not None:
            site_z = site_rot[:, 2]
            orient_ok = np.linalg.norm(np.cross(site_z, target_z)) < tol_orient
        if pos_err < tol_pos and orient_ok:
            return q.copy(), True

    return q.copy(), False


def solve_ik_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    *,
    target_z: np.ndarray | None = None,
    ee_offset: np.ndarray | None = None,
    pitch_joint: int = 3,
    roll_joint: int = 4,
    n_pitch: int = 45,
    max_pos_err: float = 0.025,
    weight_orient: float = 0.6,
) -> tuple[np.ndarray | None, bool]:
    """IK that sweeps the wrist pitch joint and solves position-only IK with the
    remaining three arm joints, picking the configuration that best satisfies
    both position and gripper-down orientation.

    This is more robust than a single weighted DLS solve for a 5-DOF arm whose
    wrist joints have limited range.
    """
    if ee_offset is None:
        ee_offset = np.zeros(3)
    ee_offset = np.asarray(ee_offset, dtype=float)
    target_z = None if target_z is None else np.asarray(target_z, dtype=float)
    target_z = target_z / np.linalg.norm(target_z)

    lo, hi = model.jnt_range[pitch_joint]
    pitch_grid = np.linspace(lo, hi, n_pitch)
    joint_ids = [0, 1, 2]

    best_q = None
    best_pitch = None
    best_score = np.inf
    best_errs = None
    for pitch in pitch_grid:
        data.qpos[pitch_joint] = pitch
        q, ok = compute_ik(
            model,
            data,
            site_id,
            target_pos,
            target_z=None,
            ee_offset=ee_offset,
            joint_ids=joint_ids,
            max_iter=200,
            damping=0.05,
        )
        pos, rot = (
            data.site_xpos[site_id] + data.site_xmat[site_id].reshape(3, 3) @ ee_offset,
            data.site_xmat[site_id].reshape(3, 3),
        )
        pos_err = float(np.linalg.norm(pos - target_pos))
        # Cosine distance on the site z-axis (properly distinguishes up/down).
        orient_err = (
            float(1.0 - np.dot(rot[:, 2], target_z))
            if target_z is not None
            else 0.0
        )
        if pos_err > max_pos_err:
            continue
        score = pos_err + weight_orient * orient_err
        if score < best_score:
            best_score = score
            best_q = q.copy()
            best_pitch = pitch
            best_errs = (pos_err, orient_err)

    if best_q is None:
        return None, False
    data.qpos[pitch_joint] = best_pitch
    data.qpos[joint_ids] = best_q
    mujoco.mj_forward(model, data)
    return best_q, best_errs[1] < 0.05
