# Third-party notices

## SO-ARM100 (SO-101) robot model

The MJCF robot model and mesh assets under `assets/so_arm100/` are from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
distributed under the Apache License 2.0 (see `assets/so_arm100/LICENSE`).

The model is used unmodified apart from the following simulation adaptations:

- The gripper jaw was converted from the stock rack-and-pinion mesh to a
  **linear parallel-jaw** representation (slide joint), matching how LeRobot
  represents the SO-ARM100 gripper (0 = closed, 1 = open).
- The wrist servo housing is visual-only in collision (only the finger boxes
  participate in contacts), which keeps grasp contacts clean and predictable.
- Joint stiffness was raised (`kp=200`) so the position-controlled arm tracks
  IK targets without sag, and the gripper actuator was strengthened
  (`kp=200`, `forcerange=±10`).

## Learned-policy stack

- [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0)
- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0)
- ACT policy as implemented in LeRobot
  ([Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705))
