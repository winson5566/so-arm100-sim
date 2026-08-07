# ALOHA Transfer Cube — official ACT replication

复现 LeRobot 官方模型卡 `lerobot/act_aloha_sim_transfer_cube_human` 的结果：

| 项目 | 官方值 |
| --- | --- |
| 数据集 | `lerobot/aloha_sim_transfer_cube_human`（50 演示 / 20,000 帧 / 50fps） |
| 模型 | ACT，LeRobot 默认配置（dim 512 / resnet18），52M 参数 |
| 训练 | 80k 步，batch 8 |
| 官方训练耗时 | A100 约 1h45 |
| 官方评估 | 500 回合，83% 成功率（LeRobot 实现） |

## 环境

独立 venv（gym-aloha 要求 mujoco<3.9，与 so-arm100-sim 主环境的 3.11 冲突）：

```bash
uv venv --python 3.12 ../.venv-aloha
uv pip install --python ../.venv-aloha/bin/python \
  "lerobot[dataset,training,timm-dep]==0.6.1" gym-aloha imageio imageio-ffmpeg
```

## 训练

```bash
./train.sh              # 默认 mps（本机约 15 小时）；GPU 机器改 device=cuda
```

等价命令：

```bash
../.venv-aloha/bin/lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.type=act --policy.push_to_hub=false --policy.device=mps \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=80000 --batch_size=8 --save_freq=10000 \
  --output_dir=outputs/act_aloha_transfer_80k
```

> 为忠实复现官方 83%，不缩放图像（640×480 直接进 ResNet），也不用 AMP。

## 评估（官方口径：500 回合）

```bash
./eval.sh               # 读取 outputs/act_aloha_transfer_80k/checkpoints/last/pretrained_model
```

```bash
../.venv-aloha/bin/lerobot-eval \
  --policy.type=act --policy.pretrained_path=outputs/act_aloha_transfer_80k/checkpoints/last/pretrained_model \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=500 --eval.batch_size=50 --policy.device=mps
```
