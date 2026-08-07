# so-arm100-sim

在 **MuJoCo** 仿真环境中完成 SO-ARM100(SO-101)机械臂的**单臂抓取任务**:拿起桌面上的红色方块,放到绿色目标区。数据采集使用 **LeRobot 标准数据集格式**,策略训练使用 **ACT(Action Chunking Transformer)**,评估在仿真环境闭环完成。

本项目已在 macOS(Apple Silicon)上完整跑通:**脚本基线控制器 100% 成功**,数据采集、ACT 训练、策略评估管线端到端可用。

## 架构

```text
assets/so_arm100/              官方 SO-101 MJCF + 网格(带 Apache-2.0 许可)
  └─ so_arm100_pick.xml        任务场景:机器人 + 方块 + 目标区 + 相机
src/so_arm100_sim/
  ├─ env.py                     MuJoCo 环境(状态/图像观测、动作、成功判定)
  ├─ ik.py                      阻尼最小二乘 IK(+腕部翻滚偏置)
  ├─ baseline.py                脚本化抓取控制器(演示数据来源)
  ├─ collect.py                 采集演示 → LeRobot 数据集
  ├─ eval_act.py                加载 ACT checkpoint 在仿真中评估 + 录制视频
  └─ scripts/                   CLI 入口
configs/train_act.yaml          ACT 训练配置(lerobot-train)
data/                           采集的数据集(gitignore)
outputs/                        训练 checkpoint 与评估视频(gitignore)
```

## 安装

需要 macOS + Python 3.12(推荐用 `uv` 管理):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

> 已测试版本:`mujoco==3.11.0`、`lerobot==0.6.1`、`torch==2.11.0`(见 `requirements.txt`)。

## 快速开始

### 1. 运行脚本化基线(演示抓取任务)

```bash
make baseline          # 或 PYTHONPATH=src .venv/bin/python -m so_arm100_sim.scripts.run_baseline
```

输出 `outputs/baseline.mp4`(如启用视频),控制台打印 `success=True`。

### 2. 采集演示数据

```bash
make collect           # 默认 20 个成功回合 → data/local/so_arm100_pick
make collect50         # 50 个成功回合、方块 ±1cm 抖动(ACT 论文配方)
```

每个回合约 900 帧(30 Hz),观测包含 `observation.state`(5 关节角 + 夹爪)与 `observation.images.top`(640×480 RGB,MP4 存储),动作是**绝对关节目标**。macOS 上默认用 `h264_videotoolbox`(Apple 硬件编码,速度快);其他平台可用 `--video-codec h264`(libx264)或 `libsvtav1`。

### 3. 训练 ACT

```bash
make train             # .venv/bin/lerobot-train --config_path=configs/train_act.yaml
```

小规模冒烟测试(验证管线):

```bash
.venv/bin/lerobot-train --config_path=configs/train_act.yaml --steps=50 --policy.device=mps --output_dir=outputs/train_act_smoke
```

### 4. 评估策略

```bash
make eval              # 默认读取 outputs/train_act/checkpoints/last/pretrained_model
```

评估结果写入 `outputs/eval_*/rollout.mp4`,控制台报告成功率。

## 任务与环境

- **动作空间(6D)**:5 个臂关节的绝对角度(rad)+ 夹爪目标 `[0,1]`(0=闭合,1=张开),与 LeRobot 的 SO-100/101 约定一致。
- **观测**:`observation.state`(6D)+ 相机图像 `observation.images.top`(可加 `wrist`)。
- **成功判定**:方块先被提起,再被释放到目标区(半径 5 cm)。
- **夹爪**:官方 MJCF 的关节夹爪行程不足以可靠抓取 3 cm 方块,且 LeRobot 官方文档建议将 SO-ARM100 夹爪表示为线性关节(0=闭合、100=张开),因此本项目将夹爪改为**线性平行爪**(滑轨 ±4.5 cm),保留官方机械臂其余部分。

## 最佳实践

### 数据采集

- 单任务建议 **50–200 个演示**;回合数越多,ACT 泛化越好。本项目默认 20 个用于快速验证。
- 在真实/仿真采集时,演示应覆盖目标位置的小范围抖动(`--cube-jitter 0.01` 等),否则策略只会记忆固定轨迹。
- 相机越多越好(推荐 `--cameras top,wrist`),代价是训练更慢、数据更大。
- 视频编码默认 `h264_videotoolbox`(macOS 硬件编码);磁盘敏感可改 `--video-codec libsvtav1`(小)。
- 每个回合写成独立小 MP4 文件、关键帧间隔 1 s(`g=30`),训练时随机取帧比单个大文件快得多。

### ACT 训练

- **ACT 论文配方(Transfer Cube,约 90% 成功率)**:50 个演示、2000 epochs、batch 8。对应到本数据集为 `steps = 2000 × ceil(总帧数/8)`。
- **图像缩放是必须的**:LeRobot 的 ACT 会把数据集图像原分辨率喂给 ResNet。本项目数据集是 640×480,若不做缩放,ResNet18 的计算量约为 224×224 输入的 6 倍。配置里已用 `image_transforms` 统一缩到 **224×224**(ACT 论文的标准做法),实测 MPS 上从 **1.9 步/秒提升到约 6.7 步/秒**。评估脚本默认做同样的缩放,训练/评估保持一致。
- 训练阶段用 `dataset.eval_split` 留出部分回合评估泛化;本项目演示配置为 0。
- 评估时在仿真闭环中统计成功率,而不是只看训练损失。

### 仿真细节

- macOS 上 MuJoCo 离屏渲染使用 `glfw` 后端(`MUJOCO_GL` 不支持 `osmesa`/`egl`)。
- 官方 MJCF 的位置执行器 `kp=17.8` 会因重力下垂导致 ~3 cm 末端误差,本项目提高到 `kp=200`(见 `THIRD_PARTY_NOTICES.md`)。

## 已验证结果

| 项目 | 结果 |
| --- | --- |
| 脚本基线(固定位置,5 回合) | 5/5 成功 |
| 脚本基线(±1 cm 抖动,5 回合) | 5/5 成功 |
| 数据采集 → LeRobot 数据集 | 格式正确,可被 `lerobot-train` 直接读取 |
| ACT 训练 1500 步(MPS) | loss 15.5→1.5,策略学会“接近”阶段,未收敛(仅 0.65 epoch) |
| 评估闭环 + 视频 | 正常输出 `rollout.mp4` |

## 复刻 ACT 论文配方(50 演示 / 2000 epochs)

1. 采集 50 个演示(含 ±1cm 方块抖动):

   ```bash
   make collect50
   ```

2. 训练配置 `configs/train_act.yaml` 已按配方设置:`batch_size=8`、
   2000 epochs 对应的总步数(11,884,000)、`resnet18` 骨干、`dim_model=256`、
   图像 224×224。
   运行:

   ```bash
   make train
   ```

3. 闭环评估(默认读取 `outputs/train_act_50ep` 最新 checkpoint):

   ```bash
   make eval
   ```

> **硬件说明**:本机为 Apple Silicon(MPS),图像 224×224 时实测约 6.7 步/秒。
> 2000 epochs(1188 万步)在本机约需 **3 周**,不建议完整跑;要复现论文的
> ~90% 成功率,建议把本仓库拷贝到带 NVIDIA GPU 的机器/云主机上,执行同样的
> `make collect50 && make train && make eval`(配置里 `policy.device` 会自动选择
> `cuda`,无需改代码)。
>
> 若想先在本机验证策略能不能学会抓取,可先跑 100–200 epochs(约 1–2 天):
>
> ```bash
> .venv/bin/lerobot-train --config_path=configs/train_act.yaml --steps=594200
> # 150 epochs: --steps=891300
> ```
>
> 社区经验(SO-100 简单 pick-place、50 演示)通常 100–150 epochs 即可获得
> 较高的闭环成功率,2000 epochs 是为了逼近论文的 ~90% 上界。

## 常见问题

- **`objc: Class AVFFrameReceiver ...` 警告**:PyAV 与 Homebrew ffmpeg 的重复符号警告,无害。
- **训练时报 `'repo_id' argument missing`**:训练配置里 `policy.push_to_hub` 需为 `false`(本地训练)。
- **渲染失败/黑屏**:确认 `MUJOCO_GL=glfw`(macOS 默认),或减少相机数量。
- **采集慢**:视频编码是瓶颈,macOS 默认已用 `h264_videotoolbox` 硬件编码;也可减小图像尺寸。

## 参考

- [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)(Apache-2.0)
- [LeRobot](https://github.com/huggingface/lerobot)
- ACT:[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
