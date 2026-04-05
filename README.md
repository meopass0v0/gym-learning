# gym-learning

强化学习训练框架，基于 gymnasium 环境。

## 环境说明

**Acrobot-v1** — 双连杆摆荡控制

```
[固定端] → [关节1] ——— torque ———→ [关节2] → [末端]
```

- **目标**：施加扭矩（-1/0/+1 N·m），使末端达到目标高度
- **终止条件**：`−cos(θ₁) − cos(θ₂+θ₁) > 1.0` 或 500 步
- **奖励**：每步 -1，到达目标为 0（成功阈值：reward > -100）

物理上是非线性欠驱动系统，需要适时利用动量将连杆甩过头顶。

## 快速开始

```bash
pip install -r acrobot_ppo/requirements.txt
python acrobot_ppo/train.py
```

训练完成后自动生成：
- `metrics_*.csv` — 训练指标
- `learning_curves_*.png` — 学习曲线
- `sample_success.mp4` — 成功案例
- `sample_failure.mp4` — 失败案例

## 轻量评估

```bash
python acrobot_ppo/eval.py              # 数值评估（无窗口，秒级）
python acrobot_ppo/eval.py --video     # 生成 MP4 视频
```

## 算法

| 算法 | 特点 | 适用场景 |
|------|------|---------|
| **PPO** | 稳定、样本效率高、调参友好 | 本项目默认 |

超参数：`lr=3e-4, γ=0.99, λ=0.95, clip=0.2, ent_coef=0.01`

## Metrics 说明

| 指标 | 描述 |
|------|------|
| Episode Reward | 越高越好（接近0=成功） |
| Episode Length | 越低=越快到达目标 |
| Success Rate | 滚动50episode的达成率 |
| Cumulative Reward | 全程累计，用于观察学习趋势 |

## 项目结构

```
gym-learning/
├── README.md
└── acrobot_ppo/
    ├── train.py          # 训练入口
    ├── eval.py           # 评估入口
    ├── requirements.txt  # 依赖
    └── logs/             # TensorBoard 日志
```

## 后续计划

- [ ] 对比 SAC、DQN 等算法
- [ ] CartPole、Pendulum 等经典控制环境扩展
- [ ] 引入 Optuna 超参搜索
- [ ] 分布式训练支持
