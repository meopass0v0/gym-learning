# Acrobot-v1 强化学习项目总结

## 📊 项目结论

对于 **Acrobot-v1** 任务：

| 水平 | Steps 范围 | 说明 |
|------|-----------|------|
| 经典 RL / 控制方法 | ~60-80 | 正常收敛水平 |
| **强 Baseline** | **≈ 60** | 项目当前水平 |
| 极优（接近理论最优） | 50-60 | 前沿研究 |

> 🎯 **当前项目已达到强 Baseline 水平，可以结项。**

---

## 🧠 技术方案

### 最终方案：Double DQN + Replay Buffer

```
网络结构：MLP (256 → 256 → 3)
优化器：Adam (lr=1e-3)
目标网络更新频率：500 steps
Replay Buffer：100000 容量
ε-贪婪：1.0 → 0.01（线性衰减）
训练步数：200000 steps
```

### 关键设计点

1. **Double DQN** — 动作选择用 Main Network，Q值评估用 Target Network，减少 Q 值过估计
2. **Replay Buffer** — 打破时序依赖，稳定训练
3. **Target Network 定期同步** — 每 500 步从 Main 复制参数
4. **ε-贪婪探索** — 前期充分探索，后期稳定利用

---

## 📁 项目结构

```
C:\gym-learning\acrobot_ppo\
├── ddqn.py              # Double DQN 主代码
├── README.md            # 本文档
└── docs/                # 详细技术文档（可选）
```

---

## 🚀 运行方式

### 快速开始

```bash
# 默认配置（3 seeds，200k steps）
python ddqn.py

# 自定义配置
python ddqn.py --seeds 5 --steps 500000 --lr 5e-4
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seeds` | 3 | 独立实验次数 |
| `--steps` | 200000 | 总训练步数 |
| `--hidden` | 256 | 隐藏层大小 |
| `--lr` | 1e-3 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--batch-size` | 64 | 批次大小 |
| `--replay-capacity` | 100000 | Replay Buffer 容量 |
| `--target-update-freq` | 500 | Target Network 更新频率 |
| `--epsilon-decay` | 10000 | ε 衰减步数 |
| `--eval-freq` | 5000 | 评估频率 |

---

## 📈 评估指标

训练过程中记录：
- **Success Rate (SR)** — 到达目标的 episode 比例
- **Episode Reward** — 每个 episode 的累计 reward
- **Training Loss** — TD 损失变化
- **Epsilon** — 探索率衰减曲线

最终评估使用 50 个 episode 的平均值。

---

## 🔬 消融实验记录

### 已尝试的方案

| 方案 | 结果 | 备注 |
|------|------|------|
| PPO + Critic-Separation | 收敛慢 | 2M steps 才稳定 |
| PPO + Potential Shaping | 收敛改善 | 但实现复杂 |
| **Double DQN + Replay Buffer** | **60 steps** | **简单有效** |

### 教训

1. **Acrobot 任务不需要复杂策略** — 简单的 DQN 变体就能做得很好
2. **Policy Gradient 方法收敛慢** — 对于这种稀疏奖励、动作空间小的任务，Value-Based 方法更合适
3. **Potential Shaping 有用但非必须** — 核心还是要有足够的探索
4. **ε 衰减要够快** — Acrobot 任务需要快速从探索转向利用

---

## 🎯 理论最优分析

Acrobot 的最优轨迹约 **50-60 steps**，原因：
- 初始状态：两根杆自然下垂（θ1=π, θ2=0）
- 目标：将杆甩到垂直向上（θ1+θ2=π）
- 最优控制：利用重力加速，在合适的时机反转 torque
- 物理限制：每个 torque 步只能给一个关节加力

---

## 📝 技术笔记

### Double DQN vs Naive DQN

Naive DQN 的过估计问题：
```
Q(s,a) = r + γ * max_a' Q_target(s', a')
```
如果 Q_target 对所有动作都过估计，会导致次优策略被高估。

Double DQN 缓解：
```
a* = argmax_a' Q_main(s', a')
Q(s,a) = r + γ * Q_target(s', a*)
```

### ε-贪婪策略

线性衰减公式：
```python
epsilon = max(epsilon_end, epsilon_start - step / epsilon_decay)
```

对于 Acrobot：
- 前期（ε=1.0）：随机动作，充分探索状态空间
- 后期（ε=0.01）：几乎贪婪，利用已学策略

---

## ✅ 结项确认

- [x] Double DQN 实现完成
- [x] 达到 60 steps 强 Baseline 水平
- [x] 多 seed 验证结果稳定
- [x] 项目文档整理完成
- [x] GitHub 仓库同步

---

_项目完成日期：2026-04-07_
