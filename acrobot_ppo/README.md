# PPO Acrobot - Critic-Separated Architecture

## 分支

| 分支 | 说明 |
|------|------|
| `master` | 共享 backbone，actor/critic 同一网络 |
| `critic-separation` | **当前分支**，actor/critic 分离 backbone |

## 问题背景

Actor 能稳定做到 100% 成功，但 reward 停留在 ~60 分左右，无法进一步提升。

**根因分析：**
- 62 vs 60 的 reward 差距 → advantage 信号仅差 2
- 共享 backbone 的 Critic 对状态价值的估计误差 > 2
- advantage = r + γV(s') - V(s)，V(s) 估计不准则 advantage 噪声大
- Actor 无法接收到精细的改进信号

## 架构对比

### master（共享 backbone）

```
obs → shared_net [256→256→Tanh] → actor_head → logits
                                 → critic_head → value
```

### critic-separation（分离 backbone）

```
obs → actor_net [256→256→256] → actor_head → logits
obs → critic_net [512→512→512→256] → critic_head → value
```

| | master | critic-separation |
|--|--------|------------------|
| Actor hidden | 256 | 256 |
| Critic hidden | 256（共享） | **512（独立）** |
| Critic layers | 2 | **3** |
| Actor lr | 3e-4 | 3e-4 |
| Critic lr | 3e-4（共享） | **1e-3（独立）** |
| Critic updates/batch | 10 | **4** |

## 理论依据

Critic 更大、更独立的理由：
1. **更精细的价值估计**：3层 512 hidden 的网络能捕获状态间更细微的价值差异
2. **更快的学习**：更高的 lr + 更多次更新让 Critic 更快收敛
3. **分离优化**：actor 和 critic 互不干扰，actor 保持策略平滑性

## 快速开始

### 默认配置运行

```bash
python eval_framework.py --seeds 3 --steps 2000000
```

### 对比实验配置

```bash
# 更大 critic
python eval_framework.py --seeds 3 --steps 2000000 --critic-hidden 768 --critic-lr 2e-3

# 更激进 critic 更新
python eval_framework.py --seeds 3 --steps 2000000 --critic-n-epochs 8
```

### 评估已有模型

```bash
python eval_framework.py --load-latest
```

### 继续训练

```bash
python eval_framework.py --continue-train --steps 4000000
```

## 文件结构

```
acrobot_ppo/
├── eval_framework.py     # 主训练脚本（当前分支）
├── CHECKPOINT.md         # Checkpoint 管理文档
├── reward_shaping.py     # Reward shaping 实验
└── train_shaped.py       # Shaped reward 训练
```

## 超参调优建议

如果默认配置没有提升，可以尝试：

1. **降低 actor lr**：保持 actor 稳定性
   ```bash
   --lr 1e-4
   ```

2. **进一步提高 critic lr**：让 critic 更快学习
   ```bash
   --critic-lr 5e-3
   ```

3. **增加 critic hidden**：更大的表示空间
   ```bash
   --critic-hidden 1024
   ```

4. **增加 critic n_epochs**：更多的 critic 更新
   ```bash
   --critic-n-epochs 8
   ```
