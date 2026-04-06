# PPO Checkpoint 管理

## 分支说明

- **master** — 共享 backbone（actor/critic 同一网络）
- **critic-separation** — 分离 backbone（actor/critic 独立网络）

## 概述

支持训练中断恢复、单 checkpoint 续训练、仅评估三种模式。

## 核心设计

### Checkpoint 结构

每次保存生成两个文件：

| 文件 | 内容 |
|------|------|
| `ppo_final_{ts}.pt` | model_state + actor_opt_state + critic_opt_state + current_update |
| `config_{ts}.json` | 完整超参配置 |

## 使用方法

### 训练（critic-separation 分支）

```bash
# 默认配置（actor=256, critic=512, critic_lr=1e-3, critic_n_epochs=4）
python eval_framework.py --seeds 3 --steps 2000000

# 自定义 critic 配置
python eval_framework.py --seeds 3 --steps 2000000 \
    --critic-hidden 512 \
    --critic-lr 1e-3 \
    --critic-n-epochs 4
```

### 继续训练

```bash
python eval_framework.py --continue-train --steps 4000000
```

### 仅评估

```bash
python eval_framework.py --load-latest
```

### 指定 checkpoint

```bash
python eval_framework.py --checkpoint "C:\gym-learning\acrobot_ppo\ppo_final_20260406_123045.pt"
```

## 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seeds` | 1 | 训练 seed 数量 |
| `--steps` | 2000000 | 总训练步数 |
| `--eval-every` | 1 | 每隔多少 update 评估一次 |
| `--load-latest` | False | 加载最新 checkpoint，仅评估 |
| `--continue-train` | False | 加载最新 checkpoint，继续训练 |
| `--checkpoint` | None | 指定 checkpoint 路径 |

### Critic 分离架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden` | 256 | Actor hidden size |
| `--critic-hidden` | 512 | Critic hidden size（更大） |
| `--critic-lr` | 1e-3 | Critic 学习率（默认比 actor 高） |
| `--critic-n-epochs` | 4 | Critic 每 batch 更新次数（actor 是 10） |

## 实现细节

### 保存逻辑

```python
torch.save({
    "model_state": model.state_dict(),
    "actor_opt_state": actor_opt.state_dict(),
    "critic_opt_state": critic_opt.state_dict(),
    "current_update": current_update,
}, model_path)
```

### 加载逻辑

```python
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt["model_state"])
actor_opt.load_state_dict(ckpt["actor_opt_state"])
critic_opt.load_state_dict(ckpt["critic_opt_state"])
current_update = ckpt["current_update"]
```

### 续训练逻辑

```python
h, model, actor_opt, critic_opt = train_and_evaluate(
    env_id,
    seed=42,
    config=config,
    start_update=current_update + 1,
    loaded_model=model,
    loaded_actor_opt=actor_opt,
    loaded_critic_opt=critic_opt,
)
```

## 注意事项

1. `continue-train` 默认使用 seed=42，因为 checkpoint 是特定 seed 的
2. `--steps` 在续训练时表示新的总步数，不是增量
3. 中断训练建议使用 Ctrl+C
4. **critic-separation 分支的 checkpoint 格式与 master 不兼容**（分离的优化器）
