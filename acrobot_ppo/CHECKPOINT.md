# PPO Checkpoint 管理

## 概述

支持训练中断恢复、单 checkpoint 续训练、仅评估三种模式。

## 核心设计

### 1. Checkpoint 结构

每次保存生成两个文件：

| 文件 | 内容 |
|------|------|
| `ppo_final_{ts}.pt` | model_state + optimizer_state + current_update |
| `config_{ts}.json` | 完整超参配置 |

### 2. 保存时机

- 每次完整训练结束后自动保存
- 支持手动中断后继续训练

## 使用方法

### 训练

```bash
python eval_framework.py --seeds 3 --steps 2000000
```

### 继续训练

```bash
# 从最新 checkpoint 继续训练
python eval_framework.py --continue-train

# 指定 steps 继续（总步数会叠加）
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

| 参数 | 说明 |
|------|------|
| `--seeds` | 训练 seed 数量 |
| `--steps` | 总训练步数 |
| `--eval-every` | 每隔多少 update 评估一次 |
| `--load-latest` | 加载最新 checkpoint，仅评估 |
| `--continue-train` | 加载最新 checkpoint，继续训练 |
| `--checkpoint` | 指定 checkpoint 路径 |

## 实现细节

### 保存逻辑

```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "current_update": current_update,
}, model_path)
```

### 加载逻辑

```python
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt["model_state"])
optimizer.load_state_dict(ckpt["optimizer_state"])
current_update = ckpt["current_update"]
```

### 续训练逻辑

```python
train_and_evaluate(
    env_id,
    seed=42,
    config=config,
    start_update=current_update + 1,
    loaded_model=model,
    loaded_optimizer=optimizer,
)
```

## 注意事项

1. `continue-train` 默认使用 seed=42，因为 checkpoint 是特定 seed 的
2. `--steps` 在续训练时表示新的总步数，不是增量
3. 中断训练建议使用 Ctrl+C，避免损坏 checkpoint
