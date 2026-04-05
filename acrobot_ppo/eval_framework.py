"""
PPO Evaluation Framework - 严谨研究版

Metrics 设计原则:
1. 多 seed 统计（mean ± std），单次 run 不具有统计意义
2. 收敛曲线：每个 eval step 记录完整指标
3. Baseline 对比：random policy 作为参照
4. 训练稳定性：方差分析
5. 样本效率：到达阈值所需步数
6. 超参 + 环境配置完整记录
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import json
import argparse
import imageio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# 1. Actor-Critic Network
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        f = self.net(x)
        return self.actor(f), self.critic(f)

    def get_action(self, obs, deterministic=False):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val.squeeze(-1)

    def get_action_and_logprob(self, obs, action):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), val.squeeze(-1)


# ─────────────────────────────────────────────
# 2. Rollout (已修复 reset bug)
# ─────────────────────────────────────────────
def collect_rollout(env, model, n_steps):
    obs_buf, act_buf, rew_buf = [], [], []
    done_buf, logp_buf, val_buf, ent_buf = [], [], [], []

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action(obs)
        obs_np, reward, done, truncated, _ = env.step(action.item())

        if done or truncated:
            obs_new, _ = env.reset()
            obs_new = torch.tensor(obs_new, dtype=torch.float32, device=DEVICE)
        else:
            obs_new = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)

        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(done or truncated)
        logp_buf.append(log_prob)
        val_buf.append(value)
        ent_buf.append(entropy)
        obs = obs_new

    return (torch.stack(obs_buf), torch.stack(act_buf),
            torch.tensor(rew_buf, device=DEVICE),
            torch.tensor(done_buf, device=DEVICE),
            torch.stack(logp_buf), torch.stack(val_buf), torch.stack(ent_buf))


# ─────────────────────────────────────────────
# 3. GAE (修正版)
# ─────────────────────────────────────────────
def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
    n = len(rew_buf)
    advantages = torch.zeros(n, device=DEVICE)
    next_value = val_buf[-1].item()
    next_gae = 0.0

    for t in reversed(range(n)):
        mask = 1.0 - done_buf[t].float()
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        next_gae = delta + gamma * lam * mask * next_gae
        advantages[t] = next_gae
        next_value = val_buf[t].item()

    # 标准化只给 actor 用，不污染 critic target
    raw_advantages = advantages.clone()
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    returns = raw_advantages + val_buf  # 用未标准化的原始 advantage 给 critic
    return returns, advantages


# ─────────────────────────────────────────────
# 4. PPO Update
# ─────────────────────────────────────────────
def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, returns_b, advantages_b,
               clip_range, value_coef, ent_coef, batch_size, n_epochs):
    n = obs_b.shape[0]
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent, _ = model.get_action_and_logprob(obs_b[mb], act_b[mb])
            ratio = torch.exp(logp_new - old_logp_b[mb])
            surr1 = ratio * advantages_b[mb]
            ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            surr2 = ratio_clipped * advantages_b[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(
                model.critic(model.net(obs_b[mb])).squeeze(-1), returns_b[mb])
            entropy_loss = -ent.mean()
            loss = policy_loss + value_coef * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


# ─────────────────────────────────────────────
# 5. Evaluation (诚实评估)
# ─────────────────────────────────────────────
def evaluate_honest(env, model, n_episodes=50):
    """
    诚实评估：区分 terminated（到达目标）和 truncated（超时）
    返回完整 episode 数据，不只是平均值
    """
    model.eval()
    rewards, lengths, success_flags = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        total_r, steps = 0, 0
        done = truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward
            steps += 1

        rewards.append(total_r)
        lengths.append(steps)
        # terminated + not truncated = 真正到达目标
        success_flags.append(done and not truncated)

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    success_flags = np.array(success_flags)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "success": success_flags,
        "sr": success_flags.mean(),
        "sr_se": success_flags.std() / np.sqrt(len(success_flags)),
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(),
        "reward_se": rewards.std() / np.sqrt(len(rewards)),
        "length_mean": lengths.mean(),
        "length_std": lengths.std(),
        "length_se": lengths.std() / np.sqrt(len(lengths)),
    }


def evaluate_random(env, n_episodes=200):
    """Random policy baseline，用于对比"""
    rewards, lengths, success_flags = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, steps = 0, 0
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
        rewards.append(total_r)
        lengths.append(steps)
        success_flags.append(done and not truncated)
    rewards, lengths, success_flags = np.array(rewards), np.array(lengths), np.array(success_flags)
    return {
        "rewards": rewards, "lengths": lengths, "success": success_flags,
        "sr": success_flags.mean(), "reward_mean": rewards.mean(), "length_mean": lengths.mean(),
    }



# ─────────────────────────────────────────────
# Reward Shaping Wrapper
# ─────────────────────────────────────────────
class RewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping with two terms:
      height_shaping  = k_h * (cos(theta1) + cos(theta1 + theta2))
        - 越接近"摆起来" reward 越大，∈ [-2k_h, +2k_h]
      velocity_shaping = k_v * omega1
        - 鼓励有效摆动，omega1 ∈ ~[-12, +12] rad/s
    """

    def __init__(self, env, k_h=0.0, k_v=0.0):
        super().__init__(env)
        self.k_h = k_h
        self.k_v = k_v

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        # obs = [cos(theta1), sin(theta1), cos(theta2), sin(theta2), omega1, omega2]
        cos_t1, sin_t1 = obs[0], obs[1]
        cos_t2, sin_t2 = obs[2], obs[3]
        omega1 = obs[4]

        theta1 = np.arctan2(sin_t1, cos_t1)
        theta2 = np.arctan2(sin_t2, cos_t2)

        # height shaping: k_h * (cos(theta1) + cos(theta1 + theta2))
        height_shaping = self.k_h * (cos_t1 + np.cos(theta1 + theta2))

        # velocity shaping: k_v * omega1
        velocity_shaping = self.k_v * omega1

        # 到达目标保持 0 reward
        if done and not truncated:
            shaped_reward = 0.0
        else:
            shaped_reward = base_reward + height_shaping + velocity_shaping

        return obs, shaped_reward, done, truncated, info



# ─────────────────────────────────────────────
# 6. 完整训练 + 记录
# ─────────────────────────────────────────────
def train_and_evaluate(env_id, seed, config, eval_every=5):
    """
    单次训练 run，返回完整指标历史
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id)
    env.reset(seed=seed)
    k_h = config.get("k_h", 0.0)
    k_v = config.get("k_v", 0.0)
    if k_h > 0 or k_v > 0:
        env = RewardShapingWrapper(env, k_h=k_h, k_v=k_v)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden=config.get("hidden", 256)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    n_steps = config["n_steps"]
    total_updates = config["total_steps"] // n_steps

    # 历史记录
    history = {
        "update": [], "timestep": [],
        "sr": [], "sr_se": [],
        "reward_mean": [], "reward_se": [],
        "length_mean": [], "length_std": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
    }

    for it in range(1, total_updates + 1):
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, config["gamma"], config["lam"])

        # 记录训练 loss
        with torch.no_grad():
            logp_new, ent_new, _ = model.get_action_and_logprob(obs_b, act_b)
            ratio = torch.exp(logp_new - logp_b)
            surr1 = ratio * advantages_b
            ratio_clipped = torch.clamp(ratio, 1 - config["clip_range"], 1 + config["clip_range"])
            surr2 = ratio_clipped * advantages_b
            pol_loss = (-torch.min(surr1, surr2)).mean().item()
            val_loss = nn.functional.mse_loss(
                model.critic(model.net(obs_b)).squeeze(-1), returns_b).item()
            ent_loss = ent_new.mean().item()

        ppo_update(model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
                    config["clip_range"], config["value_coef"], config["ent_coef"],
                    config["batch_size"], config["n_epochs"])

        # 评估
        if it % eval_every == 0 or it == total_updates:
            result = evaluate_honest(env, model, n_episodes=50)
            ts = it * n_steps
            history["update"].append(it)
            history["timestep"].append(ts)
            history["sr"].append(result["sr"])
            history["sr_se"].append(result["sr_se"])
            history["reward_mean"].append(result["reward_mean"])
            history["reward_se"].append(result["reward_se"])
            history["length_mean"].append(result["length_mean"])
            history["length_std"].append(result["length_std"])
            history["policy_loss"].append(pol_loss)
            history["value_loss"].append(val_loss)
            history["entropy"].append(ent_loss)

            print(f"  [seed={seed}] U{it:3d} ts={ts:6d} | "
                  f"SR={result['sr']*100:5.1f}%±{result['sr_se']*100:4.1f}% | "
                  f"R={result['reward_mean']:7.1f}±{result['reward_se']:5.1f} | "
                  f"L={result['length_mean']:5.1f}±{result['length_std']:4.1f}")

    return history, model


# ─────────────────────────────────────────────
# 7. 多 Seed 聚合统计
# ─────────────────────────────────────────────
def aggregate_runs(all_histories, random_baseline):
    """
    聚合多个 seed 的结果，计算 mean ± std
    """
    df = pd.DataFrame(all_histories[0])

    # 按 timestep 对齐，不同样本数可能不同
    # 简单做法：取最后一个 eval 点作为最终结果
    final = {m: [] for m in ["sr", "reward_mean", "length_mean"]}
    for h in all_histories:
        final["sr"].append(h["sr"][-1])
        final["reward_mean"].append(h["reward_mean"][-1])
        final["length_mean"].append(h["length_mean"][-1])

    return {
        "sr_mean": np.mean(final["sr"]),
        "sr_std": np.std(final["sr"]),
        "sr_se": np.std(final["sr"]) / np.sqrt(len(final["sr"])),
        "reward_mean_mean": np.mean(final["reward_mean"]),
        "reward_mean_std": np.std(final["reward_mean"]),
        "length_mean_mean": np.mean(final["length_mean"]),
        "length_mean_std": np.std(final["length_mean"]),
        "random_sr": random_baseline["sr"],
        "random_reward": random_baseline["reward_mean"],
        "random_length": random_baseline["length_mean"],
        "n_seeds": len(all_histories),
        "all_histories": all_histories,
    }




# ─────────────────────────────────────────────
# 9. Episode Video Recording
# ─────────────────────────────────────────────
def record_episode_videos(model, env, save_dir, n_success=1, n_failure=1):
    """
    Record episodes as MP4:
    - n_success: first n_success successful episodes
    - n_failure: first n_failure failed episodes
    Records both success and failure cases for qualitative analysis.
    """
    model.eval()
    success_frames, failure_frames = [], []
    success_meta, failure_meta = [], []
    max_frames = 1500

    print("\n[Video] Recording episodes...")
    ep_count = 0

    for ep in range(n_success * 10 + n_failure * 10):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        ep_frames = []
        total_r, steps = 0, 0
        done = truncated = False

        while not (done or truncated):
            frame = env.render()  # rgb_array
            ep_frames.append(frame)
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward
            steps += 1

        is_success = done and not truncated
        meta = {"ep": ep + 1, "reward": total_r, "length": steps, "success": is_success}

        if is_success and len(success_frames) < n_success:
            trimmed = ep_frames[:max_frames:2]
            success_frames.append(trimmed)
            success_meta.append(meta)
            print(f"  [SUCCESS] ep={ep+1} reward={total_r:.0f} length={steps}")
        elif not is_success and len(failure_frames) < n_failure:
            trimmed = ep_frames[:max_frames:2]
            failure_frames.append(trimmed)
            failure_meta.append(meta)
            print(f"  [FAILURE] ep={ep+1} reward={total_r:.0f} length={steps}")

        if len(success_frames) >= n_success and len(failure_frames) >= n_failure:
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (frames, meta) in enumerate(zip(success_frames, success_meta)):
        path = os.path.join(save_dir, f"video_success_{i+1}_{ts}.mp4")
        print(f"  Encoding {len(frames)} frames -> {path}")
        imageio.mimwrite(path, frames, fps=30, codec="libx264", quality=8)

    for i, (frames, meta) in enumerate(zip(failure_frames, failure_meta)):
        path = os.path.join(save_dir, f"video_failure_{i+1}_{ts}.mp4")
        print(f"  Encoding {len(frames)} frames -> {path}")
        imageio.mimwrite(path, frames, fps=30, codec="libx264", quality=8)

    return success_meta, failure_meta


# ─────────────────────────────────────────────
# 8. 可视化
# ─────────────────────────────────────────────
def plot_results(agg, config, save_dir):
    histories = agg["all_histories"]

    # 转换为 DataFrame 方便处理
    df = pd.DataFrame({
        "timestep": histories[0]["timestep"],
        "sr_mean": np.mean([h["sr"] for h in histories], axis=0),
        "sr_std": np.std([h["sr"] for h in histories], axis=0),
        "reward_mean": np.mean([h["reward_mean"] for h in histories], axis=0),
        "reward_std": np.std([h["reward_mean"] for h in histories], axis=0),
        "length_mean": np.mean([h["length_mean"] for h in histories], axis=0),
        "length_std": np.std([h["length_mean"] for h in histories], axis=0),
        "pol_loss": np.mean([h["policy_loss"] for h in histories], axis=0),
        "val_loss": np.mean([h["value_loss"] for h in histories], axis=0),
        "entropy": np.mean([h["entropy"] for h in histories], axis=0),
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"PPO on {config['env_id']} | {agg['n_seeds']} seeds | "
                 f"lr={config['lr']} n_steps={config['n_steps']}", fontsize=13, fontweight="bold")

    ts = df["timestep"]

    # 1. Success Rate with CI band
    ax = axes[0, 0]
    sr = df["sr_mean"].values
    sr_err = df["sr_std"].values / np.sqrt(agg["n_seeds"]) * 1.96  # 95% CI
    ax.plot(ts, sr, color=C.green, lw=2)
    ax.fill_between(ts, np.clip(sr - sr_err, 0, 1), np.clip(sr + sr_err, 0, 1),
                   color=C.green, alpha=0.15)
    ax.axhline(y=agg["random_sr"], color=C.red, linestyle="--", label=f"Random {agg['random_sr']*100:.1f}%")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Episode Reward with CI band
    ax = axes[0, 1]
    rm = df["reward_mean"].values
    rm_err = df["reward_std"].values / np.sqrt(agg["n_seeds"]) * 1.96
    ax.plot(ts, rm, color=C.blue, lw=2)
    ax.fill_between(ts, rm - rm_err, rm + rm_err, color=C.blue, alpha=0.15)
    ax.axhline(y=agg["random_reward"], color=C.red, linestyle="--", label=f"Random {agg['random_reward']:.1f}")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Episode Reward (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)

    # 3. Episode Length with CI band
    ax = axes[0, 2]
    lm = df["length_mean"].values
    lm_err = df["length_std"].values / np.sqrt(agg["n_seeds"]) * 1.96
    ax.plot(ts, lm, color=C.orange, lw=2)
    ax.fill_between(ts, np.clip(lm - lm_err, 0, None), lm + lm_err,
                   color=C.orange, alpha=0.15)
    ax.axhline(y=agg["random_length"], color=C.red, linestyle="--", label=f"Random {agg['random_length']:.1f}")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Avg Episode Length")
    ax.set_title("Episode Length (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)

    # 4. Individual seed runs (SR)
    ax = axes[1, 0]
    for i, h in enumerate(histories):
        ax.plot(h["timestep"], h["sr"], alpha=0.4, lw=1)
    ax.plot(ts, df["sr_mean"], color=C.green, lw=2, label="Mean")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Success Rate")
    ax.set_title(f"Individual Seeds (n={agg['n_seeds']})"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # 5. Policy & Value Loss
    ax = axes[1, 1]
    ax.plot(ts, df["pol_loss"], color=C.orange, lw=2, label="Policy Loss")
    ax.plot(ts, df["val_loss"], color=C.blue, lw=2, label="Value Loss")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Loss")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(alpha=0.3)

    # 6. Entropy
    ax = axes[1, 2]
    ax.plot(ts, df["entropy"], color=C.purple, lw=2)
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy (exploration)"); ax.grid(alpha=0.3)

    plt.tight_layout()

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"eval_results_{ts_str}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {path}")
    return path, df


def save_csv(df, agg, config, save_dir):
    """保存 CSV 用于后续分析"""
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"eval_data_{ts_str}.csv")
    df.to_csv(csv_path, index=False)

    # 保存汇总统计
    summary_path = os.path.join(save_dir, f"eval_summary_{ts_str}.json")
    summary = {
        "config": config,
        "random_baseline": {
            "sr": float(agg["random_sr"]),
            "reward_mean": float(agg["random_reward"]),
            "length_mean": float(agg["random_length"]),
        },
        "final": {
            "sr_mean": float(agg["sr_mean"]),
            "sr_std": float(agg["sr_std"]),
            "sr_se": float(agg["sr_se"]),
            "reward_mean": float(agg["reward_mean_mean"]),
            "reward_std": float(agg["reward_mean_std"]),
            "length_mean": float(agg["length_mean_mean"]),
            "length_std": float(agg["length_mean_std"]),
        "length_se": float(agg["length_mean_std"]) / np.sqrt(agg["n_seeds"]),
        },
        "n_seeds": agg["n_seeds"],
        "timestamp": ts_str,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {summary_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
C = type('C', (), {
    'green': '#3fb950', 'blue': '#58a6ff', 'orange': '#f0883e',
    'purple': '#d2a8ff', 'red': '#f85149', 'yellow': '#e3b341'
})()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    config = {
        "env_id": "Acrobot-v1",
        "total_steps": args.steps,
        "n_steps": 8192,
        "batch_size": 256,
        "n_epochs": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "ent_coef": 0.01,
        "hidden": 256,
    }

    print("=" * 70)
    print("PPO Evaluation Framework - 严谨研究版")
    print("=" * 70)
    print(f"  Config: {json.dumps(config, indent='  ')}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Random baseline
    print("\n[Random Baseline]")
    env = gym.make(config["env_id"])
    random_bl = evaluate_random(env, n_episodes=500)
    print(f"  SR={random_bl['sr']*100:.1f}% | R={random_bl['reward_mean']:.1f} | L={random_bl['length_mean']:.1f}")

    # Multi-seed training
    print(f"\n[Training {args.seeds} seeds...]")
    all_histories = []
    last_model = None
    for seed in range(42, 42 + args.seeds):
        print(f"\n  --- Seed {seed} ---")
        h, last_model = train_and_evaluate(config["env_id"], seed, config, eval_every=args.eval_every)
        all_histories.append(h)

    # Aggregate
    agg = aggregate_runs(all_histories, random_bl)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Aggregate)")
    print("=" * 70)
    print(f"  Seeds:            {agg['n_seeds']}")
    print(f"  Final SR:         {agg['sr_mean']*100:5.1f}% ± {agg['sr_std']*100:4.1f}%  (SE: {agg['sr_se']*100:.2f}%)")
    print(f"  Final Reward:     {agg['reward_mean_mean']:7.1f} ± {agg['reward_mean_std']:6.1f}")
    print(f"  Final Length:     {agg['length_mean_mean']:6.1f} ± {agg['length_mean_std']:5.1f}")
    print(f"  Random SR:        {agg['random_sr']*100:5.1f}%")
    print(f"  Random Reward:    {agg['random_reward']:7.1f}")
    print(f"  Improvement SR:   +{(agg['sr_mean'] - agg['random_sr'])*100:.1f}% over random")
    print("=" * 70)

    # Save model from last seed
    model_path = os.path.join(SAVE_DIR, f"ppo_final_{ts}.pt")
    torch.save(last_model.state_dict(), model_path)
    print(f"\n[Saved] Model: {model_path}")

    # Save
    path, df = plot_results(agg, config, SAVE_DIR)
    save_csv(df, agg, config, SAVE_DIR)

    # Record episode videos
    eval_env = gym.make(config["env_id"], render_mode="rgb_array")
    record_episode_videos(last_model, eval_env, SAVE_DIR, n_success=2, n_failure=2)
    eval_env.close()

    print("\n[DONE]")
