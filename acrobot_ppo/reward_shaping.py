"""
Reward Shaping Experiment
三个 shaping term:
  1. base_reward = -1 (原生 sparse reward)
  2. height_shaping = k_h * (cos(theta1) + cos(theta1 + theta2))
     - 越接近"摆起来" reward 越大，∈ [-2k_h, +2k_h]
  3. velocity_shaping = k_v * angular_velocity
     - 鼓励有效摆动，∈ [-k_v*12, +k_v*12]
     (angular_velocity = omega1，即关节1的角速度，范围 ~±12 rad/s)
"""

import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# 模型（与 eval_framework.py 一致）
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
# Reward Shaping Wrapper（三个 shaping term）
# ─────────────────────────────────────────────
class RewardShapingWrapper(gym.Wrapper):
    """
    参数:
        k_h: height shaping 系数（推荐 0.0 ~ 2.0）
        k_v: velocity shaping 系数（推荐 0.0 ~ 0.5）
    """

    def __init__(self, env, k_h=0.5, k_v=0.1):
        super().__init__(env)
        self.k_h = k_h  # height shaping 系数
        self.k_v = k_v  # velocity shaping 系数

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        # obs = [cos(theta1), sin(theta1), cos(theta2), sin(theta2), omega1, omega2]
        cos_t1, sin_t1 = obs[0], obs[1]
        cos_t2, sin_t2 = obs[2], obs[3]
        omega1 = obs[4]

        theta1 = np.arctan2(sin_t1, cos_t1)
        theta2 = np.arctan2(sin_t2, cos_t2)

        # ── term 1: height shaping ──
        # = k_h * (cos(theta1) + cos(theta1 + theta2))
        # 范围 [-2*k_h, +2*k_h]
        height_shaping = self.k_h * (cos_t1 + np.cos(theta1 + theta2))

        # ── term 2: velocity shaping ──
        # = k_v * omega1 (关节1的角速度)
        # omega1 范围约 ±12 rad/s
        velocity_shaping = self.k_v * omega1

        # 到达目标（terminated）保持 0 reward
        if done and not truncated:
            shaped_reward = 0.0
        else:
            shaped_reward = base_reward + height_shaping + velocity_shaping

        return obs, shaped_reward, done, truncated, info


# ─────────────────────────────────────────────
# Rollout
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
# GAE
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
    raw_advantages = advantages.clone()
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    returns = raw_advantages + val_buf
    return returns, advantages


# ─────────────────────────────────────────────
# PPO Update
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
# 诚实评估（deterministic policy）
# ─────────────────────────────────────────────
def evaluate(env, model, n_episodes=50):
    model.eval()
    rewards, lengths, successes = [], [], []
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
        successes.append(done and not truncated)
    return {
        "sr": np.mean(successes) * 100,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
    }


# ─────────────────────────────────────────────
# 单配置训练
# ─────────────────────────────────────────────
def train(config_name, k_h, k_v, total_steps=50000, seed=42, eval_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_env = gym.make("Acrobot-v1")
    env = RewardShapingWrapper(base_env, k_h=k_h, k_v=k_v)
    base_env.close()

    model = ActorCritic(6, 3, hidden=256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_steps = 4096
    total_updates = total_steps // n_steps
    history = {"update": [], "sr": [], "length_mean": [], "reward_mean": []}

    for it in range(1, total_updates + 1):
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, 0.99, 0.95)
        ppo_update(model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
                    0.2, 0.5, 0.01, 256, 10)

        if it % eval_every == 0 or it == total_updates:
            # 评估用原生 env（公平对比 SR 和 length）
            eval_env = gym.make("Acrobot-v1")
            result = evaluate(eval_env, model, n_episodes=50)
            ts = it * n_steps
            history["update"].append(ts)
            history["sr"].append(result["sr"])
            history["length_mean"].append(result["length_mean"])
            history["reward_mean"].append(result["reward_mean"])
            print(f"  [{config_name}] U{it:3d} ts={ts:6d} | "
                  f"SR={result['sr']:5.1f}% | L={result['length_mean']:5.1f} | R={result['reward_mean']:7.2f}")
            eval_env.close()

    env.close()
    return history, model


# ─────────────────────────────────────────────
# Main: 验证可行性
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Reward Shaping Experiment (验证可行性)")
    print("  1. Baseline: k_h=0, k_v=0  (原生 -1 reward)")
    print("  2. Height:   k_h=0.5, k_v=0")
    print("  3. Velocity: k_h=0, k_v=0.1")
    print("  4. Both:     k_h=0.5, k_v=0.1")
    print("=" * 60)

    configs = [
        ("Baseline (k_h=0, k_v=0)",   0.0, 0.0),
        ("Height (k_h=0.5, k_v=0)",   0.5, 0.0),
        ("Velocity (k_h=0, k_v=0.1)", 0.0, 0.1),
        ("Both (k_h=0.5, k_v=0.1)",   0.5, 0.1),
    ]

    all_results = {}

    for config_name, k_h, k_v in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print("=" * 60)
        history, model = train(config_name, k_h, k_v, total_steps=50000, seed=42, eval_every=5)
        df = pd.DataFrame(history)
        final = df.iloc[-1]
        all_results[config_name] = {
            "df": df,
            "final_sr": final["sr"],
            "final_length": final["length_mean"],
            "final_reward": final["reward_mean"],
        }
        print(f"\n  Final: SR={final['sr']:.1f}% | L={final['length_mean']:.1f} | R={final['reward_mean']:.2f}")

        # 保存模型
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(SAVE_DIR, f"ppo_shaped_{config_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_{ts}.pt")
        torch.save(model.state_dict(), model_path)

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Reward Shaping Comparison (50k steps, 1 seed)", fontsize=13, fontweight="bold")

    colors = {"Baseline (k_h=0, k_v=0)": "#f85149",
               "Height (k_h=0.5, k_v=0)": "#58a6ff",
               "Velocity (k_h=0, k_v=0.1)": "#3fb950",
               "Both (k_h=0.5, k_v=0.1)": "#d2a8ff"}

    for name, data in all_results.items():
        df = data["df"]
        c = colors[name]
        lw = 3 if "Baseline" in name else 2
        axes[0].plot(df["update"], df["sr"], color=c, lw=lw, label=name)
        axes[1].plot(df["update"], df["length_mean"], color=c, lw=lw, label=name)
        axes[2].plot(df["update"], df["reward_mean"], color=c, lw=lw, label=name)

    axes[0].set_xlabel("Timesteps"); axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 105)
    axes[1].set_xlabel("Timesteps"); axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length (Efficiency)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[2].set_xlabel("Timesteps"); axes[2].set_ylabel("Episode Reward")
    axes[2].set_title("Episode Reward"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"shaping_compare_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()

    # 汇总
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    baseline_len = all_results["Baseline (k_h=0, k_v=0)"]["final_length"]
    print(f"  {'Config':<30} {'SR':>8} {'Length':>10} {'vs Baseline':>12}")
    print("  " + "-" * 62)
    for name, data in all_results.items():
        diff = data["final_length"] - baseline_len
        sign = "+" if diff > 0 else ""
        print(f"  {name:<30} {data['final_sr']:6.1f}% {data['final_length']:8.1f}  {sign}{diff:+.1f}")
    print("=" * 60)
