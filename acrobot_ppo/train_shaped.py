"""
Reward Shaping Training Experiment
用 shaped reward 重新训练，对比 baseline (原 reward) 和 shaped reward 两种配置
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# 模型
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
# Reward Shaping Wrapper
# ─────────────────────────────────────────────
def compute_height(obs):
    theta1 = np.arctan2(obs[1], obs[0])
    theta2 = np.arctan2(obs[3], obs[2])
    return -np.cos(theta1) - np.cos(theta2 + theta1)


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, shaping_coef=0.0):
        super().__init__(env)
        self.shaping_coef = shaping_coef
        self.prev_height = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_height = compute_height(obs)
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        current_height = compute_height(obs)
        height_delta = current_height - self.prev_height
        self.prev_height = current_height

        if done and not truncated:
            shaped_reward = 0.0
        else:
            shaped_reward = base_reward + self.shaping_coef * height_delta

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
# 诚实评估
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
# 训练 + 对比
# ─────────────────────────────────────────────
def train_with_reward(reward_config_name, shaping_coef, total_steps=100_000,
                      n_steps=4096, batch_size=256, n_epochs=10, lr=3e-4,
                      gamma=0.99, lam=0.95, clip_range=0.2,
                      value_coef=0.5, ent_coef=0.01, eval_every=5, seed=42):
    """
    用指定 reward 配置训练一个 seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建环境（wrapped 或原生）
    base_env = gym.make("Acrobot-v1")
    if shaping_coef > 0:
        env = RewardShapingWrapper(base_env, shaping_coef=shaping_coef)
    else:
        env = base_env

    model = ActorCritic(6, 3, hidden=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_updates = total_steps // n_steps
    history = {"update": [], "sr": [], "length_mean": [], "reward_mean": []}

    for it in range(1, total_updates + 1):
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, gamma, lam)
        ppo_update(model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
                    clip_range, value_coef, ent_coef, batch_size, n_epochs)

        if it % eval_every == 0 or it == total_updates:
            # 评估用原生 env（公平对比）
            eval_env = gym.make("Acrobot-v1")
            result = evaluate(eval_env, model, n_episodes=50)
            ts = it * n_steps
            history["update"].append(ts)
            history["sr"].append(result["sr"])
            history["length_mean"].append(result["length_mean"])
            history["reward_mean"].append(result["reward_mean"])
            print(f"  [{reward_config_name}] U{it:3d} ts={ts:6d} | "
                  f"SR={result['sr']:5.1f}% | L={result['length_mean']:5.1f} | R={result['reward_mean']:7.2f}")
            eval_env.close()

    env.close()
    base_env.close()
    return history, model


# ─────────────────────────────────────────────
# Main: 对比实验
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    configs = [
        ("Baseline (r=-1)",        0.0),
        ("Shaped (coef=0.5)",     0.5),
        ("Shaped (coef=1.0)",     1.0),
        ("Shaped (coef=2.0)",     2.0),
    ]

    print("=" * 70)
    print("Reward Shaping Training Experiment")
    print(f"  Steps: {args.steps} | Seeds per config: {args.seeds}")
    print("=" * 70)

    all_results = {}

    for config_name, shaping_coef in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config_name} (shaping_coef={shaping_coef})")
        print("=" * 70)

        seed_histories = []
        for seed in range(42, 42 + args.seeds):
            print(f"\n  --- Seed {seed} ---")
            h, model = train_with_reward(
                config_name, shaping_coef,
                total_steps=args.steps, seed=seed
            )
            seed_histories.append(h)

        # 聚合多 seed
        df = pd.DataFrame({
            "update": seed_histories[0]["update"],
            "sr_mean": np.mean([h["sr"] for h in seed_histories], axis=0),
            "sr_std": np.std([h["sr"] for h in seed_histories], axis=0),
            "length_mean": np.mean([h["length_mean"] for h in seed_histories], axis=0),
            "length_std": np.std([h["length_mean"] for h in seed_histories], axis=0),
            "reward_mean": np.mean([h["reward_mean"] for h in seed_histories], axis=0),
        })

        final = df.iloc[-1]
        all_results[config_name] = {
            "df": df,
            "final_sr": final["sr_mean"],
            "final_sr_std": final["sr_std"],
            "final_length": final["length_mean"],
            "final_length_std": final["length_std"],
        }

        print(f"\n  [{config_name}] Final: SR={final['sr_mean']:.1f}% | L={final['length_mean']:.1f}±{final['length_std']:.1f}")

    # ── 可视化对比 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Reward Shaping Comparison", fontsize=14, fontweight="bold")

    colors = {"Baseline (r=-1)": "#f85149", "Shaped (coef=0.5)": "#58a6ff",
               "Shaped (coef=1.0)": "#3fb950", "Shaped (coef=2.0)": "#d2a8ff"}

    for config_name, data in all_results.items():
        df = data["df"]
        color = colors[config_name]
        lw = 3 if "Baseline" in config_name else 2
        ax = axes[0]
        ax.plot(df["update"], df["sr_mean"], color=color, lw=lw, label=config_name)
        ax.fill_between(df["update"],
                        np.clip(df["sr_mean"] - df["sr_std"], 0, 100),
                        np.clip(df["sr_mean"] + df["sr_std"], 0, 100),
                        color=color, alpha=0.1)
        ax = axes[1]
        ax.plot(df["update"], df["length_mean"], color=color, lw=lw, label=config_name)
        ax.fill_between(df["update"],
                        np.clip(df["length_mean"] - df["length_std"], 0, None),
                        df["length_mean"] + df["length_std"],
                        color=color, alpha=0.1)

    axes[0].set_xlabel("Timesteps"); axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 105)
    axes[1].set_xlabel("Timesteps"); axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length (Efficiency)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Bar chart: final comparison
    names = list(all_results.keys())
    x = np.arange(len(names))
    lengths = [all_results[n]["final_length"] for n in names]
    length_stds = [all_results[n]["final_length_std"] for n in names]
    axes[2].bar(x, lengths, yerr=length_stds, color=[colors[n] for n in names],
                capsize=5, alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels([n.split(" ")[0] for n in names], rotation=15)
    axes[2].set_ylabel("Final Episode Length")
    axes[2].set_title("Final Length Comparison (lower = better)")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"shaping_compare_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()

    # 汇总表格
    print("\n" + "=" * 70)
    print("Final Summary Table")
    print("=" * 70)
    print(f"{'Config':<25} {'SR':>8} {'Length':>10} {'vs Baseline':>12}")
    print("-" * 70)
    baseline_len = all_results["Baseline (r=-1)"]["final_length"]
    for name, data in all_results.items():
        diff = data["final_length"] - baseline_len
        sign = "+" if diff > 0 else ""
        print(f"  {name:<23} {data['final_sr']:6.1f}%  {data['final_length']:6.1f}±{data['final_length_std']:.1f}  {sign}{diff:+.1f}")
    print("=" * 70)
