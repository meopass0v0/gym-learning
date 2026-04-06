"""
PPO Evaluation Framework - Critic-Separated Architecture

主要改动（相比 master 分支）：
1. Actor/Critic 分离 backbone，不再共享
2. Critic hidden size 更大（默认 512 vs actor 256）
3. Critic 独立优化器，lr 可单独配置
4. Critic 每个 batch 更新多次（critic_n_epochs > n_epochs）
5. 支持 --critic-hidden, --critic-lr, --critic-n-epochs 参数

理论依据：
- Actor 100% 成功说明策略结构没问题
- 62 vs 60 这种细微 reward 差距需要更精细的 value 估计
- Advantage = r + gamma*V(s') - V(s)，value 估计不准则 advantage 信号噪声大
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
import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# 1. Actor-Critic Network (Separated Backbones)
# ─────────────────────────────────────────────
class ActorCriticSep(nn.Module):
    """
    Actor 和 Critic 分离 backbone，各自独立学习。

    设计考量：
    - Actor 需要保持策略分布的平滑性，不宜过大
    - Critic 需要更精细的状态价值估计，更大 hidden 能捕获更复杂的状态差异
    - 分离后各自优化互不干扰
    """

    def __init__(self, obs_dim, act_dim, hidden=256, critic_hidden=512):
        super().__init__()

        # Actor backbone：保持和原来一样的大小
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)

        # Critic backbone：更大，更深
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden), nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden // 2), nn.Tanh(),
        )
        self.critic_head = nn.Linear(critic_hidden // 2, 1)

    def forward(self, x):
        """
        返回 (logits, value)
        """
        actor_features = self.actor_net(x)
        logits = self.actor_head(actor_features)

        critic_features = self.critic_net(x)
        value = self.critic_head(critic_features).squeeze(-1)

        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), val

    def get_action_and_logprob(self, obs, action):
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), val


# ─────────────────────────────────────────────
# 2. Rollout
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
# 3. GAE
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
# 4. PPO Update (分离 Critic 更新)
# ─────────────────────────────────────────────
def ppo_update_separated(model, actor_opt, critic_opt,
                         obs_b, act_b, old_logp_b, returns_b, advantages_b,
                         clip_range, ent_coef,
                         batch_size, n_epochs, critic_n_epochs):
    """
    分离的 PPO 更新：
    - Actor 每 batch 更新 n_epochs 次
    - Critic 每 batch 更新 critic_n_epochs 次（更多次，更强的 value 学习）
    """
    n = obs_b.shape[0]

    # ── Critic 更新（多个 epoch）──
    for _ in range(critic_n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            _, _, val = model.get_action_and_logprob(obs_b[mb], act_b[mb])
            # 用 returns 作为 target，mse loss
            critic_loss = nn.functional.mse_loss(val, returns_b[mb])
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.critic_head.parameters(), 0.5)
            critic_opt.step()

    # ── Actor 更新 ──
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
            entropy_loss = -ent.mean()
            loss = policy_loss + ent_coef * entropy_loss
            actor_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.actor_head.parameters(), 0.5)
            actor_opt.step()


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate_honest(env, model, n_episodes=50):
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
    def __init__(self, env, k_h=0.0, k_v=0.0):
        super().__init__(env)
        self.k_h = k_h
        self.k_v = k_v

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        cos_t1, sin_t1 = obs[0], obs[1]
        cos_t2, sin_t2 = obs[2], obs[3]
        omega1 = obs[4]

        theta1 = np.arctan2(sin_t1, cos_t1)
        theta2 = np.arctan2(sin_t2, cos_t2)

        height_shaping = self.k_h * (cos_t1 + np.cos(theta1 + theta2))
        velocity_shaping = self.k_v * omega1

        if done and not truncated:
            shaped_reward = 0.0
        else:
            shaped_reward = base_reward + height_shaping + velocity_shaping

        return obs, shaped_reward, done, truncated, info


# ─────────────────────────────────────────────
# 6. 训练
# ─────────────────────────────────────────────
def train_and_evaluate(env_id, seed, config, eval_every=5, start_update=1,
                       loaded_model=None, loaded_actor_opt=None, loaded_critic_opt=None):
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

    hidden = config.get("hidden", 256)
    critic_hidden = config.get("critic_hidden", 512)

    if loaded_model is not None:
        model = loaded_model
    else:
        model = ActorCriticSep(obs_dim, act_dim, hidden=hidden, critic_hidden=critic_hidden).to(DEVICE)

    # 分离的优化器
    if loaded_actor_opt is not None:
        actor_opt = loaded_actor_opt
    else:
        actor_opt = optim.Adam(model.actor_net.parameters(), lr=config["lr"])
        actor_opt.add_param_group({
            "params": model.actor_head.parameters(),
            "lr": config["lr"]
        })

    if loaded_critic_opt is not None:
        critic_opt = loaded_critic_opt
    else:
        critic_lr = config.get("critic_lr", 1e-3)
        critic_opt = optim.Adam(
            list(model.critic_net.parameters()) + list(model.critic_head.parameters()),
            lr=critic_lr
        )

    n_steps = config["n_steps"]
    total_updates = config["total_steps"] // n_steps
    n_epochs = config.get("n_epochs", 10)
    critic_n_epochs = config.get("critic_n_epochs", 4)  # 默认 critic 更新更多次
    batch_size = config.get("batch_size", 256)

    history = {
        "update": [], "timestep": [],
        "sr": [], "sr_se": [],
        "reward_mean": [], "reward_se": [],
        "length_mean": [], "length_std": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
    }

    for it in range(start_update, total_updates + 1):
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, config["gamma"], config["lam"])

        # 记录训练 loss（用更新前的模型）
        with torch.no_grad():
            logp_new, ent_new, val_new = model.get_action_and_logprob(obs_b, act_b)
            ratio = torch.exp(logp_new - logp_b)
            surr1 = ratio * advantages_b
            ratio_clipped = torch.clamp(ratio, 1 - config["clip_range"], 1 + config["clip_range"])
            surr2 = ratio_clipped * advantages_b
            pol_loss = (-torch.min(surr1, surr2)).mean().item()
            val_loss = nn.functional.mse_loss(val_new, returns_b).item()
            ent_loss = ent_new.mean().item()

        ppo_update_separated(
            model, actor_opt, critic_opt,
            obs_b, act_b, logp_b, returns_b, advantages_b,
            config["clip_range"], config["ent_coef"],
            batch_size, n_epochs, critic_n_epochs
        )

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

    return history, model, actor_opt, critic_opt


# ─────────────────────────────────────────────
# 7. 聚合统计
# ─────────────────────────────────────────────
def aggregate_runs(all_histories, random_baseline):
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
# 8. Video Recording
# ─────────────────────────────────────────────
def record_episode_videos(model, env, save_dir, n_success=1, n_failure=1):
    model.eval()
    success_frames, failure_frames = [], []
    success_meta, failure_meta = [], []
    max_frames = 1500

    print("\n[Video] Recording episodes...")

    for ep in range(n_success * 10 + n_failure * 10):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        ep_frames = []
        total_r, steps = 0, 0
        done = truncated = False

        while not (done or truncated):
            frame = env.render()
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
# 9. 可视化
# ─────────────────────────────────────────────
def plot_results(agg, config, save_dir):
    histories = agg["all_histories"]

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
    fig.suptitle(f"PPO Sep-Critic on {config['env_id']} | {agg['n_seeds']} seeds | "
                 f"lr={config['lr']} critic_lr={config.get('critic_lr','?')}", fontsize=13, fontweight="bold")

    ts = df["timestep"]

    ax = axes[0, 0]
    sr = df["sr_mean"].values
    sr_err = df["sr_std"].values / np.sqrt(agg["n_seeds"]) * 1.96
    ax.plot(ts, sr, color=C.green, lw=2)
    ax.fill_between(ts, np.clip(sr - sr_err, 0, 1), np.clip(sr + sr_err, 0, 1),
                   color=C.green, alpha=0.15)
    ax.axhline(y=agg["random_sr"], color=C.red, linestyle="--", label=f"Random {agg['random_sr']*100:.1f}%")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[0, 1]
    rm = df["reward_mean"].values
    rm_err = df["reward_std"].values / np.sqrt(agg["n_seeds"]) * 1.96
    ax.plot(ts, rm, color=C.blue, lw=2)
    ax.fill_between(ts, rm - rm_err, rm + rm_err, color=C.blue, alpha=0.15)
    ax.axhline(y=agg["random_reward"], color=C.red, linestyle="--", label=f"Random {agg['random_reward']:.1f}")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Episode Reward (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    lm = df["length_mean"].values
    lm_err = df["length_std"].values / np.sqrt(agg["n_seeds"]) * 1.96
    ax.plot(ts, lm, color=C.orange, lw=2)
    ax.fill_between(ts, np.clip(lm - lm_err, 0, None), lm + lm_err,
                   color=C.orange, alpha=0.15)
    ax.axhline(y=agg["random_length"], color=C.red, linestyle="--", label=f"Random {agg['random_length']:.1f}")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Avg Episode Length")
    ax.set_title("Episode Length (95% CI band)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, h in enumerate(histories):
        ax.plot(h["timestep"], h["sr"], alpha=0.4, lw=1)
    ax.plot(ts, df["sr_mean"], color=C.green, lw=2, label="Mean")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Success Rate")
    ax.set_title(f"Individual Seeds (n={agg['n_seeds']})"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    ax = axes[1, 1]
    ax.plot(ts, df["pol_loss"], color=C.orange, lw=2, label="Policy Loss")
    ax.plot(ts, df["val_loss"], color=C.blue, lw=2, label="Value Loss")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Loss")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(alpha=0.3)

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
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"eval_data_{ts_str}.csv")
    df.to_csv(csv_path, index=False)

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
# 0. Checkpoint 管理
# ─────────────────────────────────────────────
def get_latest_checkpoint(save_dir):
    pattern = os.path.join(save_dir, "ppo_final_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None, None
    latest_pt = max(files, key=os.path.getmtime)
    basename = os.path.basename(latest_pt)
    ts_part = os.path.splitext(basename)[0].replace("ppo_final_", "")
    config_path = os.path.join(save_dir, f"config_{ts_part}.json")
    return latest_pt, config_path


def save_checkpoint(model, actor_opt, critic_opt, config, save_dir, current_update):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"ppo_final_{ts}.pt")
    config_path = os.path.join(save_dir, f"config_{ts}.json")

    torch.save({
        "model_state": model.state_dict(),
        "actor_opt_state": actor_opt.state_dict(),
        "critic_opt_state": critic_opt.state_dict(),
        "current_update": current_update,
    }, model_path)

    config_with_meta = dict(config)
    config_with_meta["_checkpoint_ts"] = ts
    with open(config_path, "w") as f:
        json.dump(config_with_meta, f, indent=2)
    return model_path, config_path


def load_checkpoint(checkpoint_path, config_path, device=DEVICE, default_config=None):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            print(f"[WARN] config file not found, using default")
            config = dict(default_config) if default_config else get_default_config()

        env = gym.make(config["env_id"])
        k_h = config.get("k_h", 0.0)
        k_v = config.get("k_v", 0.0)
        if k_h > 0 or k_v > 0:
            env = RewardShapingWrapper(env, k_h=k_h, k_v=k_v)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        env.close()

        hidden = config.get("hidden", 256)
        critic_hidden = config.get("critic_hidden", 512)
        model = ActorCriticSep(obs_dim, act_dim, hidden=hidden, critic_hidden=critic_hidden).to(device)
        model.load_state_dict(ckpt["model_state"])

        actor_opt = optim.Adam(model.actor_net.parameters(), lr=config["lr"])
        actor_opt.add_param_group({"params": model.actor_head.parameters(), "lr": config["lr"]})

        critic_lr = config.get("critic_lr", 1e-3)
        critic_opt = optim.Adam(
            list(model.critic_net.parameters()) + list(model.critic_head.parameters()),
            lr=critic_lr
        )

        if "actor_opt_state" in ckpt:
            actor_opt.load_state_dict(ckpt["actor_opt_state"])
        if "critic_opt_state" in ckpt:
            critic_opt.load_state_dict(ckpt["critic_opt_state"])

        current_update = ckpt.get("current_update", 1)
    else:
        # 旧格式
        print(f"[WARN] Old checkpoint format, using default config")
        config = dict(default_config) if default_config else get_default_config()

        env = gym.make(config["env_id"])
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        env.close()

        model = ActorCriticSep(obs_dim, act_dim).to(device)
        model.load_state_dict(ckpt)

        actor_opt = optim.Adam(model.actor_net.parameters(), lr=config["lr"])
        critic_opt = optim.Adam(
            list(model.critic_net.parameters()) + list(model.critic_head.parameters()),
            lr=config.get("critic_lr", 1e-3)
        )
        current_update = 1

    return model, actor_opt, critic_opt, config, current_update


def get_default_config():
    return {
        "env_id": "Acrobot-v1",
        "total_steps": 2000000,
        "n_steps": 8192,
        "batch_size": 256,
        "n_epochs": 10,
        "critic_n_epochs": 4,
        "lr": 3e-4,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "ent_coef": 0.01,
        "hidden": 256,
        "critic_hidden": 512,
    }


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
    parser.add_argument("--load-latest", action="store_true")
    parser.add_argument("--continue-train", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    # Critic 相关参数
    parser.add_argument("--hidden", type=int, default=256, help="Actor hidden size")
    parser.add_argument("--critic-hidden", type=int, default=512, help="Critic hidden size")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--critic-n-epochs", type=int, default=4, help="Critic updates per batch")
    args = parser.parse_args()

    config = {
        "env_id": "Acrobot-v1",
        "total_steps": args.steps,
        "n_steps": 8192,
        "batch_size": 256,
        "n_epochs": 10,
        "critic_n_epochs": args.critic_n_epochs,
        "lr": 3e-4,
        "critic_lr": args.critic_lr,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "ent_coef": 0.01,
        "hidden": args.hidden,
        "critic_hidden": args.critic_hidden,
    }

    print("=" * 70)
    print("PPO Evaluation Framework - Critic-Separated Architecture")
    print("=" * 70)
    print(f"  Actor hidden:     {config['hidden']}")
    print(f"  Critic hidden:    {config['critic_hidden']}")
    print(f"  Actor lr:         {config['lr']}")
    print(f"  Critic lr:        {config['critic_lr']}")
    print(f"  Critic n_epochs:  {config['critic_n_epochs']} (per batch)")
    print(f"  Seeds:            {args.seeds}")
    print(f"  Device:          {DEVICE}")
    print("=" * 70)

    if args.load_latest or args.checkpoint or args.continue_train:
        if args.checkpoint:
            ckpt_path = args.checkpoint
            basename = os.path.basename(ckpt_path)
            ts_part = os.path.splitext(basename)[0].replace("ppo_final_", "")
            config_path = os.path.join(os.path.dirname(ckpt_path), f"config_{ts_part}.json")
        else:
            ckpt_path, config_path = get_latest_checkpoint(SAVE_DIR)

        if ckpt_path is None:
            print(f"[ERROR] 找不到 checkpoint")
            exit(1)

        print(f"[Load] Checkpoint: {ckpt_path}")
        print(f"[Load] Config:     {config_path if os.path.exists(config_path) else '(none)'}")
        model, actor_opt, critic_opt, config, current_update = load_checkpoint(
            ckpt_path, config_path, default_config=config)

        print(f"[Info] current_update={current_update}")

        if args.continue_train:
            print(f"\n[Continue Training] from update {current_update}")
            env = gym.make(config["env_id"])
            result = evaluate_honest(env, model, n_episodes=50)
            print(f"  [Before] SR={result['sr']*100:5.1f}% | R={result['reward_mean']:7.1f}")
            env.close()

            h, last_model, last_actor_opt, last_critic_opt = train_and_evaluate(
                config["env_id"], seed=42, config=config,
                eval_every=args.eval_every,
                start_update=current_update + 1,
                loaded_model=model,
                loaded_actor_opt=actor_opt,
                loaded_critic_opt=critic_opt,
            )

            final_update = h["update"][-1] if h["update"] else current_update
            model_path, config_path_new = save_checkpoint(
                last_model, last_actor_opt, last_critic_opt, config, SAVE_DIR, final_update)
            print(f"\n[Saved] Model: {model_path}")

            env = gym.make(config["env_id"])
            result = evaluate_honest(env, last_model, n_episodes=200)
            print(f"\n[Final] SR={result['sr']*100:5.1f}% | R={result['reward_mean']:7.1f}")
            env.close()

            eval_env = gym.make(config["env_id"], render_mode="rgb_array")
            record_episode_videos(last_model, eval_env, SAVE_DIR, n_success=2, n_failure=2)
            eval_env.close()
            print("\n[DONE]")
        else:
            print("\n[Evaluate]")
            env = gym.make(config["env_id"])
            result = evaluate_honest(env, model, n_episodes=200)
            print(f"  SR={result['sr']*100:5.1f}% | R={result['reward_mean']:7.1f}")
            env.close()
            print("\n[DONE]")
        exit(0)

    # ── 训练模式 ──
    print(f"  Config: {json.dumps(config, indent='  ')}")

    print("\n[Random Baseline]")
    env = gym.make(config["env_id"])
    random_bl = evaluate_random(env, n_episodes=500)
    print(f"  SR={random_bl['sr']*100:.1f}% | R={random_bl['reward_mean']:.1f} | L={random_bl['length_mean']:.1f}")

    print(f"\n[Training {args.seeds} seeds...]")
    all_histories = []
    last_model, last_actor_opt, last_critic_opt, last_update = None, None, None, 0
    for seed in range(42, 42 + args.seeds):
        print(f"\n  --- Seed {seed} ---")
        h, last_model, last_actor_opt, last_critic_opt = train_and_evaluate(
            config["env_id"], seed, config, eval_every=args.eval_every)
        all_histories.append(h)
        last_update = h["update"][-1] if h["update"] else 0

    agg = aggregate_runs(all_histories, random_bl)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Final SR:         {agg['sr_mean']*100:5.1f}% ± {agg['sr_std']*100:4.1f}%")
    print(f"  Final Reward:     {agg['reward_mean_mean']:7.1f}")
    print(f"  Random SR:        {agg['random_sr']*100:5.1f}%")
    print("=" * 70)

    model_path, config_path = save_checkpoint(
        last_model, last_actor_opt, last_critic_opt, config, SAVE_DIR, last_update)
    print(f"\n[Saved] Model: {model_path}")

    path, df = plot_results(agg, config, SAVE_DIR)
    save_csv(df, agg, config, SAVE_DIR)

    eval_env = gym.make(config["env_id"], render_mode="rgb_array")
    record_episode_videos(last_model, eval_env, SAVE_DIR, n_success=2, n_failure=2)
    eval_env.close()

    print("\n[DONE]")
