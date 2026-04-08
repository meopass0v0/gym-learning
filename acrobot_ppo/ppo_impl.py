"""
PPO Core Implementation - 手写版（完整优化版）

包含:
1. actor-critic 网络
2. rollout 收集（episode 结束后必须 reset）
3. return 计算 + GAE
4. PPO clipped objective
5. Value clipping（与 policy clip 对称）
6. KL early stopping
7. Entropy bonus
8. Minibatch 更新
9. LR / Clip decay
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 1. Actor-Critic 网络
# ─────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.net(x)
        return self.actor(features), self.critic(features)

    def get_action(self, obs):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_action_and_logprob(self, obs, action):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# ─────────────────────────────────────────────
# 2. Rollout 收集
# ─────────────────────────────────────────────
def collect_rollout(env, model, n_steps):
    obs_buffer, act_buffer, rew_buffer = [], [], []
    done_buffer, logp_buffer, val_buffer = [], [], []
    ent_buffer = []

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

        obs_buffer.append(obs)
        act_buffer.append(action)
        rew_buffer.append(reward)
        done_buffer.append(done or truncated)
        logp_buffer.append(log_prob)
        val_buffer.append(value)
        ent_buffer.append(entropy)

        obs = obs_new

    return (torch.stack(obs_buffer), torch.stack(act_buffer),
            torch.tensor(rew_buffer, device=DEVICE),
            torch.tensor(done_buffer, device=DEVICE),
            torch.stack(logp_buffer), torch.stack(val_buffer),
            torch.stack(ent_buffer))


# ─────────────────────────────────────────────
# 3 & 4. GAE
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
# 5, 6, 7. PPO Loss (with Value Clipping)
# ─────────────────────────────────────────────
def ppo_loss(model, obs_b, act_b, old_logp_b, old_v_b, returns_b, advantages_b,
             logp_new_b, ent_b, clip_range, value_coef, ent_coef):
    # Policy loss (PPO clipped objective)
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss with clipping (symmetric with policy clip)
    values_new = model.critic(model.net(obs_b)).squeeze(-1)
    v_clipped = old_v_b + torch.clamp(values_new - old_v_b, -clip_range, clip_range)
    value_loss = torch.max(
        (values_new - returns_b) ** 2,
        (v_clipped - returns_b) ** 2
    ).mean()

    # Entropy loss
    entropy_loss = -ent_b.mean()

    return (policy_loss + value_coef * value_loss + ent_coef * entropy_loss,
            policy_loss, value_loss, entropy_loss)


# ─────────────────────────────────────────────
# 8 & 9. Minibatch 更新 (with KL Early Stopping)
# ─────────────────────────────────────────────
def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, old_v_b,
               returns_b, advantages_b, clip_range, value_coef, ent_coef,
               batch_size, n_epochs, target_kl=None):
    n = obs_b.shape[0]

    for epoch in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        kl_sum = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent, _ = model.get_action_and_logprob(obs_b[mb], act_b[mb])

            loss, p_loss, v_loss, ent_loss = ppo_loss(
                model, obs_b[mb], act_b[mb], old_logp_b[mb], old_v_b[mb],
                returns_b[mb], advantages_b[mb],
                logp_new, ent, clip_range, value_coef, ent_coef)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            kl_sum += (logp_new - old_logp_b[mb]).mean().item()
            num_batches += 1

        # KL early stopping
        avg_kl = kl_sum / num_batches if num_batches > 0 else 0.0
        if target_kl is not None and avg_kl > target_kl:
            return p_loss.item(), v_loss.item(), ent_loss.item(), avg_kl, epoch + 1

    return p_loss.item(), v_loss.item(), ent_loss.item(), avg_kl, n_epochs


# ─────────────────────────────────────────────
# 线性衰减
# ─────────────────────────────────────────────
def linear_decay(initial, final, progress):
    return initial - (initial - final) * progress


# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def compute_goal_success(env, model, n_episodes=20):
    model.eval()
    rewards, lengths, true_successes = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        total_r = 0
        steps = 0
        done = truncated = False

        while not (done or truncated):
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t)
            obs, reward, done, truncated, _ = env.step(action.item())
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            total_r += reward
            steps += 1

        rewards.append(total_r)
        lengths.append(steps)
        true_successes.append(done and not truncated)

    model.train()
    sr = np.mean(true_successes) * 100
    return np.mean(rewards), np.mean(lengths), sr, true_successes


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train(env_id, total_steps=100_000, n_steps=4096, batch_size=256,
          n_epochs=10, lr=3e-4, gamma=0.99, lam=0.95,
          clip_range_init=0.2, clip_range_final=0.1,
          value_coef=0.5, ent_coef=0.01,
          target_kl=0.015,
          eval_every=10, log_every=2):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("=" * 60)
    print("PPO Core Implementation - 完整优化版")
    print(f"  env={env_id} | obs={obs_dim} | act={act_dim}")
    print(f"  n_steps={n_steps} | batch={batch_size} | epochs={n_epochs}")
    print(f"  lr={lr} | gamma={gamma} | lam={lam}")
    print(f"  clip: {clip_range_init} -> {clip_range_final}")
    print(f"  value_clip + kl_early_stop + lr/clip_decay")
    print(f"  device={DEVICE}")
    print("=" * 60)

    total_updates = total_steps // n_steps
    history = {"update": [], "train_reward": [], "train_length": [],
               "eval_reward": [], "eval_length": [], "eval_sr": [],
               "policy_loss": [], "value_loss": [], "entropy": [], "kl": []}

    for it in range(1, total_updates + 1):
        progress = (it - 1) / max(total_updates - 1, 1)

        # Decay
        current_lr = linear_decay(lr, lr * 0.1, progress)
        current_clip = linear_decay(clip_range_init, clip_range_final, progress)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Rollout
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)

        # GAE
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, gamma, lam)

        # Train Metrics
        ep_rewards, ep_lengths = [], []
        ep_r, ep_l = 0.0, 0
        for i in range(n_steps):
            ep_r += rew_b[i].item()
            ep_l += 1
            if done_b[i]:
                ep_rewards.append(ep_r)
                ep_lengths.append(ep_l)
                ep_r, ep_l = 0.0, 0

        avg_train_r = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_train_l = np.mean(ep_lengths) if ep_lengths else 0.0

        # Update
        p_loss, v_loss, entropy, kl, _ = ppo_update(
            model, optimizer, obs_b, act_b, logp_b, val_b,
            returns_b, advantages_b,
            current_clip, value_coef, ent_coef,
            batch_size, n_epochs, target_kl)

        # Log
        if it % log_every == 0:
            print(f"[U{it:3d}/{total_updates}] "
                  f"train_R={avg_train_r:7.2f} | train_L={avg_train_l:6.1f} | "
                  f"p_loss={p_loss:.4f} | v_loss={v_loss:.4f} | "
                  f"ent={entropy:.4f} | kl={kl:.4f}")

        # Eval
        if it % eval_every == 0 or it == total_updates:
            eval_r, eval_l, eval_sr, _ = compute_goal_success(env, model, n_episodes=20)
            history["update"].append(it)
            history["policy_loss"].append(p_loss)
            history["value_loss"].append(v_loss)
            history["entropy"].append(entropy)
            history["kl"].append(kl)
            history["eval_reward"].append(eval_r)
            history["eval_length"].append(eval_l)
            history["eval_sr"].append(eval_sr)
            print(f"  >> [Eval] goal_SR={eval_sr:5.1f}% | eval_R={eval_r:7.2f} | eval_L={eval_l:5.1f}")

    return model, history


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────
def plot(history, save_dir):
    updates = history["update"]
    if not updates:
        return

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    fig.suptitle("PPO Training - Acrobot", fontsize=14, fontweight="bold")

    def subplot(ax, y, title, ylabel, color, hline=None):
        ax.plot(updates, y, color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if hline:
            ax.axhline(y=hline, color="green", linestyle="--", alpha=0.7)

    subplot(axes[0, 0], history["eval_reward"], "Avg Reward", "Reward", "blue")
    subplot(axes[0, 1], history["eval_length"], "Avg Length", "Length", "orange")
    subplot(axes[0, 2], history["eval_sr"], "Success Rate (%)", "SR", "green", 100)
    subplot(axes[1, 0], history["policy_loss"], "Policy Loss", "Loss", "red")
    subplot(axes[1, 1], history["value_loss"], "Value Loss", "Loss", "purple")
    subplot(axes[1, 2], history["kl"], "KL Divergence", "KL", "brown")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"ppo_full_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.close()


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    save_dir = r"C:\gym-learning\acrobot_ppo"
    model, history = train(
        env_id="Acrobot-v1",
        total_steps=args.steps,
        n_steps=4096, batch_size=256, n_epochs=10,
        lr=3e-4, gamma=0.99, lam=0.95,
        clip_range_init=0.2, clip_range_final=0.1,
        value_coef=0.5, ent_coef=0.01,
        target_kl=0.015,
        eval_every=5, log_every=2
    )

    # 最终评估
    env = gym.make("Acrobot-v1")
    final_r, final_l, final_sr, successes = compute_goal_success(env, model, n_episodes=50)
    print("\n" + "=" * 60)
    print("Final Honest Evaluation (50 episodes)")
    print(f"  Goals reached: {sum(successes)}/50")
    print(f"  Final SR: {final_sr:.1f}% | Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)

    plot(history, save_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(save_dir, f"ppo_full_{ts}.pt"))
    print(f"\n[Saved] ppo_full_{ts}.pt")
