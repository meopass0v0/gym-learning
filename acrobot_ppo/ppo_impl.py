"""
PPO Core Implementation - 手写版（修正版）

核心链条:
1. actor-critic 网络
2. rollout 收集（修复: episode 结束后必须 reset）
3. return 计算
4. advantage / GAE（修复: bootstrap 用 next_value）
5. PPO clipped objective
6. value loss
7. entropy bonus
8. minibatch 更新
9. old log prob 缓存
10. 简单训练循环
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

    def get_action(self, obs, deterministic=False):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_action_and_logprob(self, obs, action):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1)


# ─────────────────────────────────────────────
# 2. Rollout 收集（修复: episode 结束后必须 reset）
# ─────────────────────────────────────────────
def collect_rollout(env, model, n_steps):
    """
    收集 n_steps 步数据。

    关键: episode 结束后必须 env.reset()，否则后续数据全在 terminal state
    上踩出来的无效轨迹。
    """
    obs_buffer, act_buffer, rew_buffer = [], [], []
    done_buffer, logp_buffer, val_buffer = [], [], []
    ent_buffer = []

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action(obs)

        obs_np, reward, done, truncated, _ = env.step(action.item())

        # ✅ 关键修复: episode 结束后必须 reset
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
# 3 & 4. Return 计算 + GAE（修复: bootstrap 用 next_value）
# ─────────────────────────────────────────────
def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
    """
    GAE (Generalized Advantage Estimation)

    核心公式（从后向前递推）：
        δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)       # TD residual
        A_t = δ_t + γ * λ * mask * A_{t+1}               # GAE 递推

    其中：
        λ=0 -> 纯 TD(1)，低方差高偏差
        λ=1 -> 纯 Monte Carlo，无偏差高方差
        常用 λ=0.95 做平衡

    注意：
        - A_t 累积的是 previous gae，不是 next_value
        - mask=0 时（episode 结束），GAE 退化为 δ_t = r_t - V(s_t)
        - bootstrap: 最后一个 timestep 用 val_buf[-1]
    """
    n = len(rew_buf)
    advantages = torch.zeros(n, device=DEVICE)

    next_value = val_buf[-1].item()   # bootstrap: 用最后一个状态的 value
    next_gae = 0.0                     # 最末时刻的 gae 初始化为 0

    for t in reversed(range(n)):
        mask = 1.0 - done_buf[t].float()   # 0 if done else 1
        # TD residual: critic 的"意外程度"
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        # ✅ 正确递推：累积的是 gae，不是 next_value
        # A_t = δ_t + γλmask * A_{t+1}
        next_gae = delta + gamma * lam * mask * next_gae
        advantages[t] = next_gae
        # 更新 next_value：当前状态的价值向前传播
        next_value = val_buf[t].item()

    # 标准化只给 actor 用，不污染 critic target
    raw_advantages = advantages.clone()
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    returns = raw_advantages + val_buf  # 用未标准化的原始 advantage 给 critic
    return returns, advantages


# ─────────────────────────────────────────────
# 5, 6, 7. PPO Loss
# ─────────────────────────────────────────────
def ppo_loss(model, obs_b, act_b, old_logp_b, returns_b, advantages_b,
             logp_new_b, ent_b, clip_range, value_coef, ent_coef):
    ratio = torch.exp(logp_new_b - old_logp_b)
    surr1 = ratio * advantages_b
    ratio_clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    surr2 = ratio_clipped * advantages_b
    policy_loss = -torch.min(surr1, surr2).mean()

    values_new = model.critic(model.net(obs_b)).squeeze(-1)
    value_loss = nn.functional.mse_loss(values_new, returns_b)

    entropy_loss = -ent_b.mean()

    return (policy_loss + value_coef * value_loss + ent_coef * entropy_loss,
            policy_loss, value_loss, entropy_loss)


# ─────────────────────────────────────────────
# 8 & 9. Minibatch 更新
# ─────────────────────────────────────────────
def ppo_update(model, optimizer, obs_b, act_b, old_logp_b, returns_b, advantages_b,
               clip_range, value_coef, ent_coef, batch_size, n_epochs):
    n = obs_b.shape[0]
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logp_new, ent, _ = model.get_action_and_logprob(obs_b[mb], act_b[mb])
            loss, _, _, _ = ppo_loss(model, obs_b[mb], act_b[mb], old_logp_b[mb],
                                      returns_b[mb], advantages_b[mb],
                                      logp_new, ent, clip_range, value_coef, ent_coef)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


# ─────────────────────────────────────────────
# 10. 训练循环 + Honest Metrics
# ─────────────────────────────────────────────
def compute_goal_success(env, model, n_episodes=20):
    """
    诚实评估: 用环境的真实终止条件判断成功
    Acrobot: done=True (terminated) = 到达目标
             truncated=True          = 超过步数上限，非成功
    """
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
        # ✅ 关键: done 且 not truncated 才算真正到达目标
        true_successes.append(done and not truncated)

    sr = np.mean(true_successes) * 100
    return np.mean(rewards), np.mean(lengths), sr, true_successes


def train(env_id, total_steps=100_000, n_steps=4096, batch_size=256,
          n_epochs=10, lr=3e-4, gamma=0.99, lam=0.95,
          clip_range=0.2, value_coef=0.5, ent_coef=0.01,
          eval_every=10, log_every=2):
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("=" * 60)
    print("PPO Core Implementation - 修正版")
    print(f"  env={env_id} | obs={obs_dim} | act={act_dim}")
    print(f"  n_steps={n_steps} | batch={batch_size} | epochs={n_epochs}")
    print(f"  lr={lr} | gamma={gamma} | lam={lam} | clip={clip_range}")
    print(f"  device={DEVICE}")
    print("=" * 60)

    total_updates = total_steps // n_steps
    history = {"update": [], "train_reward": [], "train_length": [],
                "eval_reward": [], "eval_length": [], "eval_sr": []}

    for it in range(1, total_updates + 1):
        # ── Rollout ──
        obs_b, act_b, rew_b, done_b, logp_b, val_b, ent_b = collect_rollout(env, model, n_steps)

        # ── GAE ──
        returns_b, advantages_b = compute_returns_and_advantages(val_b, rew_b, done_b, gamma, lam)

        # ── Train Metrics（诚实统计）─
        # 切分真实的 episode：累积 reward/length，done 时才算一个 episode 结束
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
        train_sr = np.mean([r > -100 for r in ep_rewards]) * 100 if ep_rewards else 0.0

        # ── Update ──
        ppo_update(model, optimizer, obs_b, act_b, logp_b, returns_b, advantages_b,
                   clip_range, value_coef, ent_coef, batch_size, n_epochs)

        # ── Log ──
        if it % log_every == 0:
            print(f"[U{it:3d}/{total_updates}] "
                  f"train_R={avg_train_r:7.2f} | train_L={avg_train_l:6.1f} | "
                  f"train_SR={train_sr:5.1f}%")

        # ── Honest Eval ──
        if it % eval_every == 0 or it == total_updates:
            eval_r, eval_l, eval_sr, _ = compute_goal_success(env, model, n_episodes=20)
            history["update"].append(it)
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PPO Training - Honest Metrics", fontsize=14, fontweight="bold")

    axes[0].plot(updates, history["eval_reward"], "b-", linewidth=2)
    axes[0].set_xlabel("Update"); axes[0].set_ylabel("Avg Reward")
    axes[0].set_title("Average Episode Reward (Higher = Better)")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=-100, color="red", linestyle="--", label="success threshold")
    axes[0].legend()

    axes[1].plot(updates, history["eval_length"], "orange", linewidth=2)
    axes[1].set_xlabel("Update"); axes[1].set_ylabel("Avg Length")
    axes[1].set_title("Average Episode Length (Lower = Faster)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(updates, history["eval_sr"], "g-", linewidth=2)
    axes[2].set_xlabel("Update"); axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title("Goal Success Rate (Higher = Better)")
    axes[2].set_ylim(0, 105); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"ppo_honest_{ts}.png")
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
        clip_range=0.2, value_coef=0.5, ent_coef=0.01,
        eval_every=5, log_every=2
    )

    # 最终诚实评估
    env = gym.make("Acrobot-v1")
    final_r, final_l, final_sr, successes = compute_goal_success(env, model, n_episodes=50)
    print("\n" + "=" * 60)
    print("Final Honest Evaluation (50 episodes)")
    print(f"  Goals reached: {sum(successes)}/50")
    print(f"  Final SR: {final_sr:.1f}% | Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)

    plot(history, save_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(save_dir, f"ppo_from_scratch_{ts}.pt"))
    print(f"\n[Saved] ppo_from_scratch_{ts}.pt")
