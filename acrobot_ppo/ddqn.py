"""
Double DQN + Replay Buffer for Acrobot-v1

- Main network 和 Target network 分离
- Double DQN: 动作选择用 main network，Q值评估用 target network
- Replay Buffer: 容量 100000，batch_size 64
- ε-贪婪: 从 1.0 线性衰减到 0.01
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# Q Network
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.net[-1].out_features - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            q_values = self.forward(obs_t)
            return q_values.argmax().item()


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(env, q_network, n_episodes=50):
    q_network.eval()
    rewards, lengths, successes = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, steps = 0, 0
        done = truncated = False

        while not (done or truncated):
            action = q_network.get_action(obs, epsilon=0.0)
            obs, reward, done, truncated, _ = env.step(action)
            total_r += reward
            steps += 1

        rewards.append(total_r)
        lengths.append(steps)
        successes.append(done and not truncated)

    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "length_mean": np.mean(lengths),
        "sr": np.mean(successes),
        "sr_se": np.std(successes) / np.sqrt(len(successes)),
    }


def evaluate_random(env, n_episodes=200):
    rewards, lengths, successes = [], [], []
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
        successes.append(done and not truncated)
    return {
        "reward_mean": np.mean(rewards),
        "sr": np.mean(successes),
        "length_mean": np.mean(lengths),
    }


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_ddqn(env_id, seed, config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Main network and Target network
    q_main = QNetwork(obs_dim, act_dim, hidden=config["hidden"]).to(DEVICE)
    q_target = QNetwork(obs_dim, act_dim, hidden=config["hidden"]).to(DEVICE)
    q_target.load_state_dict(q_main.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_main.parameters(), lr=config["lr"])

    replay_buffer = ReplayBuffer(capacity=config["replay_capacity"])

    # Exploration schedule
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = config["epsilon_decay"]

    # Training history
    history = {
        "step": [],
        "epsilon": [],
        "train_loss": [],
        "reward_mean": [],
        "reward_std": [],
        "sr": [],
        "sr_se": [],
        "length_mean": [],
    }

    obs, _ = env.reset()
    total_steps = 0
    episode_count = 0

    while total_steps < config["total_steps"]:
        # Current epsilon
        epsilon = max(epsilon_end, epsilon_start - total_steps / epsilon_decay)

        # Select action
        action = q_main.get_action(obs, epsilon=epsilon)

        # Step
        next_obs, reward, done, truncated, _ = env.step(action)
        real_done = done or truncated

        replay_buffer.push(obs, action, reward, next_obs, real_done)
        obs = next_obs

        # Train when buffer has enough samples
        if len(replay_buffer) >= config["batch_size"]:
            # Sample batch
            states, actions, rewards_b, next_states, dones = replay_buffer.sample(config["batch_size"])

            # Double DQN: use main network to select action, target network to evaluate
            with torch.no_grad():
                # Best action from main network
                best_actions = q_main(next_states).argmax(dim=1)
                # Q value from target network for those actions
                target_q = q_target(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                # TD target
                td_target = rewards_b + config["gamma"] * target_q * (1 - dones)
                # mask for terminal states handled above

            # Current Q from main network
            current_q = q_main(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = nn.functional.mse_loss(current_q, td_target)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_main.parameters(), 1.0)
            optimizer.step()

            # Record loss
            if total_steps % config.get("log_interval", 1000) == 0:
                history["train_loss"].append(loss.item())

        # Update target network
        if total_steps % config["target_update_freq"] == 0:
            q_target.load_state_dict(q_main.state_dict())

        # Reset if episode ended
        if done or truncated:
            obs, _ = env.reset()
            episode_count += 1

        total_steps += 1

        # Periodic evaluation
        if total_steps % config["eval_freq"] == 0 or total_steps >= config["total_steps"]:
            result = evaluate(env, q_main, n_episodes=50)
            history["step"].append(total_steps)
            history["epsilon"].append(epsilon)
            history["reward_mean"].append(result["reward_mean"])
            history["reward_std"].append(result["reward_std"])
            history["sr"].append(result["sr"])
            history["sr_se"].append(result["sr_se"])
            history["length_mean"].append(result["length_mean"])

            print(f"  [seed={seed}] steps={total_steps:6d} | "
                  f"SR={result['sr']*100:5.1f}% | "
                  f"R={result['reward_mean']:7.1f} | "
                  f"ε={epsilon:.4f}")

    env.close()
    return history, q_main


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_results(histories, random_baseline, config, save_dir):
    # Average across seeds
    steps = histories[0]["step"]
    sr_mean = np.mean([h["sr"] for h in histories], axis=0)
    sr_std = np.std([h["sr"] for h in histories], axis=0)
    r_mean = np.mean([h["reward_mean"] for h in histories], axis=0)
    r_std = np.std([h["reward_mean"] for h in histories], axis=0)
    loss_mean = np.mean([h["train_loss"] for h in histories], axis=0) if histories[0]["train_loss"] else None
    eps_mean = np.mean([h["epsilon"] for h in histories], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Double DQN on {config['env_id']} | {len(histories)} seeds", fontsize=13, fontweight="bold")

    # Success Rate
    ax = axes[0, 0]
    ax.plot(steps, sr_mean, color='#3fb950', lw=2)
    ax.fill_between(steps,
                    np.clip(sr_mean - sr_std, 0, 1),
                    np.clip(sr_mean + sr_std, 0, 1),
                    color='#3fb950', alpha=0.15)
    ax.axhline(y=random_baseline["sr"], color='#f85149', linestyle='--',
               label=f"Random {random_baseline['sr']*100:.1f}%")
    ax.set_xlabel("Steps"); ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate"); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # Reward
    ax = axes[0, 1]
    ax.plot(steps, r_mean, color='#58a6ff', lw=2)
    ax.fill_between(steps, r_mean - r_std, r_mean + r_std, color='#58a6ff', alpha=0.15)
    ax.axhline(y=random_baseline["reward_mean"], color='#f85149', linestyle='--',
               label=f"Random {random_baseline['reward_mean']:.1f}")
    ax.set_xlabel("Steps"); ax.set_ylabel("Avg Reward")
    ax.set_title("Episode Reward"); ax.legend(); ax.grid(alpha=0.3)

    # Loss
    if loss_mean is not None and len(loss_mean) > 0:
        ax = axes[1, 0]
        ax.plot(loss_mean, color='#f0883e', lw=2)
        ax.set_xlabel("Update"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss"); ax.grid(alpha=0.3)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(steps, eps_mean, color='#d2a8ff', lw=2)
    ax.set_xlabel("Steps"); ax.set_ylabel("Epsilon")
    ax.set_title("Exploration (ε)"); ax.grid(alpha=0.3)

    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"ddqn_results_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {path}")
    return path


def save_csv(history, config, save_dir):
    import pandas as pd
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, f"ddqn_data_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")
    return csv_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epsilon-decay", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    args = parser.parse_args()

    config = {
        "env_id": "Acrobot-v1",
        "total_steps": args.steps,
        "hidden": args.hidden,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "target_update_freq": args.target_update_freq,
        "epsilon_decay": args.epsilon_decay,
        "eval_freq": args.eval_freq,
        "log_interval": args.eval_freq,
    }

    print("=" * 70)
    print("Double DQN + Replay Buffer")
    print("=" * 70)
    print(f"  Hidden:         {config['hidden']}")
    print(f"  LR:             {config['lr']}")
    print(f"  Gamma:          {config['gamma']}")
    print(f"  Batch Size:     {config['batch_size']}")
    print(f"  Replay:         {config['replay_capacity']}")
    print(f"  Target Update:  every {config['target_update_freq']} steps")
    print(f"  Epsilon Decay:  {config['epsilon_decay']} steps")
    print(f"  Seeds:          {args.seeds}")
    print(f"  Device:         {DEVICE}")
    print("=" * 70)

    # Random baseline
    print("\n[Random Baseline]")
    env = gym.make(config["env_id"])
    random_bl = evaluate_random(env, n_episodes=500)
    print(f"  SR={random_bl['sr']*100:.1f}% | R={random_bl['reward_mean']:.1f}")
    env.close()

    # Train
    print(f"\n[Training {args.seeds} seeds...]")
    all_histories = []
    for seed in range(42, 42 + args.seeds):
        print(f"\n  --- Seed {seed} ---")
        h, _ = train_ddqn(config["env_id"], seed, config)
        all_histories.append(h)

    # Plot
    plot_results(all_histories, random_bl, config, SAVE_DIR)

    # Final eval
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for i, h in enumerate(all_histories):
        sr = h["sr"][-1] if h["sr"] else 0
        r = h["reward_mean"][-1] if h["reward_mean"] else 0
        print(f"  Seed {42+i}: SR={sr*100:5.1f}% | R={r:7.1f}")
    print(f"  Random:        SR={random_bl['sr']*100:5.1f}%")
    print("=" * 70)

    print("\n[DONE]")
