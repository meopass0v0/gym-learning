"""
Acrobot PPO Demo - TLM
Using PPO to solve Acrobot swing-up
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import imageio

SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


# ─────────────────────────────────────────────
# Metrics Callback
# ─────────────────────────────────────────────
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.cumulative_reward = 0
        self.success_count = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.cumulative_reward += ep_reward
                self.total_episodes += 1
                if ep_reward > -100:
                    self.success_count += 1
                if self.total_episodes % 20 == 0:
                    sr = self.success_count / self.total_episodes * 100
                    avg_r = np.mean(self.episode_rewards[-20:])
                    avg_l = np.mean(self.episode_lengths[-20:])
                    print(f"[Ep {self.total_episodes}] "
                          f"AvgR={avg_r:7.2f} | AvgL={avg_l:6.1f} | SR={sr:5.1f}%")
        return True

    def get_metrics_df(self):
        df = pd.DataFrame({
            "episode": range(1, len(self.episode_rewards) + 1),
            "reward": self.episode_rewards,
            "length": self.episode_lengths,
        })
        df["cumulative_reward"] = df["reward"].cumsum()
        df["success"] = df["reward"] > -100
        df["success_rate"] = df["success"].rolling(50, min_periods=1).mean() * 100
        df["reward_smooth"] = df["reward"].rolling(50, min_periods=1).mean()
        return df


# ─────────────────────────────────────────────
# Post-training: sample success + failure videos
# ─────────────────────────────────────────────
def sample_and_record(model, n_max=100):
    """
    Run episodes until we collect at least one success and one failure.
    Save them as MP4 using rgb_array (no pygame window).
    Returns (success_frames, failure_frames) lists.
    """
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    success_frames, failure_frames = [], []
    success_meta, failure_meta = None, None

    print("\n[Sample] Collecting success/failure cases...")
    for ep in range(n_max):
        obs, _ = env.reset()
        done = truncated = False
        ep_frames = []
        ep_reward = 0

        while not (done or truncated):
            frame = env.render()
            ep_frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

        success = ep_reward > -100
        meta = {"ep": ep+1, "reward": ep_reward, "length": len(ep_frames)}

        if success and not success_frames:
            success_frames = ep_frames[:]
            success_meta = meta
            print(f"  [SUCCESS] Ep {ep+1}: reward={ep_reward:.1f}, {len(ep_frames)} steps")
        elif not success and not failure_frames:
            failure_frames = ep_frames[:]
            failure_meta = meta
            print(f"  [FAILURE] Ep {ep+1}: reward={ep_reward:.1f}, {len(ep_frames)} steps")

        if success_frames and failure_frames:
            break

    env.close()
    return success_frames, failure_frames, success_meta, failure_meta


def save_video(frames, path, fps=30):
    """Save frames as MP4, limiting to 1500 frames max."""
    if len(frames) > 1500:
        frames = frames[::2]
    print(f"  Encoding {len(frames)} frames -> {path}")
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_learning_curves(df, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Acrobot PPO Training Metrics", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["episode"], df["reward"], alpha=0.3, color="steelblue")
    ax.plot(df["episode"], df["reward_smooth"], color="steelblue", linewidth=2)
    ax.axhline(y=-100, color="red", linestyle="--", label="Success threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward (Higher = Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["episode"], df["success_rate"], color="green", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate (Rolling 50)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1, 0]
    ax.plot(df["episode"], df["length"], alpha=0.3, color="orange")
    ax.plot(df["episode"], df["length"].rolling(50, min_periods=1).mean(),
            color="orange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length (Lower = Faster)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["episode"], df["cumulative_reward"], color="purple", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"learning_curves_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] Learning curves: {path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Acrobot PPO Trainer")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()

    print("=" * 60)
    print("Acrobot PPO Demo - TLM")
    print("=" * 60)

    TOTAL_TIMESTEPS = 50_000

    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    env = Monitor(env)

    print(f"[Env] Acrobot-v1 | Action: {env.action_space} | Obs: {env.observation_space.shape}")
    print(f"[Train] {TOTAL_TIMESTEPS} timesteps | device={args.device}")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=4096, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(SAVE_DIR, "logs"),
        device=args.device
    )

    metrics_cb = MetricsCallback()

    start = datetime.now()
    model.learn(TOTAL_TIMESTEPS, callback=metrics_cb, progress_bar=True)
    train_time = (datetime.now() - start).total_seconds()

    print(f"\n[DONE] Training: {train_time:.1f}s")

    df = metrics_cb.get_metrics_df()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"metrics_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Metrics: {csv_path}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Total Episodes:     {df['episode'].iloc[-1]}")
    print(f"  Final SR (50):      {df['success_rate'].iloc[-1]:.1f}%")
    print(f"  Final Reward (avg):{df['reward'].tail(50).mean():.2f}")
    print(f"  Final Length (avg):{df['length'].tail(50).mean():.1f}")

    plot_learning_curves(df, SAVE_DIR)

    model_path = os.path.join(SAVE_DIR, f"ppo_acrobot_{ts}")
    model.save(model_path)
    print(f"[Saved] Model: {model_path}.zip")

    # ── Post-training: sample success + failure videos ──
    print("\n[Sample] Running post-training case sampling...")
    success_frames, failure_frames, s_meta, f_meta = sample_and_record(model)

    if success_frames:
        path = os.path.join(SAVE_DIR, "sample_success.mp4")
        save_video(success_frames, path)
        print(f"  Success case: {s_meta}")
    else:
        print("  No success case captured")

    if failure_frames:
        path = os.path.join(SAVE_DIR, "sample_failure.mp4")
        save_video(failure_frames, path)
        print(f"  Failure case: {f_meta}")
    else:
        print("  No failure case captured")

    print("\n[DONE] All outputs saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
