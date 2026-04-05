"""
Acrobot PPO - 轻量评估脚本
默认只输出数值指标，不渲染窗口
按需生成视频

用法:
  python eval.py          # 仅数值评估（快速、无窗口）
  python eval.py --video  # 生成 MP4 视频
"""

import argparse
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
import imageio

SAVE_DIR = r"C:\gym-learning\acrobot_ppo"


def find_latest_model():
    models = [f for f in os.listdir(SAVE_DIR)
               if f.startswith("ppo_acrobot_") and f.endswith(".zip")]
    models.sort()
    return os.path.join(SAVE_DIR, models[-1])


def eval_only(model_path, n_episodes=20):
    """纯数值评估，不渲染窗口"""
    model = PPO.load(model_path)
    env = gym.make("Acrobot-v1", render_mode=None)  # 无渲染

    rewards, lengths, successes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = truncated = False
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(total_reward > -100)

        if ep < 5:  # 只打印前5个
            status = "OK" if successes[-1] else "FAIL"
            print(f"  Ep {ep+1:2d}: reward={total_reward:7.2f}, length={steps:4d} [{status}]")

    env.close()

    sr = sum(successes) / len(successes) * 100
    print(f"\n[SUMMARY] Success Rate: {sr:5.1f}% | "
          f"Avg Reward: {np.mean(rewards):7.2f} | "
          f"Avg Length: {np.mean(lengths):6.1f} | "
          f"Min Reward: {min(rewards):7.2f}")
    return sr, np.mean(rewards), np.mean(lengths)


def eval_with_video(model_path, n_episodes=3):
    """评估 + 生成 MP4"""
    model = PPO.load(model_path)
    env = gym.make("Acrobot-v1", render_mode="rgb_array")

    all_frames = []
    ep_info = []

    print("\n[Record] Capturing episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = 0
        ep_reward = 0

        while not (done or truncated):
            frame = env.render()
            all_frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

        success = ep_reward > -100
        ep_info.append((ep+1, steps, ep_reward, "OK" if success else "FAIL"))
        print(f"  Ep {ep+1}: {steps} steps, reward={ep_reward:.1f} "
              f"[{'OK' if success else 'FAIL'}]")

    env.close()

    # 限制帧数
    if len(all_frames) > 1500:
        all_frames = all_frames[::2]

    mp4_path = os.path.join(SAVE_DIR, "acrobot_eval.mp4")
    print(f"\n[Encode] Saving MP4 ({len(all_frames)} frames)...")
    imageio.mimwrite(mp4_path, all_frames, fps=30, codec="libx264", quality=8)
    print(f"[Saved] {mp4_path}")

    # 数值评估
    return eval_only(model_path, n_episodes=10)


def main():
    parser = argparse.ArgumentParser(description="Acrobot PPO Evaluator")
    parser.add_argument("--video", action="store_true",
                        help="Generate MP4 video after eval")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default: 20)")
    args = parser.parse_args()

    model_path = find_latest_model()
    print(f"[Model] {model_path}")

    if args.video:
        eval_with_video(model_path, n_episodes=3)
    else:
        print(f"\n[Eval] {args.episodes} episodes (no rendering)...")
        eval_only(model_path, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
