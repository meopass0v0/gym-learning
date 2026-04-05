"""
Acrobot PPO - 高效视频录制
使用 imageio 直接从 rgb_array 合成视频
"""

import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from stable_baselines3 import PPO
import imageio

# 找到最新的模型
SAVE_DIR = r"C:\gym-learning\acrobot_ppo"
models = [f for f in os.listdir(SAVE_DIR) if f.startswith("ppo_acrobot_") and f.endswith(".zip")]
models.sort()
latest_model = os.path.join(SAVE_DIR, models[-1])
print(f"Loading: {latest_model}")

model = PPO.load(latest_model)

# 录制配置
env = gym.make("Acrobot-v1", render_mode="rgb_array")

print("\n[Record] Capturing episodes...")
all_frames = []
ep_info = []

for ep in range(5):
    obs, _ = env.reset()
    done = False
    truncated = False
    ep_reward = 0
    steps = 0

    while not (done or truncated):
        frame = env.render()  # (H, W, 3) uint8
        all_frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1

    success = ep_reward > -100
    ep_info.append({"ep": ep+1, "steps": steps, "reward": ep_reward, "success": success})
    print(f"  Episode {ep+1}: {steps} steps, reward={ep_reward:.1f}, {'OK' if success else 'FAIL'}")

env.close()

# 每隔2帧取一帧，限制总帧数在 1500 以内
if len(all_frames) > 1500:
    step = len(all_frames) // 1500
    all_frames = all_frames[::step]
print(f"Total frames for video: {len(all_frames)}")

# 合成 MP4
print("\n[Encode] Saving MP4...")
mp4_path = os.path.join(SAVE_DIR, "acrobot_ppo_eval.mp4")
imageio.mimwrite(mp4_path, all_frames, fps=30, codec="libx264", quality=8)
print(f"[Saved] MP4: {mp4_path}")

# 同时跑一次 human 渲染看结果（快速）
print("\n[Eval] Quick eval (10 eps, human render)...")
eval_env = gym.make("Acrobot-v1", render_mode="human")
rewards = []
lengths = []
successes = []

for ep in range(10):
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False
    truncated = False
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        steps += 1

    rewards.append(total_reward)
    lengths.append(steps)
    successes.append(total_reward > -100)
    print(f"  Episode {ep+1}: reward={total_reward:7.2f}, length={steps:4d} [{'OK' if successes[-1] else 'FAIL'}]")

eval_env.close()

sr = sum(successes) / len(successes) * 100
print(f"\n[Summary] Success Rate: {sr:.0f}% | Avg Reward: {np.mean(rewards):.2f} | Avg Length: {np.mean(lengths):.1f}")
