path = r'C:\gym-learning\acrobot_ppo\ppo_impl.py'
content = open(path, 'r', encoding='utf-8').read()

old = '''def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
    """
    GAE 计算。

    修复说明:
    - 每个 timestep t 的 advantage 需要 V(s_{t+1})，即下一步的价值
    - 对于非终止状态，用 val_buf[t+1]
    - 对于终止状态（done=True），不需要 V(s_{t+1})，因为 mask=0 会把它消掉
    - bootstrap: 最后一个 timestep 用 val_buf[-1] 作为 next_value
    """
    n = len(rew_buf)
    advantages = torch.zeros(n, device=DEVICE)

    # ✅ 正确: 用最后一个 value 作为 bootstrap 的 next_value
    next_value = val_buf[-1].item()

    for t in reversed(range(n)):
        mask = 1.0 - done_buf[t].float()   # 0 if done else 1
        delta = rew_buf[t] + gamma * next_value * mask - val_buf[t]
        advantages[t] = gae = delta + gamma * lam * mask * next_value
        next_value = val_buf[t].item()      # 当前 step 的 value 作为下一步的 next_value

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + val_buf
    return returns, advantages'''

new = '''def compute_returns_and_advantages(val_buf, rew_buf, done_buf, gamma, lam):
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

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + val_buf
    return returns, advantages'''

content = content.replace(old, new)
open(path, 'w', encoding='utf-8').write(content)
print('done')
