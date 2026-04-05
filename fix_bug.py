path = r'C:\gym-learning\acrobot_ppo\ppo_impl.py'
content = open(path, 'r', encoding='utf-8').read()

old = '''    for i, (r, l, ok) in enumerate(ep_results[:10]):
        print(f"  Ep {i+1:2d}: reward={r:7.2f}, length={l:4d}, "
              f"goal={'OK' if ok else 'FAIL'}")
    print(f"  ... ({sum(ep_results)}/50 goals reached)")
    print(f"\\n  Final SR: {final_sr:.1f}% | Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)'''

new = '''    successes = ep_results
    print(f"\\n  Goals reached: {sum(successes)}/50")
    print(f"  Final SR: {final_sr:.1f}% | Avg Reward: {final_r:.2f} | Avg Length: {final_l:.1f}")
    print("=" * 60)'''

content = content.replace(old, new)
open(path, 'w', encoding='utf-8').write(content)
print('done')
