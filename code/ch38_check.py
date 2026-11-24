"""Checks for the numbers in Chapter 38's worked stories (behavioural interviews).

Run: python3 code/ch38_check.py
"""
import math
from scipy import stats

z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)

print("== 38.12 research story: a single-seed gain that was seed noise")
sigma = 0.6        # seed-to-seed std of the 8-benchmark average, points
gain_1seed = 1.1
se1 = sigma * math.sqrt(2)                     # one seed per arm
print(f"one seed per arm: SE of difference = {se1:.2f}; gain/SE = {gain_1seed/se1:.2f}")
n = 5
diff5 = 0.3
se5 = sigma * math.sqrt(2 / n)
print(f"{n} seeds per arm: SE of difference = {se5:.2f}; 95% CI = {diff5-1.96*se5:+.1f} to {diff5+1.96*se5:+.1f}")
for delta in (0.5, 1.0):
    print(f"seeds per arm to detect {delta} points at 80% power: {math.ceil(2 * z**2 * sigma**2 / delta**2)}")

print("\n== Story 5 failure: extraction change inflated a maths score")
reported, from_extraction = 3.0, 2.4
print(f"true gain = {reported - from_extraction:.1f} points; share of reported gain from the harness = {from_extraction/reported:.0%}")

print("\n== 38.10 mid-scope example: shared eval harness")
runs_per_week, minutes_saved = 40, 25
print(f"{runs_per_week} runs/week x {minutes_saved} min = {runs_per_week*minutes_saved/60:.0f} engineer-hours/week saved")
spread = [61.8, 63.4, 65.9]
print(f"spread across three teams' prompts = {max(spread)-min(spread):.1f} points")

print("\n== 38.12 cost of the extra seeds")
flops = 6 * 1e9 * 25e9                  # 1B parameters, 25B tokens
gpu_h = flops / (989e12 * 0.40) / 3600  # H100 dense bf16 at 40% MFU
print(f"one 1B/25B ablation: {flops:.1e} FLOPs = {gpu_h:.0f} H100-hours at 40% MFU; 8 extra runs = {8*gpu_h:.0f}")
