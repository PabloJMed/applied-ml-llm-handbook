"""Checks for the worked numbers in Chapter 39 (thinking like a senior).

Run: python3 code/ch39_check.py
"""
import math
from scipy import stats

PEAK = 989e12          # H100 dense bf16 FLOP/s
MFU = 0.40
def gpu_hours(n_params, tokens):
    return 6 * n_params * tokens / (PEAK * MFU) / 3600

print("== 39.10 research scoping: ablation ladder at ~20 tokens per parameter")
ladder = {"150M": (150e6, 3e9), "400M": (400e6, 8e9), "1B": (1e9, 20e9), "3B": (3e9, 60e9)}
gh = {k: gpu_hours(*v) for k, v in ladder.items()}
for k, v in gh.items():
    print(f"  {k}: {v:,.0f} GPU-hours per run")
budget = 16 * 24 * 21
print(f"budget: 16 GPUs x 3 weeks = {budget:,} GPU-hours")
A = 2 * gh["1B"]
sweep = 2 * 5 * (gh["150M"] + gh["400M"])
seeds = 2 * 3 * (gh["150M"] + gh["400M"] + gh["1B"])
B = sweep + seeds
C = B + 2 * gh["3B"]
for name, v in (("A: one 1B run per arm", A), ("B: ladder + LR sweep + 3 seeds", B), ("C: B + one 3B run per arm", C)):
    print(f"  option {name}: {v:,.0f} GPU-hours = {v / budget:.0%} of budget")
print(f"  wall-clock of one 3B run on 16 GPUs: {gh['3B'] / 16:.0f} h")

print("\n== 39.11 triage: cost of a decision on a large run")
gpus, days_left, ckpt_gap_h, price = 2048, 18, 4, 2.0
print(f"rollback of {ckpt_gap_h} h: {gpus*ckpt_gap_h:,} GPU-hours = ${gpus*ckpt_gap_h*price:,.0f}")
print(f"remaining run: {gpus*24*days_left:,} GPU-hours = ${gpus*24*days_left*price/1e6:.2f}M")
print(f"one hour of the whole run: {gpus} GPU-hours = ${gpus*price:,.0f}")

print("\n== Q4: was the A/B test powered?")
p, n = 0.10, 200_000
se = math.sqrt(2 * p * (1 - p) / n)
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
mde = z * se
print(f"SE of difference = {se*100:.3f} points; MDE = {mde*100:.2f} points = {mde/p:.1%} relative")

print("\n== Q5: an interval for +1.2% with SE 0.46")
est, se5 = 1.2, 0.46
print(f"95% CI = {est-1.96*se5:+.1f} to {est+1.96*se5:+.1f}")

print("\n== Q3: churn triage arithmetic")
active, churn = 10_000, 0.02
print(f"monthly churners = {active*churn:.0f}")
