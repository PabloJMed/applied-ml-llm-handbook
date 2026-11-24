"""Chapter 34 · Research craft: runnable checks (NumPy + SciPy, seconds)."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# 34.6 how many seeds? Power to detect a true improvement of 0.3 points when seed-to-seed SD is 0.4
true_gain, seed_sd = 0.3, 0.4
for n_seeds in (1, 3, 5, 10):
    hits = 0
    for _ in range(4000):
        a = rng.normal(0, seed_sd, n_seeds); b = rng.normal(true_gain, seed_sd, n_seeds)
        if n_seeds == 1:
            hits += b[0] > a[0]                      # "it went up"
        else:
            hits += stats.ttest_ind(b, a, equal_var=False, alternative="greater").pvalue < 0.05
    label = "P(new > old)" if n_seeds == 1 else "power at p<0.05"
    print(f"{n_seeds:>2} seeds per arm: {label} = {hits/4000:.2f}")
print("one seed each, no real gain: P(new looks better) = 0.50 by symmetry; P(gain > 0.3) = %.2f"
      % (1 - stats.norm.cdf(0.3 / (seed_sd * np.sqrt(2)))))

# 34.6 winner's curse in a hyperparameter sweep: 12 configs, all equally good
sweep = rng.normal(70.0, 0.5, (5000, 12))           # 12 configurations with the same true score 70.0
best = sweep.max(1)
print(f"12 equal configs, noise SD 0.5: best observed score averages {best.mean():.2f} (true 70.00); "
      f"re-running the winner gives 70.00 on average")

# 34.5 one-at-a-time ablations miss interactions: a 2x2 factorial
# true effects: A alone +1.0, B alone +0.0, A and B together +2.0 (interaction +1.0)
def score(a, b): return 50 + 1.0 * a + 0.0 * b + 1.0 * a * b + rng.normal(0, 0.2)
base = score(0, 0)
print("one-at-a-time from the baseline: A %+.2f, B %+.2f  -> 'B does nothing'" % (score(1, 0) - base, score(0, 1) - base))
grid = {(a, b): np.mean([score(a, b) for _ in range(3)]) for a in (0, 1) for b in (0, 1)}
print("factorial: B's effect without A %+.2f, with A %+.2f" % (grid[(0, 1)] - grid[(0, 0)], grid[(1, 1)] - grid[(1, 0)]))
print("leave-one-out from the full system: removing B costs %.2f" % (grid[(1, 1)] - grid[(1, 0)]))
print("factorial cell means (A, B):", {k: round(v, 2) for k, v in grid.items()})

# 34.7 compute budget for a project: a 4-size ladder x 3 learning rates x 2 seeds, plus one 7B confirmation
ladder = [(125e6, 2.5e9), (350e6, 7e9), (760e6, 15e9), (1.4e9, 28e9)]
gpu_flops = 989e12 * 0.4                               # one GPU at 40% MFU
sweep_flops = sum(6 * N * D for N, D in ladder) * 3 * 2
final_flops = 6 * 7e9 * 140e9
print(f"ladder sweep: {sweep_flops:.2e} FLOPs = {sweep_flops/gpu_flops/3600:,.0f} GPU-hours; "
      f"7B x 140B confirmation: {final_flops/gpu_flops/3600:,.0f} GPU-hours")

# 34.8 determinism: the same seed gives the same numbers only if everything else is fixed too
import torch
def run(seed):
    torch.manual_seed(seed)
    w = torch.randn(64, 64); x = torch.randn(32, 64)
    return (x @ w).relu().sum().item()
print("same seed twice:", run(1) == run(1), " different seed:", run(1) == run(2))

# 34.11 budget of the worked curriculum study: 3 sizes at 20 tokens/param, 2 arms x 3 seeds, 40% MFU on 989 TFLOP/s
sizes = (125e6, 350e6, 760e6)
per_set = sum(6 * n * 20 * n for n in sizes)
total = per_set * 6
print("FLOPs per set of sizes %.2e; total %.2e = %.0f GPU-hours" % (per_set, total, total / (0.4 * 989e12) / 3600))
print("largest single run: %.1f GPU-hours" % (6 * 760e6 * 20 * 760e6 / (0.4 * 989e12) / 3600))

# 34.6 formulas behind the tables
za, zb = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
print("seeds per arm, normal approximation: %.1f (d/sigma = 0.75), %.1f (0.375), %.1f (1.5)"
      % tuple(2 * (za + zb) ** 2 / r ** 2 for r in (0.75, 0.375, 1.5)))
def t_power(n, d):
    df = 2 * n - 2
    return 1 - stats.nct.cdf(stats.t.ppf(0.95, df), df, d * np.sqrt(n / 2))
print("exact t-test power at d/sigma = 0.75: n=22 %.3f, n=23 %.3f" % (t_power(22, 0.75), t_power(23, 0.75)))
print("one seed each: P(new > old) = Phi(%.2f) = %.2f" % (0.3 / (0.4 * np.sqrt(2)), stats.norm.cdf(0.3 / (0.4 * np.sqrt(2)))))
from scipy import integrate
def e_max(k):
    return integrate.quad(lambda x: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1), -10, 10)[0]
print("E[max of k standard normals]:", {k: round(e_max(k), 2) for k in (2, 5, 10, 12, 30, 100)},
      "sqrt(2 ln 12) = %.2f" % np.sqrt(2 * np.log(12)))
# 34.11 how uncertain is a standard deviation from few seeds?
for df in (2, 4, 9):
    lo, hi = np.sqrt(df / stats.chi2.ppf(0.975, df)), np.sqrt(df / stats.chi2.ppf(0.025, df))
    print(f"95% interval for sigma with {df} df: {lo:.2f} to {hi:.2f} times the estimate")
print("power with 3 seeds if the true SD is 6.3x the estimate: %.2f" % t_power(3, (0.005 / 0.003) / 6.28))
# 34.7 per-GPU-hour FLOPs and the ladder's share
print("FLOPs per GPU-hour at 40%% MFU: %.3e; ladder share of budget %.2f" % (989e12 * 0.4 * 3600, 1349 / (1349 + 4129)))
# 34.12 chance benchmarks, extrapolation, multiple comparisons
print("SE at 25%% on 1,000 items: %.2f points" % (100 * np.sqrt(0.25 * 0.75 / 1000)))
print("exponent error 0.01 over 100x compute: %.3f of the reducible loss" % (1 - 100 ** -0.01))
print("P(at least one false positive in 20 tests) = %.2f; Bonferroni threshold %.4f" % (1 - 0.95 ** 20, 0.05 / 20))
print("SE of a 500-question score at 70%%: %.2f points; of a difference of two: %.2f" % (100 * np.sqrt(0.21 / 500), 100 * np.sqrt(2 * 0.21 / 500)))
