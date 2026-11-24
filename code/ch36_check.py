"""Chapter 36 · Research interviews: the numbers behind the worked answers (NumPy + SciPy, seconds)."""
import numpy as np
from scipy import stats

GPU_HOUR = 989e12 * 0.4 * 3600            # FLOPs per H100-hour at 40% MFU (dense bf16 peak), as in Chapter 34.7

# 36.2 budget of the synthetic-maths design: anneals of 2B tokens at 400M and 6B at 1.4B, 3 arms x 3 seeds
small = (6 * 0.4e9 * 2e9 + 6 * 1.4e9 * 6e9) * 3 * 3
big = 6 * 8e9 * 100e9 * 2                 # 8B confirmation: two 100B-token anneals
print("36.2 small-scale phase %.2e FLOPs = %.0f GPU-hours; 8B confirmation %.2e = %.0f GPU-hours; small share %.2f"
      % (small, small / GPU_HOUR, big, big / GPU_HOUR, small / (small + big)))
# minimum detectable effect on a 3,000-item aggregate at 60%
n, p = 3000, 0.6
print("36.2 SE one model %.2f pts; difference, independent %.2f pts; paired with 15%% discordant %.2f pts"
      % (100 * np.sqrt(p * (1 - p) / n), 100 * np.sqrt(2 * p * (1 - p) / n), 100 * np.sqrt(0.15 / n)))

# 36.2 second worked design: long-context drop 72% -> 64% on 400 items
se = np.sqrt(0.72 * 0.28 / 400 + 0.64 * 0.36 / 400)
print("36.2 long-context drop: SE of difference %.1f pts, drop = %.1f SE" % (100 * se, 0.08 / se))
print("36.2 best-of-8 with a perfect verifier at 30%% per sample: %.3f" % (1 - 0.7 ** 8))

# 36.8 DPO: the closed form is the maximiser of the KL-regularised objective (four-response bandit)
r = np.array([1.0, 0.5, 0.0, -1.0]); beta = 0.5; ref = np.array([0.1, 0.2, 0.3, 0.4])
def objective(pi): return (pi * r).sum() - beta * (pi * np.log(pi / ref)).sum()
Z = (ref * np.exp(r / beta)).sum(); pi_star = ref * np.exp(r / beta) / Z
rng = np.random.default_rng(0)
others = rng.dirichlet(np.ones(4), 20000)
print("36.8 pi* =", pi_star.round(3), "objective %.4f = beta log Z %.4f; best of 20,000 random policies %.4f"
      % (objective(pi_star), beta * np.log(Z), max(objective(q) for q in others)))
# identity: objective(pi) = -beta KL(pi || pi*) + beta log Z for any pi
q = others[0]
print("36.8 identity holds:", np.isclose(objective(q), -beta * (q * np.log(q / pi_star)).sum() + beta * np.log(Z)))
# loss at initialisation and at a margin; gradient weight
sig = lambda x: 1 / (1 + np.exp(-x))
print("36.8 DPO loss at pi = pi_ref: %.3f; at beta x margin = 0.1 x 1.5: %.3f, weight sigma(-0.15) = %.2f"
      % (-np.log(sig(0.0)), -np.log(sig(0.15)), sig(-0.15)))
# gradient check: dL/dh for L = -log sigma(beta * h) is -beta * sigma(-beta h)
h, eps = 1.5, 1e-6
num = (-np.log(sig(0.1 * (h + eps))) + np.log(sig(0.1 * (h - eps)))) / (2 * eps)
print("36.8 dL/dmargin numeric %.4f vs -beta*sigma(-beta*h) %.4f" % (num, -0.1 * sig(-0.15)))

# Interview Q4: one week on 64 GPUs
print("Q4: 64 x 168 = %d GPU-hours; ladder 1,350 = %.0f h on 64 GPUs; 7B run 4,130 = %.0f h; share used %.2f"
      % (64 * 168, 1350 / 64, 4130 / 64, (1350 + 4130) / (64 * 168)))
# Interview Q9: McNemar on 500 paired prompts, 40 vs 15 discordant
b, c = 40, 15
chi2 = (b - c) ** 2 / (b + c)
print("Q9 McNemar chi2 = %.1f, p = %.4f (exact binomial p = %.4f)" % (chi2, stats.chi2.sf(chi2, 1), stats.binomtest(c, b + c).pvalue))
