import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
base = rng.normal(30, 8, 50000)   # basket value AED
print("same", round(psi(base, rng.normal(30, 8, 20000)), 4))
print("shift +2", round(psi(base, rng.normal(32, 8, 20000)), 4))
print("ramadan-like +6, wider", round(psi(base, rng.normal(36, 11, 20000)), 4))
print("ks p", stats.ks_2samp(base[:5000], rng.normal(32, 8, 5000)).pvalue)
# canary guardrail: error rate comparison
def two_prop(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); z = (x2 / n2 - x1 / n1) / se
    return z, 1 - stats.norm.cdf(z)
print(two_prop(120, 95000, 21, 5000))

# --- 47.4 power of the canary test: detect a 0.10-point rise from a 0.13% error rate, canary at 5% ---
p0, p1 = 0.0013, 0.0023
z_a, z_b = stats.norm.ppf(0.99), stats.norm.ppf(0.8)       # one-sided alpha = 1%, power 80%
k = 19                                                       # control gets 19 requests per canary request
var = p0 * (1 - p0) / k + p1 * (1 - p1)                      # per canary request
n_c = (z_a + z_b) ** 2 * var / (p1 - p0) ** 2
print(f"canary requests needed {n_c:,.0f} (total traffic {n_c * 20:,.0f}); expected canary errors {n_c * p1:.0f}")

# --- peeking: a fixed-horizon z-test checked every 2,000 canary requests inflates false rollbacks ---
rng2 = np.random.default_rng(1)
def false_rollbacks(n_sims=2000, looks=25, step=2000, alpha=0.01):
    hits_peek = hits_once = 0
    for _ in range(n_sims):
        xc = np.cumsum(rng2.binomial(step * 19, p0, looks)); xt = np.cumsum(rng2.binomial(step, p0, looks))
        nc = step * 19 * np.arange(1, looks + 1); nt = step * np.arange(1, looks + 1)
        p = (xc + xt) / (nc + nt); se = np.sqrt(p * (1 - p) * (1 / nc + 1 / nt))
        z = np.where(se > 0, (xt / nt - xc / nc) / np.where(se > 0, se, 1), 0)
        crit = stats.norm.ppf(1 - alpha)
        hits_peek += (z > crit).any(); hits_once += z[-1] > crit
    return hits_once / n_sims, hits_peek / n_sims
print("false rollback rate: single look %.3f, peeking every 2,000 requests %.3f" % false_rollbacks())

# --- SPRT (Wald) for the canary error rate: H0 p = p0, H1 p = p1 ---
alpha, beta = 0.01, 0.2
A, B = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))
llr_err, llr_ok = np.log(p1 / p0), np.log((1 - p1) / (1 - p0))
print(f"SPRT bounds {B:.2f}, {A:.2f}; per error +{llr_err:.3f}, per success {llr_ok:.6f}")
def sprt(p, n_max=200_000):
    llr, n = 0.0, 0
    while n < n_max:
        x = rng2.binomial(1000, p); n += 1000
        llr += x * llr_err + (1000 - x) * llr_ok
        if llr >= A: return "rollback", n
        if llr <= B: return "pass", n
    return "undecided", n
for p, label in ((p0, "healthy"), (p1, "bad")):
    res = [sprt(p) for _ in range(1000)]
    rb = np.mean([r == "rollback" for r, _ in res]); nn = np.mean([n for _, n in res])
    print(f"SPRT {label}: rollback rate {rb:.3f}, mean canary requests {nn:,.0f}")
bad_big = [sprt(0.0042) for _ in range(500)]
print("SPRT when canary is at 0.42%: mean requests", f"{np.mean([n for _, n in bad_big]):,.0f}")

# --- 47.5 PSI = symmetric KL; sensitivity to bins and to the clip ---
def kl(p, q): return float(np.sum(p * np.log(p / q)))
e = np.array([0.25, 0.25, 0.25, 0.25]); a = np.array([0.20, 0.25, 0.25, 0.30])
print("PSI", round(float(np.sum((a - e) * np.log(a / e))), 4), "KL(a||e)+KL(e||a)", round(kl(a, e) + kl(e, a), 4))
shifted = rng.normal(32, 8, 20000)
print("PSI of +2 shift by bins:", {b: round(psi(base, shifted, bins=b), 4) for b in (5, 10, 20, 50)})
def psi_clip(e, a, eps):
    e, a = np.clip(e, eps, None), np.clip(a, eps, None); return float(np.sum((a - e) * np.log(a / e)))
e2 = np.array([0.5, 0.5, 0.0]); a2 = np.array([0.45, 0.5, 0.05])          # a new category appears
print("empty training bin, clip 1e-6:", round(psi_clip(e2, a2, 1e-6), 3), " clip 1e-4:", round(psi_clip(e2, a2, 1e-4), 3))

# --- 47.6 eval-set size to detect a 2-point change at 80% accuracy ---
d = 0.02; pacc = 0.80
n_unpaired = (1.96 + 0.84) ** 2 * 2 * pacc * (1 - pacc) / d ** 2
disc = 0.10                                                     # share of items where the two prompts disagree
n_paired = (1.96 * np.sqrt(disc) + 0.84 * np.sqrt(disc - d ** 2)) ** 2 / d ** 2
print(f"items for a 2-point change: unpaired {n_unpaired:,.0f} per prompt, paired (10% discordant) {n_paired:,.0f}")
print("SE at n=500, 80%:", round(np.sqrt(0.8 * 0.2 / 500) * 100, 2), "points")

# --- 47.9 Young/Daly checkpoint interval ---
import math
fails_per_day = 419 / 54 * 2000 / 16384; M = 24 * 60 / fails_per_day; C = 1.0
tau = math.sqrt(2 * M * C)
print(f"MTBF {M:,.0f} min, interval {tau:.0f} min, overhead C/tau + tau/(2M) = {C / tau + tau / (2 * M):.2%}")
print(f"Llama 3 rate: {419 / 54:.1f} interruptions/day on 16,384 GPUs, one every {24 * 60 / (419 / 54):.0f} min")
