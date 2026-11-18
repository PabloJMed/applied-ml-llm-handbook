import numpy as np, pandas as pd, math
from scipy import stats
rng = np.random.default_rng(0)

# z-test for two proportions
def two_prop_ztest(conv_a, n_a, conv_b, n_b):
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci = (p_b - p_a - 1.96 * se_diff, p_b - p_a + 1.96 * se_diff)
    return z, p_value, ci
print(two_prop_ztest(1000, 20000, 1080, 20000))

# sample size
def n_per_arm_proportion(p_base, mde_rel, alpha=0.05, power=0.8):
    p2 = p_base * (1 + mde_rel)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    var = p_base * (1 - p_base) + p2 * (1 - p2)
    return math.ceil((za + zb) ** 2 * var / (p2 - p_base) ** 2)
print(n_per_arm_proportion(0.05, 0.05), n_per_arm_proportion(0.05, 0.02))
def n_per_arm_mean(sd, mde_abs, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * (za + zb) ** 2 * sd ** 2 / mde_abs ** 2)
print(n_per_arm_mean(25.0, 0.5))

# CUPED
n = 20000
pre = rng.gamma(2, 10, n)
treat = rng.integers(0, 2, n)
post = 0.8 * pre + rng.normal(0, 8, n) + 0.5 * treat
theta = np.cov(post, pre)[0, 1] / np.var(pre, ddof=1)
post_cuped = post - theta * (pre - pre.mean())
for y in (post, post_cuped):
    d = y[treat == 1].mean() - y[treat == 0].mean()
    se = math.sqrt(y[treat == 1].var(ddof=1) / (treat == 1).sum() + y[treat == 0].var(ddof=1) / (treat == 0).sum())
    print(round(d, 3), round(se, 3))
print("variance reduction", round(1 - post_cuped.var() / post.var(), 3), round(np.corrcoef(post, pre)[0, 1] ** 2, 3))

# SRM
def srm_check(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    chi2, p = stats.chisquare([n_a, n_b], [total * expected_ratio, total * (1 - expected_ratio)])
    return p
print("srm p", srm_check(50310, 49690), srm_check(50100, 49900))

# Benjamini-Hochberg
def benjamini_hochberg(p_values, q=0.05):
    p = np.asarray(p_values); m = len(p)
    order = np.argsort(p)
    thresholds = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresholds
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(m, bool); reject[order[:k]] = True
    return reject
pv = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
print(benjamini_hochberg(pv), np.array(pv) < 0.05 / len(pv))

# peeking simulation: A/A test, check every day, stop at first p<0.05
def peeking_false_positive_rate(n_sims=2000, days=20, per_day=500, p=0.1):
    fp = 0
    for _ in range(n_sims):
        a = rng.random((days, per_day)) < p; b = rng.random((days, per_day)) < p
        ca, cb = a.sum(1).cumsum(), b.sum(1).cumsum(); nn = per_day * np.arange(1, days + 1)
        for d in range(days):
            if two_prop_ztest(ca[d], nn[d], cb[d], nn[d])[1] < 0.05:
                fp += 1; break
    return fp / n_sims
print("peeking FPR", peeking_false_positive_rate())

# bootstrap CI for median order value
x = rng.lognormal(3, 0.5, 2000)
boots = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
print(np.median(x), np.percentile(boots, [2.5, 97.5]))

# DiD
dfd = pd.DataFrame({"city": np.repeat(["treated", "control"], 200), "post": np.tile(np.repeat([0, 1], 100), 2)})
dfd["y"] = 10 + 2 * (dfd.city == "treated") + 1.5 * dfd.post + 0.8 * ((dfd.city == "treated") & (dfd.post == 1)) + rng.normal(0, 1, 400)
g = dfd.groupby(["city", "post"]).y.mean()
did = (g["treated", 1] - g["treated", 0]) - (g["control", 1] - g["control", 0])
print("DiD", round(did, 3))
import statsmodels.formula.api as smf
dfd["treated"] = (dfd.city == "treated").astype(int)
print(round(smf.ols("y ~ treated * post", dfd).fit().params["treated:post"], 3))

# IPW
from sklearn.linear_model import LogisticRegression
n = 20000
x1 = rng.normal(size=n)
t = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x1))).astype(int)
y = 2 * x1 + 1.0 * t + rng.normal(size=n)
naive = y[t == 1].mean() - y[t == 0].mean()
ps = LogisticRegression().fit(x1.reshape(-1, 1), t).predict_proba(x1.reshape(-1, 1))[:, 1]
ps = np.clip(ps, 0.01, 0.99)
ate = np.mean(t * y / ps) - np.mean((1 - t) * y / (1 - ps))
print("naive", round(naive, 2), "ipw", round(ate, 2))

# ---- second-edition worked examples ---------------------------------------------------------
rng = np.random.default_rng(1)
a = rng.normal(20, 5, 200); b = rng.normal(21, 5, 200)
obs = b.mean() - a.mean()
pooled = np.concatenate([a, b]); count = 0
for _ in range(10000):
    rng.shuffle(pooled)
    count += abs(pooled[200:].mean() - pooled[:200].mean()) >= abs(obs)
print(f"permutation test: observed diff {obs:.3f}, p = {count/10000:.4f}; Welch p = {stats.ttest_ind(b, a, equal_var=False).pvalue:.4f}")
# winner's curse: significant results from underpowered tests exaggerate the effect
true_effect, se = 0.1, 0.1                      # ~17% power
est = rng.normal(true_effect, se, 200000)
sig = est[np.abs(est / se) > 1.96]
print(f"power {np.mean(np.abs(est/se) > 1.96):.2f}; mean significant estimate {sig[sig > 0].mean():.3f} vs true {true_effect} "
      f"(x{sig[sig > 0].mean()/true_effect:.1f}); share of significant results with the wrong sign {np.mean(sig < 0):.3f}")
