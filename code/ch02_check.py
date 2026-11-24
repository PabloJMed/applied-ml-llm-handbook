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

# ---- review edition: derivations, causal inference, model-evaluation statistics ----------------
from sklearn.linear_model import LinearRegression
z2 = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2
# 2.1 delta method for a ratio metric (clicks per session, randomised by user)
rng = np.random.default_rng(2); n = 5000
sess = rng.poisson(lam=rng.gamma(2, 2.5, n)) + 1
clk = rng.binomial(sess, rng.beta(2, 18, n))
R = clk.sum() / sess.sum(); mx = sess.mean()
var_delta = (clk.var(ddof=1) / n - 2 * R * np.cov(clk, sess)[0, 1] / n + R ** 2 * sess.var(ddof=1) / n) / mx ** 2
boot = [clk[i].sum() / sess[i].sum() for i in rng.integers(0, n, (2000, n))]
naive = math.sqrt(R * (1 - R) / sess.sum())
print(f"ratio metric: {sess.sum()} sessions, R = {R:.4f}; SE naive {100*naive:.3f} pts, delta {100*math.sqrt(var_delta):.3f}, "
      f"user bootstrap {100*np.std(boot, ddof=1):.3f}; naive is {1 - naive/math.sqrt(var_delta):.0%} too small")
# 2.3 sample sizes behind the figure caption; 2.4 SRM, triggered analysis, A/A tests, design effect
print("n per arm at 5% base for +10/+5/+2% relative:", [n_per_arm_proportion(0.05, m) for m in (0.10, 0.05, 0.02)])
print(f"SRM 50,500 vs 49,500: p = {srm_check(50500, 49500):.4f}")
nt = n_per_arm_proportion(0.05, 0.05)
n_dil = math.ceil(z2 * (0.05 * 0.95 + 0.05025 * 0.94975) / 0.00025 ** 2)
print(f"triggered (10% of users): {nt:,} triggered = {10*nt:,} users per arm; diluted analysis {n_dil:,} users per arm")
print(f"A/A: P(>=21 of 200 significant) = {stats.binom.sf(20, 200, 0.05):.4f}; design effect m=5, rho=0.3: {1 + 4*0.3:.1f}")
print(f"15 metrics, no effect: P(at least one significant) = {1 - 0.95**15:.2f}")
# 2.6 Holm on the BH example
def holm(p_values, alpha=0.05):
    p = np.asarray(p_values); m = len(p); order = np.argsort(p); reject = np.zeros(m, bool)
    for k, i in enumerate(order):
        if p[i] > alpha / (m - k): break
        reject[i] = True
    return reject
print("Holm rejects", holm(pv).sum(), "of", len(pv))
# 2.7 sequential testing: naive peeking vs O'Brien-Fleming over five looks
rng = np.random.default_rng(5); K = 5; sims = 200_000
inc = rng.normal(size=(sims, K)); Zs = np.cumsum(inc, 1) / np.sqrt(np.arange(1, K + 1))
obf = 2.04 * np.sqrt(K / np.arange(1, K + 1))
print(f"five looks: FPR at 1.96 = {np.mean((np.abs(Zs) > 1.96).any(1)):.3f}; OBF boundaries {obf.round(2).tolist()}, FPR {np.mean((np.abs(Zs) > obf).any(1)):.3f}")
drift = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(K)
Z1 = np.cumsum(inc + drift, 1) / np.sqrt(np.arange(1, K + 1)); hit = Z1 > obf
stop = np.where(hit.any(1), hit.argmax(1) + 1, K)
print(f"with a real effect: OBF power {hit.any(1).mean():.3f} vs single look {np.mean(Z1[:, -1] > 1.96):.3f}; mean stopping look {stop.mean():.1f}")
# 2.10 collider: novelty and rigour independent, correlated among accepted papers
rng = np.random.default_rng(3); n = 100_000
nov, rig = rng.normal(size=n), rng.normal(size=n); acc = nov + rig + rng.normal(0, 0.5, n) > 1.5
print(f"collider: corr all {np.corrcoef(nov, rig)[0,1]:.2f}, among accepted ({acc.mean():.0%}) {np.corrcoef(nov[acc], rig[acc])[0,1]:.2f}")
# 2.10 AIPW with moderate confounding, including a useless outcome model
n = 20000; x = rng.normal(size=n); t = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(int); y = 2 * x + t + rng.normal(size=n)
X1 = x.reshape(-1, 1); ps = LogisticRegression().fit(X1, t).predict_proba(X1)[:, 1]
m1 = LinearRegression().fit(X1[t == 1], y[t == 1]).predict(X1); m0 = LinearRegression().fit(X1[t == 0], y[t == 0]).predict(X1)
def aipw(m1, m0, e, t, y): return np.mean(m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e))
c1, c0 = np.full(n, y[t == 1].mean()), np.full(n, y[t == 0].mean())
print(f"AIPW: good models {aipw(m1, m0, ps, t, y):.2f}; constant outcome model {aipw(c1, c0, ps, t, y):.2f}; naive {y[t==1].mean()-y[t==0].mean():.2f}")
# 2.10 clipping bias under strong confounding (true effect 1.0), 200 repetitions
rng = np.random.default_rng(4); res = {k: [] for k in ("none", "0.01", "0.05", "aipw")}
for _ in range(200):
    x = rng.normal(size=n); e = 1 / (1 + np.exp(-4 * x)); t = (rng.random(n) < e).astype(int); y = 2 * x + t + rng.normal(size=n)
    for clip in (None, 0.01, 0.05):
        ee = e if clip is None else np.clip(e, clip, 1 - clip)
        res["none" if clip is None else str(clip)].append(np.mean(t * y / ee) - np.mean((1 - t) * y / (1 - ee)))
    X1 = x.reshape(-1, 1); m1 = LinearRegression().fit(X1[t == 1], y[t == 1]).predict(X1); m0 = LinearRegression().fit(X1[t == 0], y[t == 0]).predict(X1)
    res["aipw"].append(aipw(m1, m0, np.clip(e, 0.01, 0.99), t, y))
print(f"share with e<0.01 or e>0.99: {np.mean((e < 0.01) | (e > 0.99)):.2f}; " +
      "; ".join(f"{k}: mean {np.mean(v):.2f}, sd {np.std(v):.2f}" for k, v in res.items()))
# 2.11 model evaluation: questions needed, pairing, Bradley-Terry, temperature, seeds
print(f"SE at 70%, n=1000: {100*math.sqrt(0.21/1000):.2f} pts; questions for +/-3 pt CI on a difference {1.96**2*0.42/0.03**2:.0f}; "
      f"for 80% power on 70 vs 73 unpaired {z2*(0.21+0.73*0.27)/0.03**2:.0f}")
se_p, se_u = math.sqrt((0.15 - 0.03**2) / 1000), math.sqrt((0.21 + 0.73*0.27) / 1000)
print(f"paired SE {100*se_p:.2f} vs unpaired {100*se_u:.2f} pts; paired n for 80% power {z2*(0.15-0.03**2)/0.03**2:.0f}; "
      f"power at n=1000: paired {stats.norm.cdf(0.03/se_p - 1.96):.2f}, unpaired {stats.norm.cdf(0.03/se_u - 1.96):.2f}")
# simulate (research exercise b): A right / B right with P(A only)=0.06, P(B only)=0.09, both right 0.64, both wrong 0.21
rng = np.random.default_rng(6); reps = 10000; cells = rng.multinomial(1000, [0.64, 0.06, 0.09, 0.21], size=reps)
a_acc = (cells[:, 0] + cells[:, 1]) / 1000; b_acc = (cells[:, 0] + cells[:, 2]) / 1000
d = b_acc - a_acc; disc = (cells[:, 1] + cells[:, 2]) / 1000
z_p = d / np.sqrt((disc - d**2) / 1000); pbar = (a_acc + b_acc) / 2; z_u = d / np.sqrt(2 * pbar * (1 - pbar) / 1000)
print(f"simulated power at n=1000: paired {np.mean(np.abs(z_p) > 1.96):.2f}, unpaired {np.mean(np.abs(z_u) > 1.96):.2f}")
se15 = math.sqrt((0.12 - 0.015**2) / 1000)
print(f"+1.5 pts, 12% disagreement: SE {100*se15:.2f} pts, z {0.015/se15:.2f}, p {2*stats.norm.sf(0.015/se15):.2f}; n for 80% power {z2*(0.12-0.015**2)/0.015**2:.0f}")
w, nc = 0.55, 400; sew = math.sqrt(w * (1 - w) / nc); elo = lambda p: 400 / math.log(10) * math.log(p / (1 - p))
print(f"win rate 55% of 400: SE {100*sew:.1f} pts, CI ({w-1.96*sew:.3f}, {w+1.96*sew:.3f}); BT diff {math.log(w/(1-w)):.2f}; "
      f"Elo {elo(w):.0f} ({elo(w-1.96*sew):.0f} to {elo(w+1.96*sew):.0f})")
print(f"temperature: k=4 samples per question, SE smaller by {math.sqrt(0.21/(0.12+0.09/4)):.2f}x")
print(f"seeds per arm (sigma 0.4, delta 0.3): two-sided {z2*2*0.16/0.09:.1f}, one-sided {(stats.norm.ppf(.95)+stats.norm.ppf(.8))**2*2*0.16/0.09:.1f}")
# 2.12 deciding under uncertainty: +2% revenue, p = 0.06
zd = stats.norm.ppf(0.97); sed = 2 / zd
el = sed * stats.norm.pdf(2 / sed) - 2 * stats.norm.sf(2 / sed)
s2 = sed ** 2; pm, psd = 2 / (1 + s2), math.sqrt(s2 / (1 + s2))
print(f"+2%, p=0.06: z {zd:.2f}, SE {sed:.2f}, CI ({2-1.96*sed:.1f}, {2+1.96*sed:.1f}), P(>0) {stats.norm.cdf(2/sed):.2f}, "
      f"P(<-1) {stats.norm.cdf(-3/sed):.4f}, expected loss of shipping {el:.3f}; sceptical N(0,1) prior: mean {pm:.2f}, P(>0) {stats.norm.cdf(pm/psd):.2f}")
