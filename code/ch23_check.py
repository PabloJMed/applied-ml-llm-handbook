"""Chapter 23 · Evaluating models: runnable checks (NumPy + SciPy + scikit-learn, seconds)."""
import math
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# 23.2 standard error, and clustered standard error when questions share a passage
n_passages, per_passage = 100, 5
passage_effect = rng.normal(0, 1.2, n_passages)                 # some passages are easy, some hard
logit = 0.6 + np.repeat(passage_effect, per_passage) + rng.normal(0, 0.5, n_passages * per_passage)
correct = (rng.random(n_passages * per_passage) < 1 / (1 + np.exp(-logit))).astype(float)
cluster = np.repeat(np.arange(n_passages), per_passage)
mean = correct.mean()
se_naive = correct.std(ddof=1) / math.sqrt(len(correct))
resid = correct - mean
se_clustered = math.sqrt(sum(resid[cluster == c].sum() ** 2 for c in range(n_passages))) / len(correct)
print(f"accuracy {mean:.3f}: naive SE {se_naive:.4f}, clustered SE {se_clustered:.4f} (x{se_clustered/se_naive:.2f})")

# 23.2 paired vs unpaired comparison of two models on the same 500 questions
difficulty = rng.normal(0, 1.5, 500)
a = (rng.random(500) < 1 / (1 + np.exp(-(0.4 + difficulty)))).astype(float)
b = (rng.random(500) < 1 / (1 + np.exp(-(0.55 + difficulty)))).astype(float)
diff = b - a
se_unpaired = math.sqrt(a.var(ddof=1) / 500 + b.var(ddof=1) / 500)
se_paired = diff.std(ddof=1) / math.sqrt(500)
print(f"B - A = {diff.mean():+.3f}; unpaired SE {se_unpaired:.4f}, paired SE {se_paired:.4f}; "
      f"paired 95% CI [{diff.mean()-1.96*se_paired:+.3f}, {diff.mean()+1.96*se_paired:+.3f}]")
# how many questions to detect a 2-point difference with 80% power (paired, discordance rate d)
for d in (0.10, 0.20):
    n = (1.96 + 0.84) ** 2 * d / 0.02 ** 2
    print(f"discordant-pair rate {d:.0%}: ~{n:,.0f} questions to detect a 2-point gain")

# 23.3 prompt-format variance: same model, 6 prompt templates
template_scores = 0.62 + rng.normal(0, 0.025, 6)
print("scores across 6 templates:", np.round(template_scores, 3).tolist(),
      f" range {template_scores.max()-template_scores.min():.3f}")

# 23.4 bits per byte: compare models with different tokenizers
for name, loss_nats, tokens, nbytes in (("model A (32k vocab)", 2.10, 1_000_000, 4_100_000),
                                        ("model B (128k vocab)", 2.45, 820_000, 4_100_000)):
    bpb = loss_nats * tokens / nbytes / math.log(2)
    print(f"{name}: {loss_nats} nats/token, perplexity {math.exp(loss_nats):.1f}, bits/byte {bpb:.3f}")

# 23.5 item response theory (2PL): a 100-item IRT-selected subset vs a random subset
n_models, n_items = 30, 1000
ability = rng.normal(0, 1, n_models)
disc = rng.lognormal(0, 0.4, n_items); diffi = rng.normal(0, 1.2, n_items)
P = 1 / (1 + np.exp(-disc * (ability[:, None] - diffi[None])))
R = (rng.random(P.shape) < P).astype(float)
full = R.mean(1)
def corr_subset(idx):
    return np.corrcoef(R[:, idx].mean(1), full)[0, 1]
rand_c = np.mean([corr_subset(rng.choice(n_items, 100, replace=False)) for _ in range(200)])
info = disc ** 2 * 0.25                                   # Fisher information near the item's difficulty
spread = np.argsort(diffi)
idx_irt = [spread[i] for i in np.linspace(0, n_items - 1, 400).astype(int)]
idx_irt = sorted(idx_irt, key=lambda i: -info[i])[:100]   # informative items spread across difficulty
print(f"corr with full 1,000-item score: random 100 {rand_c:.3f}, informative 100 {corr_subset(idx_irt):.3f}")

# 23.6 contamination: accuracy gap between original and rephrased items (McNemar)
orig = (rng.random(400) < 0.78).astype(int)
reph = np.where(orig == 1, (rng.random(400) < 0.85).astype(int), (rng.random(400) < 0.25).astype(int))
b01 = int(((orig == 1) & (reph == 0)).sum()); b10 = int(((orig == 0) & (reph == 1)).sum())
p = stats.binomtest(b01, b01 + b10, 0.5).pvalue
print(f"original {orig.mean():.3f} vs rephrased {reph.mean():.3f}: lost {b01}, gained {b10}, McNemar p = {p:.2g}")

# 23.7 Bradley-Terry ratings from pairwise battles (arena-style), with bootstrap CIs
true = np.array([1.2, 1.0, 0.9, 0.3, 0.0])                 # latent strengths of 5 models
battles = []
for _ in range(3000):
    i, j = rng.choice(5, 2, replace=False)
    battles.append((i, j, int(rng.random() < 1 / (1 + np.exp(-(true[i] - true[j]))))))
battles = np.array(battles)
def fit_bt(bs):
    X = np.zeros((len(bs), 5)); X[np.arange(len(bs)), bs[:, 0]] = 1; X[np.arange(len(bs)), bs[:, 1]] = -1
    m = LogisticRegression(fit_intercept=False, C=1e6).fit(X, bs[:, 2])
    s = m.coef_[0]; return 400 / math.log(10) * (s - s[-1]) + 1000     # Elo-style scale, last model = 1000
est = fit_bt(battles)
boot = np.array([fit_bt(battles[rng.integers(0, len(battles), len(battles))]) for _ in range(200)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
for k in range(5):
    print(f"model {k}: rating {est[k]:6.0f}  95% CI [{lo[k]:.0f}, {hi[k]:.0f}]")
# selection bias: testing 10 private variants of equal strength and publishing the best
noise_sd = (hi[0] - lo[0]) / 3.92
print(f"best of 10 equal variants is inflated by ~{noise_sd * np.mean(np.max(rng.normal(0,1,(5000,10)),1)):.0f} rating points")

# 23.8 position bias in an LLM judge, and the swap-and-average fix
true_pref_b = 0.55                                         # B really is better on 55% of prompts
first_bias = 0.10                                          # judge favours whichever answer is shown first
def judge(b_first):
    p = true_pref_b + (first_bias if b_first else -first_bias)
    return rng.random(10000) < p
print(f"B shown second: win rate {judge(False).mean():.3f}; B shown first: {judge(True).mean():.3f}; "
      f"averaged over both orders: {(judge(False).mean() + judge(True).mean()) / 2:.3f} (true {true_pref_b})")

# 23.11 worked comparison: 1,000 questions, models disagree on 12%, observed gain 1.8 points
import math
n, d, delta = 1000, 0.12, 0.018
se = math.sqrt((d - delta ** 2) / n); deff = 1 + (5 - 1) * 0.1
print("paired SE %.4f, 95%% CI [%.1f, %.1f] points; clustered (5 per passage, icc 0.1): SE %.4f, CI [%.1f, %.1f]"
      % (se, 100 * (delta - 1.96 * se), 100 * (delta + 1.96 * se), se * math.sqrt(deff),
         100 * (delta - 1.96 * se * math.sqrt(deff)), 100 * (delta + 1.96 * se * math.sqrt(deff))))
print("questions needed for 80%% power at this gain: %.0f (x%.1f with clustering)" % (2.8 ** 2 * d / delta ** 2, deff))
print("Elo: +50 -> %.3f, +100 -> %.3f" % (1 / (1 + 10 ** (-50 / 400)), 1 / (1 + 10 ** (-100 / 400))))
