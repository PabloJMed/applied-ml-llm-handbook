import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

rng = np.random.default_rng(0)
n = 5000
y = (rng.random(n) < 0.1).astype(int)
score = np.clip(rng.normal(0.3 + 0.35 * y, 0.18), 0, 1)

def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred, beta=1.0):
    tp, fp, fn, _ = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta ** 2
    f = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return precision, recall, f

yp = (score >= 0.5).astype(int)
print(precision_recall_f1(y, yp), skm.precision_score(y, yp), skm.recall_score(y, yp), skm.f1_score(y, yp))

def best_threshold_by_cost(y_true, scores, cost_fp, cost_fn):
    thresholds = np.unique(scores)
    costs = []
    for t in thresholds:
        tp, fp, fn, tn = confusion(y_true, (scores >= t).astype(int))
        costs.append(fp * cost_fp + fn * cost_fn)
    i = int(np.argmin(costs))
    return thresholds[i], costs[i]
print(best_threshold_by_cost(y, np.round(score, 2), cost_fp=1, cost_fn=10))

def roc_auc(y_true, scores):
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
print(roc_auc(y, score), skm.roc_auc_score(y, score))

def average_precision(y_true, scores):
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    hits = np.cumsum(y_sorted)
    precision_at_hit = hits[y_sorted == 1] / (np.flatnonzero(y_sorted == 1) + 1)
    return precision_at_hit.mean()
print(average_precision(y, score), skm.average_precision_score(y, score))

def log_loss(y_true, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
print(log_loss(y, score), skm.log_loss(y, score))

def expected_calibration_error(y_true, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y_true[mask].mean())
    return ece
print("ece raw", expected_calibration_error(y, score))

# calibrate on a held-out split
idx = rng.permutation(n); cal, test = idx[:2500], idx[2500:]
platt = LogisticRegression().fit(score[cal].reshape(-1, 1), y[cal])
p_platt = platt.predict_proba(score[test].reshape(-1, 1))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(score[cal], y[cal])
p_iso = iso.predict(score[test])
print("ece platt", expected_calibration_error(y[test], p_platt), "iso", expected_calibration_error(y[test], p_iso))
print("brier", skm.brier_score_loss(y[test], p_platt))

def correct_downsampling(p, w):
    return p / (p + (1 - p) / w)
print(correct_downsampling(np.array([0.5, 0.2]), 0.1))

# multiclass averaging
yt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
ypr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2, 0])
for avg in ["micro", "macro", "weighted"]:
    print(avg, skm.f1_score(yt, ypr, average=avg))

# regression
def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": np.mean(np.abs(err) / np.abs(y_true)),
        "WAPE": np.sum(np.abs(err)) / np.sum(np.abs(y_true)),
        "bias": np.mean(err),
    }
def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))
yt = np.array([10, 12, 0.5, 30.0]); ypd = np.array([11, 10, 2, 28.0])
print(regression_metrics(yt, ypd), pinball_loss(yt, ypd, 0.9), skm.mean_pinball_loss(yt, ypd, alpha=0.9))

# bootstrap CI clustered by user
df = pd.DataFrame({"user": rng.integers(0, 500, n), "y": y, "p": score})
def bootstrap_ci(df, metric, n_boot=500, by="user", seed=0):
    rng = np.random.default_rng(seed)
    groups = df.groupby(by).indices
    keys = np.array(list(groups))
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([groups[k] for k in sample])
        stats.append(metric(df.iloc[rows]))
    return np.percentile(stats, [2.5, 97.5])
auc = lambda d: skm.roc_auc_score(d["y"], d["p"])
print(auc(df), bootstrap_ci(df, auc, n_boot=200))

df["p_b"] = np.clip(df["p"] + rng.normal(0, 0.05, n), 0, 1)
def paired_bootstrap_diff(df, metric_a, metric_b, n_boot=500, by="user", seed=0):
    rng = np.random.default_rng(seed)
    groups = df.groupby(by).indices
    keys = np.array(list(groups))
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        s = df.iloc[rows]
        diffs.append(metric_b(s) - metric_a(s))
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])
print(paired_bootstrap_diff(df, auc, lambda d: skm.roc_auc_score(d["y"], d["p_b"]), n_boot=200))

# slicing
df["city"] = rng.choice(["BCN", "MAD", "DXB"], n)
def by_slice(df, col, metric, min_rows=200):
    rows = []
    for key, g in df.groupby(col):
        if len(g) >= min_rows and g["y"].nunique() == 2:
            rows.append({col: key, "n": len(g), "positives": int(g["y"].sum()), "metric": metric(g)})
    return pd.DataFrame(rows).sort_values("metric")
print(by_slice(df, "city", auc))

# ---- second-edition worked examples ---------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
ranked = [3, 2, 0, 1]
print(f"NDCG worked example: DCG {dcg(ranked):.3f}, ideal {dcg(sorted(ranked, reverse=True)):.3f}, NDCG {dcg(ranked)/dcg(sorted(ranked, reverse=True)):.3f}")
swapped = [0, 2, 3, 1]
print(f"best item moved to rank 3: NDCG {dcg(swapped)/dcg(sorted(swapped, reverse=True)):.3f}")
# AUC as a pairwise probability, by brute force on a tiny example
ys = np.array([1, 0, 1, 0, 0]); sc = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
pairs = [(i, j) for i in range(5) for j in range(5) if ys[i] == 1 and ys[j] == 0]
print("AUC by counting pairs:", np.mean([sc[i] > sc[j] for i, j in pairs]), "of", len(pairs), "pairs")

# ---- review-improvements worked examples ----------------------------------------------------
from math import comb
from collections import Counter
import string
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

# 4.2 PR-AUC depends on prevalence; ROC-AUC does not (same score distributions)
r = np.random.default_rng(0)
for prev in (0.01, 0.1, 0.5):
    yy = (r.random(200000) < prev).astype(int); ss = r.normal(1.5 * yy, 1)
    print(f"prevalence {prev}: ROC-AUC {skm.roc_auc_score(yy, ss):.3f}, AP {skm.average_precision_score(yy, ss):.3f}")
# trapezoidal PR area vs average precision with a coarse (3-level) score
r = np.random.default_rng(0)
yy = (r.random(100000) < 0.02).astype(int); ss = np.digitize(r.normal(1.5 * yy, 1), [0.5, 1.5])
pr, rc, _ = skm.precision_recall_curve(yy, ss)
print(f"coarse score: AP {skm.average_precision_score(yy, ss):.3f}, trapezoid {skm.auc(rc, pr):.3f}")

# 4.3 Brier is proper: pi = 0.3, predict 0.5
print("Brier identity", 0.3 * 0.5 ** 2 + 0.7 * 0.5 ** 2, (0.5 - 0.3) ** 2 + 0.3 * 0.7)
# temperature scaling
z = np.array([4.0, 1.0, 0.0])
for T in (1, 2):
    e = np.exp(z / T); print(f"T={T}: softmax {np.round(e / e.sum(), 3)}")
def fit_temperature(logits_cal, y_cal):
    def nll(T):
        zz = logits_cal / T; zz = zz - zz.max(1, keepdims=True)
        logp = zz - np.log(np.exp(zz).sum(1, keepdims=True))
        return -logp[np.arange(len(y_cal)), y_cal].mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
r = np.random.default_rng(0)
true_logits = r.normal(0, 1.5, (5000, 3)); yl = np.array([r.choice(3, p=np.exp(t) / np.exp(t).sum()) for t in true_logits])
print(f"temperature recovered for logits inflated 2.5x: {fit_temperature(2.5 * true_logits, yl):.2f}")
print("downsampling correction w=0.1:", np.round(correct_downsampling(np.array([0.5, 0.2]), 0.1), 3))

# 4.9 test-set sizing, paired SE, McNemar
print(f"n for ±1 pt at 80%: {1.96**2 * 0.16 / 0.01**2:.0f}; ±2 pt: {1.96**2 * 0.16 / 0.02**2:.0f}")
print(f"recall SE, 50 positives at 70%: {np.sqrt(0.21 / 50):.3f}")
print(f"SE of difference: unpaired {np.sqrt(2 * 0.16 / 1000):.4f}, paired (10% discordant) {np.sqrt((0.10 - 0.01 ** 2) / 1000):.4f}")
b, c = 55, 45; stat = (b - c) ** 2 / (b + c); print(f"McNemar stat {stat:.2f}, p {chi2.sf(stat, 1):.2f}")
print(f"NDCG CI half-width, 1000 queries, sd 0.3: {1.96 * 0.3 / np.sqrt(1000):.4f}")

# 4.6 ranking metrics from scratch
def recall_at_k(ranked, relevant, k): return len(set(ranked[:k]) & set(relevant)) / len(relevant)
def mrr(ranked, relevant):
    for i, item in enumerate(ranked, start=1):
        if item in relevant: return 1.0 / i
    return 0.0
def ndcg_at_k(ranked, rel, k):
    d = sum((2 ** rel.get(it, 0) - 1) / np.log2(i + 2) for i, it in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** rr - 1) / np.log2(i + 2) for i, rr in enumerate(ideal))
    return d / idcg if idcg > 0 else 0.0
ranked = ["a", "b", "c", "d"]; rel = {"a": 3, "b": 2, "c": 0, "d": 1}
print(f"ndcg {ndcg_at_k(ranked, rel, 4):.3f}, recall@2 {recall_at_k(ranked, {'b', 'd', 'e'}, 2):.3f}, mrr {mrr(ranked, {'c', 'd'}):.3f}")

# 4.10 EM and token F1
def normalise(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})
def token_f1(pred, gold):
    p_, g_ = normalise(pred).split(), normalise(gold).split()
    common = sum((Counter(p_) & Counter(g_)).values())
    if common == 0: return 0.0
    P, R = common / len(p_), common / len(g_); return 2 * P * R / (P + R)
print(f"token F1 {token_f1('The Eiffel Tower, in Paris.', 'the Eiffel Tower'):.3f}, EM {float(normalise('The Eiffel Tower!') == normalise('eiffel tower'))}")
# pass@k
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))       # running product, no overflow
print(f"pass@5 n=10 c=3: {pass_at_k(10, 3, 5):.3f} (= 1 - {comb(7, 5)}/{comb(10, 5)}), naive {1 - 0.7 ** 5:.3f}")
r = np.random.default_rng(0); cs = r.binomial(20, 0.1, 50000)
print(f"pass@10, p=0.1, n=20: true {1 - 0.9 ** 10:.3f}, unbiased {np.mean([pass_at_k(20, c, 10) for c in cs]):.3f}, naive {np.mean(1 - (1 - cs / 20) ** 10):.3f}")
# win rate with ties as half
x = np.r_[np.ones(165), 0.5 * np.ones(45), np.zeros(90)]; se = x.std() / np.sqrt(len(x))
print(f"win rate {x.mean():.3f} ± {1.96 * se:.3f} (se {se:.4f})")
# risk-coverage
corr = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0])
risk = [1 - corr[:k].mean() for k in range(1, 11)]
print("risk at coverage 10%..100%:", np.round(risk, 3), f"AURC {np.mean(risk):.3f}")
# research exercise (a): ECE of a perfectly calibrated model is biased upwards
r = np.random.default_rng(0)
for nn in (100, 1000, 10000):
    vals = {}
    for bins in (10, 30):
        e = []
        for _ in range(200):
            pp = r.random(nn); yy = (r.random(nn) < pp).astype(int); e.append(expected_calibration_error(yy, pp, bins))
        vals[bins] = np.mean(e)
    print(f"calibrated model, n={nn}: mean ECE {vals[10]:.4f} (10 bins), {vals[30]:.4f} (30 bins)")
