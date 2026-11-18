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
