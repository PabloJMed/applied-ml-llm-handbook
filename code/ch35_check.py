import numpy as np, pandas as pd, sqlite3
from dataclasses import dataclass

orders = pd.DataFrame({
    "order_id": range(1, 11),
    "user_id": [1, 1, 2, 2, 2, 3, 3, 1, 4, 4],
    "restaurant": ["a", "b", "a", "a", "c", "b", "b", "a", "c", "a"],
    "ts": pd.to_datetime(["2026-10-01 12:00", "2026-10-01 12:20", "2026-10-02 19:00", "2026-10-05 20:00",
                          "2026-10-09 13:00", "2026-10-03 21:00", "2026-10-03 21:10", "2026-10-15 12:30",
                          "2026-10-07 18:00", "2026-10-20 19:30"]),
    "amount": [18.5, 22.0, 15.0, 31.0, 12.5, 27.0, 9.0, 20.0, 14.0, 25.5],
})

# 1 top-k per group
top2 = (orders.sort_values(["user_id", "amount"], ascending=[True, False])
              .groupby("user_id").head(2))
print(top2[["user_id", "amount"]].values.tolist())

# 2 share of each user's spend by restaurant (transform)
spend = orders.groupby(["user_id", "restaurant"], as_index=False)["amount"].sum()
spend["share"] = spend["amount"] / spend.groupby("user_id")["amount"].transform("sum")
print(spend.round(2).values.tolist()[:3])

# 3 days since previous order (shift within group)
o = orders.sort_values(["user_id", "ts"]).copy()
o["days_since_prev"] = o.groupby("user_id")["ts"].diff().dt.total_seconds() / 86400
print(o[["user_id", "days_since_prev"]].round(2).values.tolist()[:4])

# 4 sessionisation: new session if gap > 30 min
o["new_session"] = (o["ts"].diff() > pd.Timedelta("30min")) | (o["user_id"] != o["user_id"].shift())
o["session_id"] = o["new_session"].cumsum()
print(o.groupby("user_id")["session_id"].nunique().to_dict())

# 5 rolling 7-day spend per user (time-based window)
roll = (o.set_index("ts").groupby("user_id")["amount"].rolling("7D").sum().reset_index(name="spend_7d"))
print(roll.values.tolist()[:3])

# 6 point-in-time join
feats = pd.DataFrame({"restaurant": ["a", "a", "b", "c"], "ts": pd.to_datetime(["2026-09-30", "2026-10-06", "2026-09-30", "2026-10-01"]), "rating": [4.2, 4.5, 3.9, 4.8]}).sort_values("ts")
pit = pd.merge_asof(orders.sort_values("ts"), feats, on="ts", by="restaurant", direction="backward")
print(pit[["order_id", "restaurant", "rating"]].sort_values("order_id").values.tolist()[:5])

# 7 retention: share of users ordering again within 7 days of first order
first = orders.groupby("user_id")["ts"].min().rename("first_ts")
m = orders.join(first, on="user_id")
returned = m[(m.ts > m.first_ts) & (m.ts <= m.first_ts + pd.Timedelta("7D"))].user_id.unique()
print("7d repeat rate", len(returned) / orders.user_id.nunique())

# SQL with window functions in sqlite
con = sqlite3.connect(":memory:")
orders.assign(ts=orders.ts.astype(str)).to_sql("orders", con, index=False)
q = """
WITH ranked AS (
  SELECT user_id, order_id, amount, ts,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts) AS n,
         LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) AS prev_ts,
         SUM(amount) OVER (PARTITION BY user_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_spend
  FROM orders)
SELECT user_id, order_id, n, prev_ts, running_spend FROM ranked WHERE user_id = 2 ORDER BY ts;
"""
print(con.execute(q).fetchall())
q2 = """
WITH firsts AS (SELECT user_id, MIN(ts) AS first_ts FROM orders GROUP BY user_id)
SELECT COUNT(DISTINCT o.user_id) * 1.0 / (SELECT COUNT(*) FROM firsts) AS repeat_7d
FROM orders o JOIN firsts f ON o.user_id = f.user_id
WHERE o.ts > f.first_ts AND julianday(o.ts) <= julianday(f.first_ts) + 7;
"""
print(con.execute(q2).fetchall())

# OOP: sklearn-compatible estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.1, n_iter=500, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        Xb = np.c_[np.ones(len(X)), X]
        self.w_ = np.zeros(Xb.shape[1])
        for _ in range(self.n_iter):
            p = 1 / (1 + np.exp(-Xb @ self.w_))
            grad = Xb.T @ (p - y) / len(y)
            grad[1:] += self.l2 * self.w_[1:]
            self.w_ -= self.lr * grad
        return self
    def predict_proba(self, X):
        check_is_fitted(self, "w_")
        X = check_array(X)
        p = 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ self.w_)))
        return np.c_[1 - p, p]
    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

X, y = make_classification(n_samples=500, n_features=5, random_state=0)
print(cross_val_score(LogisticRegressionGD(), X, y, cv=5).mean())

def test_predict_proba_sums_to_one():
    m = LogisticRegressionGD().fit(X, y)
    assert np.allclose(m.predict_proba(X).sum(1), 1)
def test_learns_separable():
    Xs = np.r_[np.random.default_rng(0).normal(-2, 1, (50, 2)), np.random.default_rng(1).normal(2, 1, (50, 2))]
    ys = np.r_[np.zeros(50), np.ones(50)]
    assert (LogisticRegressionGD().fit(Xs, ys).predict(Xs) == ys).mean() > 0.95
test_predict_proba_sums_to_one(); test_learns_separable(); print("tests ok")

# 35.9 from-scratch components for research coding rounds
import math
import torch
import torch.nn.functional as F

def stable_cross_entropy(logits, targets, ignore_index=-100):
    """logits (N, V), targets (N,). Mean loss over non-ignored targets, computed with logsumexp."""
    mask = targets != ignore_index
    logits, targets = logits[mask], targets[mask]
    m = logits.max(-1, keepdim=True).values
    lse = m.squeeze(-1) + (logits - m).exp().sum(-1).log()
    return (lse - logits.gather(-1, targets[:, None]).squeeze(-1)).mean()

torch.manual_seed(0)
lg = torch.randn(6, 10) * 50                       # large logits: naive softmax would overflow in fp16
tg = torch.tensor([1, 2, -100, 4, 5, -100])
print("stable CE matches F.cross_entropy:", torch.allclose(stable_cross_entropy(lg, tg), F.cross_entropy(lg, tg), atol=1e-4))

def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
x = torch.randn(2, 8); w = torch.ones(8)
print("RMSNorm output RMS:", [round(v, 4) for v in rms_norm(x, w).pow(2).mean(-1).sqrt().tolist()])

def top_p_filter(logits, p=0.9):
    probs = logits.softmax(-1)
    sorted_p, idx = probs.sort(descending=True)
    keep = sorted_p.cumsum(-1) - sorted_p < p          # keep tokens until the mass before them reaches p
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, idx, torch.where(keep, logits.gather(-1, idx), torch.tensor(float("-inf"))))
    return filtered
logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
print("top-p 0.9 keeps:", (top_p_filter(logits).softmax(-1) > 0).tolist())

def beam_search(step_logprobs, beam=2, steps=3, eos=0):
    """step_logprobs(prefix) -> log-probs over the vocabulary. Returns the best sequence and its score."""
    beams = [((), 0.0)]
    for _ in range(steps):
        cand = []
        for seq, score in beams:
            if seq and seq[-1] == eos: cand.append((seq, score)); continue
            lp = step_logprobs(seq)
            for tok in range(len(lp)):
                cand.append((seq + (tok,), score + lp[tok]))
        beams = sorted(cand, key=lambda c: c[1], reverse=True)[:beam]
    return beams[0]
TABLE = {(): [0.1, 0.5, 0.4], (1,): [0.3, 0.35, 0.35], (2,): [0.05, 0.9, 0.05]}   # next-token probabilities
lp = lambda seq: [math.log(p) for p in TABLE[seq]]
greedy = (1, 1)                                   # greedy picks 1 (0.5) then 1 (0.35): 0.175
best = beam_search(lp, beam=2, steps=2)
print("greedy prob %.3f; beam search finds %s with prob %.3f" % (0.5 * 0.35, best[0], math.exp(best[1])))

def grpo_loss(logp_new, logp_old, rewards, mask, eps_lo=0.2, eps_hi=0.28):
    """logp_*: (G, T) token log-probs; rewards: (G,); mask: (G, T) 1 for generated tokens. Token-level mean."""
    adv = (rewards - rewards.mean())[:, None].expand_as(logp_new)
    ratio = (logp_new - logp_old).exp()
    obj = torch.minimum(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv)
    return -(obj * mask).sum() / mask.sum()
G, T = 4, 5
lp_old = -torch.rand(G, T); lp_new = lp_old + 0.05 * torch.randn(G, T)
print("GRPO loss (toy):", round(grpo_loss(lp_new, lp_old, torch.tensor([1., 0., 1., 0.]), torch.ones(G, T)).item(), 4))

# 35.10 ROC-AUC from scratch (rank / Mann-Whitney form, ties get average ranks)
import numpy as np
def roc_auc(y_true, scores):
    y = np.asarray(y_true); s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes")
    order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):                                    # average ranks over tied scores
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def roc_auc_pairs(y, s):                                 # O(n^2) definition, for testing
    pos = [b for a, b in zip(y, s) if a == 1]; neg = [b for a, b in zip(y, s) if a == 0]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0); y = rng.integers(0, 2, 500); s = np.round(rng.random(500) + 0.3 * y, 1)   # many ties
assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12 and abs(roc_auc(y, s) - roc_auc_pairs(y, s)) < 1e-12
assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0 and roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
print("35.10 roc_auc matches sklearn and the pairwise definition: %.4f" % roc_auc(y, s))
