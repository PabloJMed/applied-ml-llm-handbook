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
print("top-p 1.0 on logits [10, 0, -20, -30] keeps:", (top_p_filter(torch.tensor([10., 0., -20., -30.]), p=1.0) > float("-inf")).tolist())

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

def grpo_loss(logp_new, logp_old, rewards, mask, group_size, eps_lo=0.2, eps_hi=0.28,
              logp_ref=None, beta=0.0, scale_std=False):
    """logp_*: (B, T) token log-probs, B = prompts x group_size, rows grouped by prompt;
    rewards: (B,); mask: (B, T), 1 on generated tokens. DAPO-style token-level mean."""
    r = rewards.view(-1, group_size)                                   # (P, G)
    adv = r - r.mean(1, keepdim=True)                                  # per-group baseline
    if scale_std:                                                      # original GRPO; Dr. GRPO drops it
        adv = adv / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.reshape(-1, 1)                                           # one advantage per response, every token
    ratio = (logp_new - logp_old).exp()
    obj = torch.minimum(ratio * adv, ratio.clamp(1 - eps_lo, 1 + eps_hi) * adv)   # asymmetric clip (DAPO)
    if logp_ref is not None and beta > 0:                              # KL in the loss (k3 estimator), not in the reward
        d = logp_ref - logp_new
        obj = obj - beta * (d.exp() - d - 1)
    return -(obj * mask).sum() / mask.sum()                            # token-level mean over the whole batch

# two prompts, G = 4: an easy one everyone solves and a hard one solved once
rw = torch.tensor([1., 1., 1., 1., 0., 0., 0., 1.])
glob = rw - rw.mean(); grp = (rw.view(2, 4) - rw.view(2, 4).mean(1, keepdim=True)).reshape(-1)
print("advantages, batch mean:", [round(v, 3) for v in glob.tolist()])
print("advantages, group mean:", [round(v, 3) for v in grp.tolist()])
torch.manual_seed(0)
P_, G, T = 2, 4, 5
lp_old = -torch.rand(P_ * G, T); lp_new = lp_old + 0.05 * torch.randn(P_ * G, T)
msk = torch.ones(P_ * G, T)
print("GRPO loss (toy):", round(grpo_loss(lp_new, lp_old, rw, msk, G).item(), 4),
      "with k3 KL beta=0.04:", round(grpo_loss(lp_new, lp_old, rw, msk, G, logp_ref=lp_old, beta=0.04).item(), 4))
# at ratio = 1 the gradient w.r.t. logp_new is -adv * mask / mask.sum(): check autograd
lp = lp_old.clone().requires_grad_(True)
grpo_loss(lp, lp_old, rw, msk, G).backward()
print("ratio=1 gradient matches -A/N:", torch.allclose(lp.grad, -grp[:, None].expand(-1, T) / msk.sum()))

# beam search with optional length normalisation
def beam_search_ln(step_logprobs, beam=2, steps=3, eos=0, alpha=0.0):
    norm = lambda c: c[1] / len(c[0]) ** alpha
    beams = [((), 0.0)]
    for _ in range(steps):
        cand = []
        for seq, score in beams:
            if seq and seq[-1] == eos: cand.append((seq, score)); continue
            for tok, lp_ in enumerate(step_logprobs(seq)):
                cand.append((seq + (tok,), score + lp_))
        beams = sorted(cand, key=norm, reverse=True)[:beam]
    return beams[0]
TABLE2 = {(): [0.4, 0.35, 0.25], (1,): [0.6, 0.2, 0.2], (2,): [0.5, 0.25, 0.25]}   # token 0 = EOS
lp2 = lambda seq: [math.log(p) for p in TABLE2[seq]]
for al in (0.0, 1.0):
    sq, sc = beam_search_ln(lp2, beam=2, steps=2, alpha=al)
    print("length penalty alpha=%.0f: best %s, prob %.2f" % (al, sq, math.exp(sc)))

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

# 35.4 top 3 restaurants per city: city lives in a restaurants table, so join first
pd.DataFrame({"restaurant": ["a", "b", "c"], "city": ["Lisbon", "Lisbon", "Porto"]}).to_sql("restaurants", con, index=False)
q3 = """
SELECT * FROM (
  SELECT r.city, o.restaurant, COUNT(*) AS n_orders,
         DENSE_RANK() OVER (PARTITION BY r.city ORDER BY COUNT(*) DESC) AS rk
  FROM orders o JOIN restaurants r ON o.restaurant = r.restaurant
  GROUP BY r.city, o.restaurant) t
WHERE rk <= 3;
"""
print("35.4 top restaurants per city:", con.execute(q3).fetchall())

# 35.9 einsum and numerical stability
q_ = torch.randn(2, 3, 5, 4); k_ = torch.randn(2, 3, 7, 4)
print("einsum bhtd,bhsd->bhts equals q @ k^T:", torch.allclose(torch.einsum("bhtd,bhsd->bhts", q_, k_), q_ @ k_.transpose(-2, -1), atol=1e-6))
m16 = torch.tensor(-20.0, dtype=torch.float16)
print("fp16 margin -20: -log(sigmoid) = %s, -logsigmoid = %s" % (-torch.log(torch.sigmoid(m16)).item(), -F.logsigmoid(m16).item()))
m32 = torch.tensor(-200.0)
print("fp32 margin -200: -log(sigmoid) = %s, -logsigmoid = %s" % (-torch.log(torch.sigmoid(m32)).item(), -F.logsigmoid(m32).item()))

# 35.9 micrograd-style scalar autograd
class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents, self._backward = data, 0.0, parents, lambda: None
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward(): self.grad += out.grad; other.grad += out.grad
        out._backward = backward; return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward(): self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = backward; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def backward(): self.grad += (1 - t * t) * out.grad
        out._backward = backward; return out
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents: visit(p)
                order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()
a_, b_, c_ = Value(0.5), Value(-1.5), Value(0.25)
f_ = (a_ * b_ + c_ * a_).tanh(); f_.backward()          # a is used twice: gradients must accumulate
ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.5, -1.5, 0.25))
tf = torch.tanh(ta * tb + tc * ta); tf.backward()
print("micrograd f=%.4f grads a=%.4f b=%.4f c=%.4f; torch agrees: %s" % (f_.data, a_.grad, b_.grad, c_.grad,
      all(abs(x - y.grad.item()) < 1e-12 for x, y in ((a_.grad, ta), (b_.grad, tb), (c_.grad, tc)))))

# 35.12 causal multi-head attention with GQA and a KV cache
import torch.nn as nn
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.kv, self.dh = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d_model, bias=False)
    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2); v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        S = k.shape[2]
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        att = q @ k.transpose(-2, -1) / self.dh ** 0.5
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
        att = att.masked_fill(~mask, float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.wo(out), new_cache

torch.manual_seed(0)
attn = CausalSelfAttention(d_model=64, n_heads=8, n_kv_heads=2)
xx = torch.randn(2, 10, 64)
full, _ = attn(xx)
# reference: PyTorch's fused attention on the same projections
qr = attn.wq(xx).view(2, 10, 8, 8).transpose(1, 2)
kr = attn.wk(xx).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
vr = attn.wv(xx).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
ref = attn.wo(F.scaled_dot_product_attention(qr, kr, vr, is_causal=True).transpose(1, 2).reshape(2, 10, 64))
print("35.12 matches F.scaled_dot_product_attention:", torch.allclose(full, ref, atol=1e-5))
# prefill 6 tokens, then decode 4 one at a time with the cache: same outputs as the full pass
out_p, cache = attn(xx[:, :6]); outs = [out_p]
for t in range(6, 10):
    o_t, cache = attn(xx[:, t:t + 1], cache); outs.append(o_t)
print("35.12 prefill + cached decode equals full forward:", torch.allclose(torch.cat(outs, 1), full, atol=1e-5),
      "cache shape", tuple(cache[0].shape))
# KV-cache memory: 2 (K and V) x layers x kv_heads x d_head x tokens x bytes
def kv_bytes(layers, kv_heads, d_head, tokens, bytes_=2): return 2 * layers * kv_heads * d_head * tokens * bytes_
print("KV cache, 32 layers, d_head 128, 32k tokens, bf16: GQA 8 KV heads %.2f GB, MHA 32 heads %.2f GB"
      % (kv_bytes(32, 8, 128, 32768) / 1e9, kv_bytes(32, 32, 128, 32768) / 1e9))
print("naive score matrix at T=32k, 32 heads, bf16, one layer: %.1f GB" % (32768 ** 2 * 32 * 2 / 1e9))
print("decode step t=1000, B=1, 32 heads, d_head 128: q (1,32,1,128), K after repeat (1,32,1000,128), scores (1,32,1,1000)")

# 35.11 the debugging round: each planted bug and the check that exposes it (about a minute on CPU)
def debugging_round():
    import torch.nn as nn
    torch.manual_seed(0); torch.set_num_threads(1)
    V, T, d = 64, 32, 64
    # token stream from a first-order Markov chain with a known entropy rate
    P = torch.distributions.Dirichlet(torch.full((V,), 0.1)).sample((V,))
    stat = torch.ones(V) / V
    for _ in range(500): stat = stat @ P
    H = -(stat[:, None] * P * P.clamp_min(1e-12).log()).sum().item()
    print("entropy rate (nats/token) %.3f; ln V = %.3f" % (H, math.log(V)))
    g = torch.Generator().manual_seed(1)
    toks = [0]
    for _ in range(20000): toks.append(torch.multinomial(P[toks[-1]], 1, generator=g).item())
    toks = torch.tensor(toks)
    
    class TinyLM(nn.Module):
        def __init__(self, causal_ok=True, p_drop=0.1):
            super().__init__()
            self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(T + 1, d)
            self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.drop = nn.Dropout(p_drop); self.head = nn.Linear(d, V); self.causal_ok = causal_ok
        def forward(self, x):
            B, L = x.shape
            h = self.tok(x) + self.pos(torch.arange(L))
            q, k, v = self.qkv(self.ln1(h)).chunk(3, -1)
            att = q @ k.transpose(-2, -1) / d ** 0.5
            ones = torch.ones(L, L, dtype=torch.bool)
            mask = torch.triu(ones, 1) if self.causal_ok else torch.tril(ones, -1)
            att = att.masked_fill(mask, float("-inf"))
            h = h + self.drop(self.proj(att.softmax(-1) @ v))
            h = h + self.drop(self.mlp(self.ln2(h)))
            return self.head(h)
    
    def lm_loss(model, b, shift=True, softmax_first=False):
        x, y = (b[:, :-1], b[:, 1:]) if shift else (b[:, :-1], b[:, :-1])
        logits = model(x)
        if softmax_first: logits = logits.softmax(-1)
        return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    
    def overfit_one_batch(steps=300, zero_grad=True, **kw):
        torch.manual_seed(0)
        model = TinyLM(causal_ok=kw.pop("causal_ok", True))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        b = torch.stack([toks[i:i + T + 1] for i in range(0, 8 * (T + 1), T + 1)])
        init = lm_loss(model, b, **kw).item()
        for _ in range(steps):
            if zero_grad: opt.zero_grad()
            loss = lm_loss(model, b, **kw); loss.backward(); opt.step()
        return init, loss.item()
    print("init/final after 300 steps on one batch of 8x32:")
    print("  fixed           %.2f -> %.3f" % overfit_one_batch())
    print("  no zero_grad    %.2f -> %.3f" % overfit_one_batch(zero_grad=False))
    print("  softmax first   %.2f -> %.3f (floor ln(1+(V-1)/e) = %.3f)" % (*overfit_one_batch(softmax_first=True), math.log(1 + (V - 1) / math.e)))
    
    def train(steps=1500, **kw):
        torch.manual_seed(0)
        model = TinyLM(causal_ok=kw.pop("causal_ok", True)); opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        n = len(toks) * 9 // 10
        for s in range(steps):
            i = torch.randint(0, n - T - 1, (32,))
            b = torch.stack([toks[j:j + T + 1] for j in i])
            opt.zero_grad(); loss = lm_loss(model, b, **kw); loss.backward(); opt.step()
        return model, loss.item()
    m_ok, l_ok = train()
    m_mask, l_mask = train(causal_ok=False)
    m_shift, l_shift = train(shift=False)
    print("train loss after 1500 steps: fixed %.3f, mask reversed %.3f, no shift %.3f (entropy rate %.3f)" % (l_ok, l_mask, l_shift, H))
    # causality test: change the last token; logits at earlier positions must not move
    x = toks[:T][None].clone(); x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % V
    for name, m in (("fixed", m_ok), ("mask reversed", m_mask)):
        m.eval()
        with torch.no_grad():
            diff = (m(x)[0, :-1] - m(x2)[0, :-1]).abs().max().item()
        print("causality test, %s: max change in earlier logits %.2e" % (name, diff))
    # eval mode: dropout on gives different numbers each call
    m_ok.train()
    vb = torch.stack([toks[j:j + T + 1] for j in range(18100, 19900, T + 1)])
    with torch.no_grad():
        a = [round(lm_loss(m_ok, vb).item(), 4) for _ in range(2)]
    m_ok.eval()
    with torch.no_grad():
        e = [round(lm_loss(m_ok, vb).item(), 4) for _ in range(2)]
    print("val loss twice in train mode", a, "in eval mode", e)
    # leakage: stride-1 windows of 33 tokens, shuffled, then split 90/10
    import random
    starts = list(range(len(toks) - T)); random.Random(0).shuffle(starts)
    cut = int(0.9 * len(starts)); train_starts = set(starts[:cut])
    leaky = sum(any(j in train_starts for j in range(i - T, i + T + 1)) for i in starts[cut:]) / (len(starts) - cut)
    print("share of validation windows that share tokens with a training window: %.3f" % leaky)
debugging_round()
