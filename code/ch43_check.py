import numpy as np
import pandas as pd

# ---- 25.1 framing
events = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 3],
    "item_id": [10, 11, 10, 12, 11, 12, 13],
    "event":   ["order", "click", "order", "order", "click", "order", "impression"],
    "ts": pd.to_datetime(["2026-10-01", "2026-10-03", "2026-10-02", "2026-10-09",
                          "2026-10-05", "2026-10-11", "2026-10-12"]),
})
weights = {"impression": 0.0, "click": 1.0, "order": 5.0}
events["w"] = events["event"].map(weights)
cutoff = pd.Timestamp("2026-10-08")
train = events[events.ts < cutoff]
test = events[(events.ts >= cutoff) & (events.event == "order")]
interactions = (train.groupby(["user_id", "item_id"], as_index=False)["w"].sum()
                     .query("w > 0"))
print(interactions); print(test)

# ---- 25.2 CF
def item_item_scores(R, user):
    X = (R > 0).astype(float)
    norms = np.linalg.norm(X, axis=0) + 1e-9
    S = (X.T @ X) / np.outer(norms, norms)
    np.fill_diagonal(S, 0.0)
    return X[user] @ S

def implicit_als(R, k=16, alpha=20.0, reg=0.1, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    U = rng.normal(scale=0.1, size=(n_users, k))
    V = rng.normal(scale=0.1, size=(n_items, k))
    P = (R > 0).astype(float)
    C = 1.0 + alpha * R
    I = reg * np.eye(k)
    for _ in range(iters):
        for u in range(n_users):
            A = (V.T * C[u]) @ V + I
            b = (V.T * C[u]) @ P[u]
            U[u] = np.linalg.solve(A, b)
        for i in range(n_items):
            A = (U.T * C[:, i]) @ U + I
            b = (U.T * C[:, i]) @ P[:, i]
            V[i] = np.linalg.solve(A, b)
    return U, V

def recommend(U, V, R, user, n=5, exclude_seen=True):
    scores = U[user] @ V.T
    if exclude_seen:
        scores[R[user] > 0] = -np.inf
    return np.argsort(-scores)[:n]

rng = np.random.default_rng(1)
R = (rng.random((50, 30)) < 0.1) * rng.integers(1, 5, (50, 30))
U, V = implicit_als(R, k=8, iters=5)
print(recommend(U, V, R, user=0), item_item_scores(R, 0)[:5])

# ---- 25.4 two tower
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, n_ids, n_feats, dim=64):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, dim)
        self.mlp = nn.Sequential(nn.Linear(dim + n_feats, 128), nn.ReLU(), nn.Linear(128, dim))

    def forward(self, ids, feats):
        x = torch.cat([self.id_emb(ids), feats], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, n_user_feats, n_item_feats, dim=64, temperature=0.05):
        super().__init__()
        self.user_tower = Tower(n_users, n_user_feats, dim)
        self.item_tower = Tower(n_items, n_item_feats, dim)
        self.temperature = temperature

    def loss(self, user_ids, user_feats, item_ids, item_feats, item_log_q=None):
        u = self.user_tower(user_ids, user_feats)
        v = self.item_tower(item_ids, item_feats)
        logits = u @ v.T / self.temperature
        if item_log_q is not None:
            logits = logits - item_log_q.unsqueeze(0)
        same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
        same_item.fill_diagonal_(False)
        logits = logits.masked_fill(same_item, float("-inf"))
        labels = torch.arange(len(user_ids))
        return F.cross_entropy(logits, labels)

n_users, n_items, n_uf, n_if, B = 1000, 500, 8, 6, 256
user_feat_table = torch.randn(n_users, n_uf)
item_feat_table = torch.randn(n_items, n_if)
pairs_u = torch.randint(0, n_users, (20_000,))
pairs_i = (torch.rand(20_000) ** 3 * n_items).long()
counts = torch.bincount(pairs_i, minlength=n_items).float() + 1
item_log_q = torch.log(counts / counts.sum())

model = TwoTower(n_users, n_items, n_uf, n_if)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(200):
    idx = torch.randint(0, len(pairs_u), (B,))
    u_ids, i_ids = pairs_u[idx], pairs_i[idx]
    loss = model.loss(u_ids, user_feat_table[u_ids], i_ids, item_feat_table[i_ids], item_log_q[i_ids])
    opt.zero_grad(); loss.backward(); opt.step()
print("two-tower loss", loss.item())

model.eval()
with torch.no_grad():
    all_items = model.item_tower(torch.arange(n_items), item_feat_table)
    user = torch.tensor([42])
    u = model.user_tower(user, user_feat_table[user])
    top100 = torch.topk(u @ all_items.T, k=100).indices
print(top100.shape)

# ---- 25.5 ranking
class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
    def forward(self, x0, x):
        return x0 * self.lin(x) + x

class DCNRanker(nn.Module):
    def __init__(self, d_in, n_cross=3, n_tasks=2):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(d_in) for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.Linear(d_in + 64, n_tasks)
    def forward(self, x0):
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return self.heads(torch.cat([x, self.deep(x0)], dim=-1))

x = torch.randn(32, 20)
y = torch.randint(0, 2, (32, 2)).float()
m = DCNRanker(20)
l = F.binary_cross_entropy_with_logits(m(x), y)
print("dcn", l.item())

import lightgbm as lgb
nreq, per = 200, 20
Xr = rng.normal(size=(nreq * per, 5))
yr = np.clip((Xr[:, 0] + rng.normal(size=nreq * per) > 1.2).astype(int) + (Xr[:, 0] > 1.8), 0, 2)
group = [per] * nreq
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=50, learning_rate=0.05, verbose=-1)
ranker.fit(Xr[:3000], yr[:3000], group=group[:150], eval_set=[(Xr[3000:], yr[3000:])], eval_group=[group[150:]], eval_at=[10])
print("lgb ok", ranker.best_score_)

# ---- 25.6 MMR
def mmr(scores, emb, k=10, lam=0.7):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    chosen, candidates = [], list(range(len(scores)))
    while candidates and len(chosen) < k:
        def value(i):
            redundancy = max((emb[i] @ emb[j] for j in chosen), default=0.0)
            return lam * scores[i] - (1 - lam) * redundancy
        best = max(candidates, key=value)
        chosen.append(best)
        candidates.remove(best)
    return chosen
print(mmr(rng.random(30), rng.normal(size=(30, 8)), k=5))

# ---- 25.8 Thompson
rng2 = np.random.default_rng(0)
true_ctr = np.array([0.04, 0.05, 0.07])
alpha, beta = np.ones(3), np.ones(3)
for t in range(20_000):
    arm = int(np.argmax(rng2.beta(alpha, beta)))
    click = rng2.random() < true_ctr[arm]
    alpha[arm] += click
    beta[arm] += 1 - click
print("pulls per arm:", (alpha + beta - 2).astype(int))

# ---- 25.9 IPS
df = pd.DataFrame({"position": rng.integers(1, 6, 1000), "clicked": rng.integers(0, 2, 1000)})
examine_prob = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.4, 5: 0.33}
p = df["position"].map(examine_prob).clip(lower=0.1)
df["w"] = np.where(df["clicked"] == 1, 1.0 / p, 1.0)
print(df.groupby("position").w.max())

# ---- 25.10 metrics
def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / max(len(relevant), 1)

def ndcg_at_k(recommended, relevant, k):
    rel = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0

print(recall_at_k([5, 3, 9, 1], [3, 7], 3), ndcg_at_k([5, 3, 9, 1], [3, 7], 3))

def ips_value(rewards, p_new, p_logged, clip=10.0):
    w = np.minimum(p_new / p_logged, clip)
    return np.mean(w * rewards)
print(ips_value(np.array([1, 0, 1.]), np.array([.5, .2, .9]), np.array([.25, .5, .3])))

# ---------------------------------------------------------------------------
# Review-pass additions
print("\n--- review-pass checks ---")
# 43.2 ALS: the sparse user step equals the dense one; cost ratio
def implicit_als_user_step(V, VtV, items, conf, reg):
    Vu = V[items]
    A = VtV + (Vu.T * (conf - 1.0)) @ Vu + reg * np.eye(V.shape[1])
    b = Vu.T @ conf
    return np.linalg.solve(A, b)
rng2 = np.random.default_rng(3); Vt = rng2.normal(size=(30, 8)); alpha_, reg_ = 20.0, 0.1
for u in range(3):
    Cu = 1.0 + alpha_ * R[u]; Pu = (R[u] > 0).astype(float)
    dense = np.linalg.solve((Vt.T * Cu) @ Vt + reg_ * np.eye(8), (Vt.T * Cu) @ Pu)
    items = np.flatnonzero(R[u] > 0)
    sparse = implicit_als_user_step(Vt, Vt.T @ Vt, items, 1.0 + alpha_ * R[u, items], reg_)
    print("ALS user", u, "sparse == dense:", np.allclose(dense, sparse))
k_, ni_, nu_ = 64, 1_000_000, 50
print("ALS per-user cost dense", k_ * k_ * ni_, "sparse", k_ * k_ * nu_ + k_ ** 3, "ratio", round(k_ * k_ * ni_ / (k_ * k_ * nu_ + k_ ** 3)))

# 43.4 temperature: share of gradient weight on each negative
sims = np.array([0.9, 0.7, 0.5, 0.3])
for tau in (1.0, 0.2, 0.05):
    z = np.exp(sims / tau); print("tau", tau, np.round(z / z.sum(), 2))

# 43.5 calibration after negative downsampling
p_true, w_ = 0.01, 0.1
q_ = p_true / (p_true + w_ * (1 - p_true)); back = q_ / (q_ + (1 - q_) / w_)
print("downsampled prediction", round(q_, 4), "corrected", round(back, 4))
# check on simulated data: logistic model on downsampled negatives
from sklearn.linear_model import LogisticRegression
xs = rng2.normal(size=(400_000, 2)); logit = -5 + 1.0 * xs[:, 0] - 0.5 * xs[:, 1]
ys = (rng2.random(len(xs)) < 1 / (1 + np.exp(-logit))).astype(int)
keep = (ys == 1) | (rng2.random(len(xs)) < w_)
m_ = LogisticRegression().fit(xs[keep], ys[keep]); qh = m_.predict_proba(xs)[:, 1]
print("true mean rate", round(ys.mean(), 4), "sampled-model mean", round(qh.mean(), 4),
      "corrected mean", round((qh / (qh + (1 - qh) / w_)).mean(), 4))

# 43.6 MMR worked example
rel_ = np.array([0.9, 0.85, 0.7]); sim_ = np.array([[1, .95, .1], [.95, 1, .1], [.1, .1, 1]])
lam_ = 0.7; print("MMR second slot values: pizza", round(lam_ * rel_[1] - (1 - lam_) * sim_[1, 0], 3),
                  "sushi", round(lam_ * rel_[2] - (1 - lam_) * sim_[2, 0], 3))

# 43.9 position bias: one item, closed form, then a simulation of three estimators
r_, th_ = 0.5, 1 / 3
print("weighted-clicks BCE optimum", round(r_ / (r_ + 1 - th_ * r_), 3), "plain click rate", round(th_ * r_, 3))
theta = np.array([1, .7, .5, .4, .33])
def position_sim(seed, theta_used):
    g = np.random.default_rng(seed); n = 200_000
    rel = g.uniform(0.05, 0.6, size=50); item = g.integers(0, 50, n); pos = g.integers(0, 5, n)
    c = (g.random(n) < theta[pos] * rel[item]).astype(float); tu = theta_used[pos]
    cnt = np.bincount(item, minlength=50)
    naive = np.bincount(item, c, 50) / cnt
    num = np.bincount(item, c / tu, 50)
    weighted = num / (num + np.bincount(item, 1 - c, 50))
    unbiased = num / cnt
    return [np.mean(naive - rel), np.mean(weighted - rel), np.mean(unbiased - rel)]
for label_, tu_ in (("true propensities", theta), ("propensities 20% too high", np.minimum(theta * 1.2, 1.0))):
    res_ = np.array([position_sim(s_, tu_) for s_ in range(20)])
    print(label_, "mean bias (plain, weighted clicks, unbiased):", np.round(res_.mean(0), 4), "sd", np.round(res_.std(0), 4))

# unbiased pointwise loss in torch: expectation equals BCE against true relevance
def unbiased_pointwise_loss(logits, clicked, position, examine_prob, clip=10.0):
    w = (clicked / examine_prob[position]).clamp(max=clip)
    return (w * -F.logsigmoid(logits) + (1 - w) * -F.logsigmoid(-logits)).mean()
ep = torch.tensor([1.0, 0.7, 0.5, 0.4, 0.33]); g_ = torch.Generator().manual_seed(0)
n_ = 400_000; pos_ = torch.randint(0, 5, (n_,), generator=g_); rel_t = 0.5
clk = (torch.rand(n_, generator=g_) < ep[pos_] * rel_t).float()
best = min(((unbiased_pointwise_loss(torch.full((n_,), float(np.log(s / (1 - s)))), clk, pos_, ep).item(), s)
            for s in np.linspace(0.3, 0.7, 41)))
print("unbiased loss minimised at s =", round(best[1], 2), "(true relevance 0.5)")

# 43.12 embedding tables and hashing
print("1B x 64 fp32 GB", 1e9 * 64 * 4 / 1e9, "Adam state GB", 2 * 1e9 * 64 * 4 / 1e9, "row-wise AdaGrad GB", 1e9 * 4 / 1e9)
for N_, M_ in ((1e7, 1e8), (1e8, 1e8)):
    print(f"hashing {N_:.0e} IDs into {M_:.0e} buckets: collision share {1 - np.exp(-N_ / M_):.3f}")
print("95% half-width, 5% rate on 400 impressions", round(1.96 * np.sqrt(0.05 * 0.95 / 400), 4))
