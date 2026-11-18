import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold

def mse(y, p): return np.mean((y - p) ** 2)
def mae(y, p): return np.mean(np.abs(y - p))
def huber(y, p, delta=1.0):
    r = np.abs(y - p)
    return np.mean(np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta)))
def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
def focal_loss(y, p, gamma=2.0, eps=1e-12):
    p = np.clip(p, eps, 1 - eps); pt = np.where(y == 1, p, 1 - p)
    return -np.mean((1 - pt) ** gamma * np.log(pt))
def quantile_loss(y, p, q): d = y - p; return np.mean(np.maximum(q * d, (q - 1) * d))
y = np.array([3.0, 5.0, 2.0, 30.0]); p = np.array([2.5, 5.5, 2.0, 8.0])
print(mse(y, p), mae(y, p), huber(y, p))

rng = np.random.default_rng(0)
n, d = 2000, 5
X = rng.normal(size=(n, d)); w_true = np.array([2.0, -1.0, 0.5, 0.0, 0.0])
yb = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true)))).astype(float)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def grad(w, Xb, yb, l2=0.0):
    return Xb.T @ (sigmoid(Xb @ w) - yb) / len(yb) + l2 * w

def train(optimizer, lr=0.1, epochs=20, batch=64, seed=0, **kw):
    rng = np.random.default_rng(seed)
    w = np.zeros(d); state = {}
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            g = grad(w, X[b], yb[b])
            w = optimizer(w, g, state, lr, **kw)
    return w

def sgd(w, g, state, lr): return w - lr * g
def momentum(w, g, state, lr, beta=0.9):
    state["v"] = beta * state.get("v", 0) + g
    return w - lr * state["v"]
def adam(w, g, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] = state.get("t", 0) + 1
    state["m"] = b1 * state.get("m", 0) + (1 - b1) * g
    state["v"] = b2 * state.get("v", 0) + (1 - b2) * g ** 2
    m_hat = state["m"] / (1 - b1 ** state["t"]); v_hat = state["v"] / (1 - b2 ** state["t"])
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)

for name, opt, lr in [("sgd", sgd, 0.1), ("momentum", momentum, 0.01), ("adam", adam, 0.01)]:
    w = train(opt, lr=lr)
    print(name, np.round(w, 2), round(log_loss(yb, sigmoid(X @ w)), 4))
sk = LogisticRegression(C=1e6).fit(X, yb)
print("sklearn", np.round(sk.coef_[0], 2))

# L1 vs L2
Xr = rng.normal(size=(500, 20)); wr = np.zeros(20); wr[:3] = [3, -2, 1]
yr = Xr @ wr + rng.normal(scale=0.5, size=500)
print("lasso nonzero", int(np.sum(np.abs(Lasso(alpha=0.1).fit(Xr, yr).coef_) > 1e-6)),
      "ridge nonzero", int(np.sum(np.abs(Ridge(alpha=10).fit(Xr, yr).coef_) > 1e-6)))

# out-of-fold target encoding
def target_encode_oof(df, col, target, n_splits=5, smoothing=20, seed=0):
    prior = df[target].mean()
    enc = pd.Series(np.nan, index=df.index)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(df):
        stats = df.iloc[tr].groupby(col)[target].agg(["sum", "count"])
        smoothed = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        enc.iloc[va] = df.iloc[va][col].map(smoothed).fillna(prior).values
    return enc
df = pd.DataFrame({"restaurant": rng.integers(0, 50, 3000)})
df["ordered"] = (rng.random(3000) < 0.1 + 0.005 * df["restaurant"]).astype(int)
df["rest_te"] = target_encode_oof(df, "restaurant", "ordered")
print(df.head(3), df["rest_te"].isna().sum())

# time-series split + group split
ts = TimeSeriesSplit(n_splits=3)
for tr, va in ts.split(np.arange(12)):
    print("train", tr.min(), "-", tr.max(), "valid", va.min(), "-", va.max())
groups = np.repeat(np.arange(6), 2)
for tr, va in GroupKFold(n_splits=3).split(np.arange(12), groups=groups):
    assert not set(groups[tr]) & set(groups[va])
print("group ok")

# ---- second-edition worked examples ---------------------------------------------------------
# leakage through a random split on drifting data: a time-trend feature looks predictive in-sample only
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(3)
n = 20000; t = np.arange(n)
regime = (t // 2000) % 2                                   # behaviour flips every 2,000 rows (e.g. promo periods)
x1 = rng.normal(size=n); signal = np.where(regime == 1, x1, -x1) * 0.8
yv = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)
Xv = np.column_stack([x1, t])                              # the timestamp is available as a feature
idx = rng.permutation(n); tr, te = idx[:15000], idx[15000:]
auc_random = roc_auc_score(yv[te], GradientBoostingClassifier(random_state=0).fit(Xv[tr], yv[tr]).predict_proba(Xv[te])[:, 1])
auc_time = roc_auc_score(yv[15000:], GradientBoostingClassifier(random_state=0).fit(Xv[:15000], yv[:15000]).predict_proba(Xv[15000:])[:, 1])
print(f"random split AUC {auc_random:.3f}; time split AUC {auc_time:.3f}")
# ridge shrinkage on one feature: w_ridge = (x.y)/(x.x + lambda)
xr = rng.normal(size=50); yr = 2 * xr + rng.normal(size=50)
for lam in (0, 10, 50, 200):
    print(f"lambda {lam}: ridge weight {xr @ yr / (xr @ xr + lam):.3f}")
