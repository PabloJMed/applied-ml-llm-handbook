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

# ---- review-improvements worked examples ----------------------------------------------------
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

# 3.2 bias-variance of a shrunk mean: mu = 2, sigma^2 = 1, n = 4
mu, s2, n4 = 2.0, 1.0, 4
for c in (1.0, 0.9, 0.8):
    b2, v = ((c - 1) * mu) ** 2, c ** 2 * s2 / n4
    print(f"shrink {c}: bias2 {b2:.4f} var {v:.4f} expected error {b2 + v + s2:.4f}")
print(f"best shrinkage {mu**2 / (mu**2 + s2 / n4):.3f}")

# 3.2 double descent with random ReLU features and the minimum-norm fit
def min_norm(P, yy, lam=0.0):
    if lam == 0 and P.shape[1] <= P.shape[0]:
        return np.linalg.lstsq(P, yy, rcond=None)[0]
    return P.T @ np.linalg.solve(P @ P.T + lam * np.eye(P.shape[0]), yy)
ps = (10, 25, 50, 90, 100, 125, 200, 400, 1000, 4000)
te_err = {p: [] for p in ps}; tr_err = {p: [] for p in ps}; ridge100 = []
for s in range(10):
    r = np.random.default_rng(s); dd = 10; nn = 100
    beta = r.normal(size=dd) / np.sqrt(dd)
    Xtr, Xte = r.normal(size=(nn, dd)), r.normal(size=(1000, dd))
    f = lambda Z: Z @ beta + 0.5 * np.tanh(2 * Z[:, 0])
    ytr = f(Xtr) + r.normal(0, 0.5, nn)
    W = r.normal(size=(dd, 4000)) / np.sqrt(dd)
    Ftr, Fte = np.maximum(0, Xtr @ W), np.maximum(0, Xte @ W)
    for p in ps:
        w = min_norm(Ftr[:, :p], ytr)
        te_err[p].append(np.mean((Fte[:, :p] @ w - f(Xte)) ** 2)); tr_err[p].append(np.mean((Ftr[:, :p] @ w - ytr) ** 2))
    w = min_norm(Ftr[:, :100], ytr, lam=10.0); ridge100.append(np.mean((Fte[:, :100] @ w - f(Xte)) ** 2))
for p in ps:
    print(f"double descent p={p}: median test {np.median(te_err[p]):.3f}, train {np.mean(tr_err[p]):.3f}")
print(f"ridge lambda 10 at p=100: median test {np.median(ridge100):.3f}")

# 3.3 focal loss is not proper: optimum for true probability 0.9
for gamma in (2, 5):
    fl = lambda q: 0.9 * (1 - q) ** gamma * -np.log(q) + 0.1 * q ** gamma * -np.log(1 - q)
    print(f"focal gamma {gamma}: optimal prediction {minimize_scalar(fl, bounds=(1e-6, 1 - 1e-6), method='bounded').x:.3f}")

# 3.4 soft-thresholding vs L2 shrinkage, lambda = 0.5
for z in (0.3, 2.0):
    print(f"z {z}: L1 {np.sign(z) * max(abs(z) - 0.5, 0):.3f}, L2 {z / 1.5:.3f}")
print(f"random search, 60 trials hit top 5%: {1 - 0.95 ** 60:.3f}")

# 3.9 class weights barely change AUC but inflate probabilities; the odds correction undoes it
r = np.random.default_rng(0); N = 50000
Xc = r.normal(size=(N, 5)); yc = (r.random(N) < 1 / (1 + np.exp(-(-5 + Xc @ np.array([1.5, -1, 0.5, 0, 0]))))).astype(int)
m0 = LogisticRegression(max_iter=1000).fit(Xc[:25000], yc[:25000])
m1 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xc[:25000], yc[:25000])
p0, p1 = m0.predict_proba(Xc[25000:])[:, 1], m1.predict_proba(Xc[25000:])[:, 1]
pi = yc[:25000].mean(); odds = p1 / (1 - p1) * pi / (1 - pi)
print(f"prevalence {yc.mean():.4f}; AUC {roc_auc_score(yc[25000:], p0):.4f} vs weighted {roc_auc_score(yc[25000:], p1):.4f}; "
      f"mean pred {p0.mean():.3f} vs {p1.mean():.3f}, corrected {np.mean(odds / (1 + odds)):.3f}, actual {yc[25000:].mean():.3f}")
def odds_correct(p, k):
    o = p / (1 - p) / k; return o / (1 + o)
print("balanced weights at 1% prevalence:", [round(odds_correct(p, 99), 4) for p in (0.5, 0.9, 0.99)])
o = 1 * (0.02 / 0.98) / (0.10 / 0.90); print(f"label shift 10% -> 2%: odds {o:.3f}, p {o / (1 + o):.3f}")

# 3.11 covariate shift: x ~ N(0,1) train, N(1,1) test, y = x^2 + noise; importance weights exp(x - 0.5)
r = np.random.default_rng(0); N = 20000
xs_tr, xs_te = r.normal(0, 1, N), r.normal(1, 1, N)
ys_tr, ys_te = xs_tr ** 2 + r.normal(0, 0.5, N), xs_te ** 2 + r.normal(0, 0.5, N)
for name, wts in (("unweighted", None), ("true weights", np.exp(xs_tr - 0.5))):
    lm = LinearRegression().fit(xs_tr[:, None], ys_tr, sample_weight=wts)
    print(f"{name}: slope {lm.coef_[0]:.2f} intercept {lm.intercept_:.2f} test MSE {np.mean((lm.predict(xs_te[:, None]) - ys_te) ** 2):.2f}")
dom = LogisticRegression().fit(np.r_[xs_tr, xs_te][:, None], np.r_[np.zeros(N), np.ones(N)])
pd_tr = dom.predict_proba(xs_tr[:, None])[:, 1]
lm = LinearRegression().fit(xs_tr[:, None], ys_tr, sample_weight=pd_tr / (1 - pd_tr))
print(f"domain-classifier weights: test MSE {np.mean((lm.predict(xs_te[:, None]) - ys_te) ** 2):.2f}; "
      f"domain AUC {roc_auc_score(np.r_[np.zeros(N), np.ones(N)], dom.predict_proba(np.r_[xs_tr, xs_te][:, None])[:, 1]):.3f}")
wts = np.exp(xs_tr - 0.5); print(f"effective sample size fraction {wts.sum() ** 2 / (wts ** 2).sum() / N:.3f} (theory {np.exp(-1):.3f})")

# Practice 4: naive target encoding of a meaningless id leaks; out-of-fold does not
r = np.random.default_rng(1)
dfn = pd.DataFrame({"id": r.integers(0, 2000, 5000), "y": (r.random(5000) < 0.3).astype(int)})
naive = dfn.groupby("id")["y"].transform("mean")
oof = target_encode_oof(dfn, "id", "y", smoothing=0)
print(f"random id: naive in-sample AUC {roc_auc_score(dfn['y'], naive):.3f}, out-of-fold AUC {roc_auc_score(dfn['y'], oof):.3f}")
