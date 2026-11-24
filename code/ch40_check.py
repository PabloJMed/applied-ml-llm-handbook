import numpy as np
from collections import Counter
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA as SKPCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)

# linear regression
Xr, yr = make_regression(n_samples=500, n_features=5, noise=10, random_state=0)
def linreg_normal_eq(X, y, l2=0.0):
    Xb = np.c_[np.ones(len(X)), X]
    I = np.eye(Xb.shape[1]); I[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + l2 * I, Xb.T @ y)
w = linreg_normal_eq(Xr, yr)
sk = LinearRegression().fit(Xr, yr)
print(np.allclose(w[1:], sk.coef_), np.isclose(w[0], sk.intercept_))

# decision tree (CART, gini)
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def best_split(X, y, min_leaf=5):
    best = (None, None, gini(y))
    n = len(y)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j]); xs, ys = X[order, j], y[order]
        for i in range(min_leaf, n - min_leaf):
            if xs[i] == xs[i - 1]:
                continue
            left, right = ys[:i], ys[i:]
            score = (len(left) * gini(left) + len(right) * gini(right)) / n
            if score < best[2]:
                best = (j, (xs[i] + xs[i - 1]) / 2, score)
    return best

class Node:
    def __init__(self, pred=None, feature=None, threshold=None, left=None, right=None):
        self.pred, self.feature, self.threshold, self.left, self.right = pred, feature, threshold, left, right

def build_tree(X, y, depth=0, max_depth=4, min_leaf=5):
    pred = Counter(y).most_common(1)[0][0]
    if depth >= max_depth or len(set(y)) == 1 or len(y) < 2 * min_leaf:
        return Node(pred=pred)
    j, t, _ = best_split(X, y, min_leaf)
    if j is None:
        return Node(pred=pred)
    mask = X[:, j] <= t
    return Node(feature=j, threshold=t,
                left=build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf),
                right=build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf))

def predict_one(node, x):
    while node.pred is None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.pred

Xc, yc = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
tree = build_tree(Xtr, ytr, max_depth=4)
acc_mine = np.mean([predict_one(tree, x) == t for x, t in zip(Xte, yte)])
acc_sk = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(Xtr, ytr).score(Xte, yte)
print("tree", round(acc_mine, 3), round(acc_sk, 3))

# random forest via bagging sklearn trees
class SimpleForest:
    def __init__(self, n_trees=100, max_features="sqrt", seed=0):
        self.n_trees, self.max_features, self.seed = n_trees, max_features, seed
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for i in range(self.n_trees):
            idx = rng.integers(0, len(X), len(X))
            t = DecisionTreeClassifier(max_features=self.max_features, random_state=i).fit(X[idx], y[idx])
            self.trees.append(t)
        return self
    def predict_proba(self, X):
        return np.mean([t.predict_proba(X)[:, 1] for t in self.trees], axis=0)
rf = SimpleForest().fit(Xtr, ytr)
print("forest", round(np.mean((rf.predict_proba(Xte) > 0.5) == yte), 3),
      round(RandomForestClassifier(100, random_state=0).fit(Xtr, ytr).score(Xte, yte), 3))

# gradient boosting from scratch (squared loss)
class SimpleGBM:
    def __init__(self, n_trees=200, lr=0.1, max_depth=3):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth
    def fit(self, X, y):
        self.base = y.mean(); pred = np.full(len(y), self.base); self.trees = []
        for _ in range(self.n_trees):
            residual = y - pred
            t = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            pred += self.lr * t.predict(X); self.trees.append(t)
        return self
    def predict(self, X):
        return self.base + self.lr * np.sum([t.predict(X) for t in self.trees], axis=0)
Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr, test_size=0.3, random_state=0)
g = SimpleGBM().fit(Xtr2, ytr2)
skg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr2, ytr2)
print("gbm rmse", round(np.sqrt(np.mean((g.predict(Xte2) - yte2) ** 2)), 2), round(np.sqrt(np.mean((skg.predict(Xte2) - yte2) ** 2)), 2))

# kNN
def knn_predict(Xtrain, ytrain, Xq, k=5):
    d = ((Xq[:, None, :] - Xtrain[None]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    return np.array([Counter(ytrain[row]).most_common(1)[0][0] for row in nn])
mu, sd = Xtr.mean(0), Xtr.std(0)
print("knn", round(np.mean(knn_predict((Xtr - mu) / sd, ytr, (Xte - mu) / sd) == yte), 3))

# linear SVM with hinge loss SGD (Pegasos style)
def linear_svm(X, y, lam=0.01, epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    y_pm = np.where(y == 1, 1, -1); w = np.zeros(X.shape[1]); b = 0.0; t = 0
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            t += 1; eta = 1 / (lam * t)
            if y_pm[i] * (X[i] @ w + b) < 1:
                w = (1 - eta * lam) * w + eta * y_pm[i] * X[i]; b += eta * y_pm[i] * 0.01
            else:
                w = (1 - eta * lam) * w
    return w, b
w, b = linear_svm((Xtr - mu) / sd, ytr)
print("svm", round(np.mean((((Xte - mu) / sd) @ w + b > 0) == yte), 3))

# multinomial naive Bayes
class MultinomialNaiveBayes:
    def fit(self, X, y, alpha=1.0):
        self.classes = np.unique(y)
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        counts = np.array([X[y == c].sum(0) for c in self.classes]) + alpha
        self.log_lik = np.log(counts / counts.sum(1, keepdims=True))
        return self
    def predict(self, X):
        return self.classes[np.argmax(X @ self.log_lik.T + self.log_prior, axis=1)]
Xn = rng.poisson(1.0, size=(300, 20)); yn = (Xn[:, :5].sum(1) > Xn[:, 5:10].sum(1)).astype(int)
print("nb", np.mean(MultinomialNaiveBayes().fit(Xn[:200], yn[:200]).predict(Xn[200:]) == MultinomialNB().fit(Xn[:200], yn[:200]).predict(Xn[200:])))

# k-means
def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels
C, labels = kmeans(Xc, 3); print("kmeans sizes", np.bincount(labels))

# PCA via SVD
def pca(X, k):
    Xc_ = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc_, full_matrices=False)
    explained = S ** 2 / np.sum(S ** 2)
    return Xc_ @ Vt[:k].T, Vt[:k], explained[:k]
Z, comps, ev = pca(Xc, 2)
skp = SKPCA(2).fit(Xc)
print("pca", np.allclose(np.abs(comps), np.abs(skp.components_)), np.round(ev, 3), np.round(skp.explained_variance_ratio_, 3))

# ---------------------------------------------------------------------------
# Hand-checkable examples added in the review pass
print("\n--- hand-checkable examples ---")
# Gini for one split: parent 5/5, children 4/1 and 1/4
g = lambda p: 1 - sum(x * x for x in p)
print("gini parent", g([.5, .5]), "child", round(g([.8, .2]), 2), "decrease", round(0.5 - g([.8, .2]), 2))

# normal equation on 3 points
X3 = np.array([[1, 0], [1, 1], [1, 2.]]); y3 = np.array([1, 3, 4.])
w3 = np.linalg.solve(X3.T @ X3, X3.T @ y3)
print("normal eq w", np.round(w3, 3), "residuals", np.round(y3 - X3 @ w3, 3), "X^T r", np.round(X3.T @ (y3 - X3 @ w3), 12))

# logistic gradient = X^T (p - y)/n, check by finite differences
Xl = rng.normal(size=(50, 3)); yl = (rng.random(50) < 0.5).astype(float); wl = rng.normal(size=3)
def nll(w):
    p = 1 / (1 + np.exp(-Xl @ w)); return -np.mean(yl * np.log(p) + (1 - yl) * np.log(1 - p))
p = 1 / (1 + np.exp(-Xl @ wl)); grad = Xl.T @ (p - yl) / len(yl)
num = np.array([(nll(wl + 1e-6 * e) - nll(wl - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
print("logistic grad matches finite differences", np.allclose(grad, num, atol=1e-6))

# PCA of a 2x2 covariance
ev, evec = np.linalg.eigh(np.array([[2, 1], [1, 2.]]))
print("pca 2x2 eigenvalues", ev, "share of PC1", ev[-1] / ev.sum(), "PC1", np.round(evec[:, -1], 3))

# L1 vs L2 on an orthonormal design: soft-thresholding vs shrinkage
z = np.array([3, 0.5, -1.2]); lam = 1.0
print("lasso", np.sign(z) * np.maximum(np.abs(z) - lam, 0), "ridge", z / (1 + lam))
from sklearn.linear_model import Lasso
Q, _ = np.linalg.qr(rng.normal(size=(100, 3)))   # orthonormal columns
Xo = Q * np.sqrt(100)                              # so that X^T X / n = I
yo = Xo @ z + 0 * rng.normal(size=100)
print("sklearn Lasso alpha=1 on orthonormal design", np.round(Lasso(alpha=1.0, fit_intercept=False).fit(Xo, yo).coef_, 3))

# bagging variance rho*s2 + (1-rho)*s2/B and the OOB fraction
for rho in [0.5, 0.2]:
    for B in [10, 100]:
        print("bagging var rho", rho, "B", B, round(rho + (1 - rho) / B, 3))
print("OOB fraction n=100", round((1 - 1 / 100) ** 100, 3), "limit 1/e", round(np.exp(-1), 3))
idx_count = np.zeros(len(Xtr))
oob_votes = np.zeros(len(Xtr)); oob_n = np.zeros(len(Xtr))
rng_b = np.random.default_rng(0)
for i in range(100):
    idx = rng_b.integers(0, len(Xtr), len(Xtr)); oob = np.setdiff1d(np.arange(len(Xtr)), idx)
    t = DecisionTreeClassifier(max_features="sqrt", random_state=i).fit(Xtr[idx], ytr[idx])
    oob_votes[oob] += t.predict_proba(Xtr[oob])[:, 1]; oob_n[oob] += 1
has = oob_n > 0
print("OOB accuracy", round(np.mean((oob_votes[has] / oob_n[has] > 0.5) == ytr[has]), 3))

# XGBoost gain and leaf weights on 4 rows, log loss, base score p = 0.5, lambda = 1, gamma = 0
yx = np.array([1, 1, 1, 0]); gx = 0.5 - yx; hx = np.full(4, 0.25); L2 = 1.0
score = lambda G, H: G * G / (H + L2)
G, H = gx.sum(), hx.sum()
for k in [1, 2, 3]:
    GL, HL, GR, HR = gx[:k].sum(), hx[:k].sum(), gx[k:].sum(), hx[k:].sum()
    gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(G, H))
    print(f"xgb split after row {k}: G_L={GL} H_L={HL} G_R={GR} H_R={HR} gain={gain:.4f} "
          f"w_L={-GL/(HL+L2):.3f} w_R={-GR/(HR+L2):.3f}")

# SVM: hard margin on 3 points, dual coefficients; polynomial kernel = explicit map
from sklearn.svm import SVC
Xs = np.array([[1, 1], [-1, -1], [2, 2.]]); ys = np.array([1, -1, 1])
svc = SVC(kernel="linear", C=1e6).fit(Xs, ys)
print("svm w", np.round(svc.coef_, 3), "b", np.round(svc.intercept_, 3), "support", svc.support_,
      "alpha*y", np.round(svc.dual_coef_, 3), "margin width", round(2 / np.linalg.norm(svc.coef_), 3))
xk, zk = np.array([1, 2.]), np.array([3, 1.])
phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
print("poly kernel", (xk @ zk) ** 2, "explicit map", round(phi(xk) @ phi(zk), 6))

# GMM responsibilities: two unit-variance components at 0 and 4, equal weights
from scipy.stats import norm
for xx in [1, 2, 3]:
    a, b2 = norm.pdf(xx, 0, 1), norm.pdf(xx, 4, 1)
    print("responsibility of component 0 for x =", xx, round(a / (a + b2), 3))
# EM on 1-D data; log-likelihood never decreases
data = np.r_[rng.normal(0, 1, 300), rng.normal(4, 1, 200)]
mu_, var_, pi_ = np.array([-1., 1.]), np.array([1., 1.]), np.array([.5, .5]); lls = []
for _ in range(50):
    dens = pi_ * norm.pdf(data[:, None], mu_, np.sqrt(var_))
    lls.append(np.log(dens.sum(1)).sum()); r = dens / dens.sum(1, keepdims=True)
    Nk = r.sum(0); pi_ = Nk / len(data); mu_ = (r * data[:, None]).sum(0) / Nk
    var_ = (r * (data[:, None] - mu_) ** 2).sum(0) / Nk
print("EM means", np.round(mu_, 2), "weights", np.round(pi_, 2), "monotone", np.all(np.diff(lls) >= -1e-9))

# Shapley values for a 2-feature game
v = {(): 10, (1,): 14, (2,): 11, (1, 2): 19}
phi1 = 0.5 * ((v[(1,)] - v[()]) + (v[(1, 2)] - v[(2,)])); phi2 = 0.5 * ((v[(2,)] - v[()]) + (v[(1, 2)] - v[(1,)]))
print("shapley", phi1, phi2, "sum", phi1 + phi2, "= f(x) - base", v[(1, 2)] - v[()])

# trees cannot extrapolate
xe = np.linspace(0, 10, 200)[:, None]; ye = 2 * xe.ravel()
ge = GradientBoostingRegressor(n_estimators=200, random_state=0).fit(xe, ye)
print("GBM trained on x in [0,10], y=2x; prediction at x=20:", round(ge.predict([[20]])[0], 2), "(truth 40)")

# research exercise: tree correlation and ensemble variance vs max_features
Xv, yv = make_regression(n_samples=400, n_features=10, noise=20, random_state=1)
Xq = Xv[300:]; resv = {}
for mf in [1.0, 0.3]:
    preds = []
    for s in range(60):
        r2 = np.random.default_rng(s); idx = r2.integers(0, 300, 300)
        preds.append(DecisionTreeRegressor(max_features=mf, random_state=s).fit(Xv[idx], yv[idx]).predict(Xq))
    P = np.array(preds); s2 = P.var(0).mean()
    C = np.corrcoef(P.T.T)  # trees x trees
    rho = (C.sum() - len(P)) / (len(P) * (len(P) - 1))
    resv[mf] = (round(rho, 2), round(s2, 0), round(rho * s2 + (1 - rho) * s2 / 60, 0))
print("max_features -> (rho, tree variance, predicted variance of 60-tree average)", resv)

# logistic regression calibration: mean prediction vs base rate, with and without class weights
from sklearn.linear_model import LogisticRegression
Xcal, ycal = make_classification(5000, weights=[0.9], random_state=0)
for kw in [dict(C=1e6), dict(C=1e6, class_weight="balanced")]:
    m = LogisticRegression(max_iter=2000, **kw).fit(Xcal, ycal)
    print("logreg", kw, "base rate", round(ycal.mean(), 3), "mean prediction", round(m.predict_proba(Xcal)[:, 1].mean(), 3))
# naive Bayes double counting: one feature with likelihood ratio 3, copied k times, prior 0.5
print("NB posterior with k copies:", [round(3 ** k / (1 + 3 ** k), 3) for k in (1, 2, 3)])
